import time
import datetime
import argparse
from concordia_components.unsloth_model import UnslothLanguageModel, UnslothLora
from concordia_components.simulation import SocialMediaSim
from concordia_components.entities import User, NewsSource
from concordia_components.engine import SimEngine
import numpy as np
import torch
import transformers
import names
import json
import networkx as nx
import random


def _infer_graph_type(random_graph: bool) -> str:
    return "random" if random_graph else "powerlaw_cluster"

def get_unique_name(used_names):
    while True:
        name = names.get_full_name()
        if name not in used_names:
            used_names.add(name)
            return name

import os
from concordia_components.optimized_engine import OptimizedSimEngine

def load_questions_and_options(file_path, question_number):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data = data[question_number]
    question = data['question']
    options = list(data["distributions"][list(data["distributions"].keys())[0]].keys())
    model_probabilities = data['distributions']
    return question, options, model_probabilities

if __name__ == "__main__":
    # Fix paths to be workspace relative
    WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Prevent any Hugging Face network downloads at runtime.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # Prefer shared scratch cache on clusters when available.
    scratch_dir = os.environ.get("SCRATCH")
    if scratch_dir:
        default_hf_home = os.path.join(scratch_dir, "HF-cache")
        # os.environ.setdefault("HF_HOME", default_hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", default_hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", default_hf_home)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=float, help="Start time of the job")
    parser.add_argument("--loras_path", type=str, help="Path to the LoRA models directory")
    parser.add_argument("--duration", type=float, default=14400, help="Duration of the job in seconds")
    parser.add_argument("--random_graph", action='store_true', help="Use a random graph instead of powerlaw cluster graph")
    parser.add_argument("--homophily", action='store_true', help="Assign users to graph nodes with homophily based on initial survey opinions")
    parser.add_argument("--survey_output", type=str, default="survey_results.json", help="Path to save survey results")
    parser.add_argument("--array_id", type=int, default=0, help="Array index for job differentiation")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID for logging purposes")
    parser.add_argument("--question_number", type=int, default=0, help="Index of the question to load from the questions file")
    parser.add_argument(
        "--proportions",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of model weights/proportions (e.g., --proportions 1 2 3 ...). Must match number of LoRA models.",
    )
    parser.add_argument(
        "--proportions_option",
        type=str,
        choices=["uniform", "blueprint", "average", "distribution"],
        default=None,
        help="Optional label describing which proportions preset was used.",
    )
    parser.add_argument(
        "--tweet_files",
        type=str,
        nargs="*",
        default=["trump_tweets.json"],
        help="Tweet JSON files to load as news feeds (space-separated).",
    )
    parser.add_argument("--add_survey_to_context", action="store_true", help="Add survey questions and responses to the context of the models")
    parser.add_argument("--base_model", type=str, default="marcelbinz/Llama-3.1-Minitaur-8B", help="Base model to use")
    parser.add_argument(
        "--lora_name_template",
        type=str,
        default="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}",
        help="Template used to build LoRA folder names from an index (must include {i}).",
    )
    parser.add_argument(
        "--num_loras",
        type=int,
        default=25,
        help="Number of LoRAs available at loras_path (used when --lora_indices is not provided).",
    )
    parser.add_argument(
        "--lora_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of LoRA indices to load. If omitted, all indices in [0, num_loras) are loaded.",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=1000,
        help="Number of user agents in the simulation graph.",
    )
    parser.add_argument(
        "--num_news_agents",
        type=int,
        default=1,
        help="Number of news source agents to add to the simulation.",
    )
    parser.add_argument("--visualizer_output", type=str, default=None, help="Path to save visualizer data")
    parser.add_argument("--metrics_output", type=str, default=None, help="Optional path to save echo-chamber and herd-effect metrics")

    args = parser.parse_args()

    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("HF offline mode:", os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
    print("HF cache:", os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE"))

    if args.num_agents <= 0:
        raise ValueError("--num_agents must be > 0")
    if args.num_news_agents < 0:
        raise ValueError("--num_news_agents must be >= 0")
    if args.num_loras <= 0:
        raise ValueError("--num_loras must be > 0")
    if "{i}" not in args.lora_name_template:
        raise ValueError("--lora_name_template must contain '{i}'")

    NUM_ENTITIES = args.num_agents
    VLLM_MODEL_NAME = args.base_model
    PREFIX_CACHING = False
    QUESTIONS_FILE_PATH = "divisive_questions_probabilities.json"

    # model = NoLanguageModel()
    print(f"Initializing Unsloth model: {VLLM_MODEL_NAME}")

    base_model = UnslothLanguageModel(
        model_name=VLLM_MODEL_NAME,
        load_in_4bit=False,
    )

    # Number of total actions from each cluster in the training data
    # We create agents according to these proportions
    default_proportions = np.array([
        80578, 170583, 2225632, 107699, 257398, 406647, 1014601, 45071, 73774,
        4668716, 116932, 304804, 104602, 814539, 70434, 32923, 149759, 422736,
        900057, 43742, 62087, 739085, 164314, 212932, 491830,
    ], dtype=float)

    models = []
    loaded_lora_indices = []

    if args.loras_path:
        if args.lora_indices is None:
            candidate_indices = list(range(args.num_loras))
        else:
            candidate_indices = args.lora_indices

        for i in candidate_indices:
            if i < 0:
                raise ValueError("LoRA indices must be non-negative.")
            lora_dir_name = args.lora_name_template.format(i=i)
            lora_path = os.path.join(args.loras_path, lora_dir_name)
            print(f"Loading LoRA model from: {lora_path}")
            model_i = UnslothLora(
                model_name=VLLM_MODEL_NAME,
                lora_path=lora_path,
                unsloth_language_model=base_model,
            )
            models.append(model_i)
            loaded_lora_indices.append(i)
            print(f"LoRA model {i} initialized successfully!")
    else:
        print("No LoRA path provided. Using base model for all agents.")
        models.append(base_model)

    if args.proportions is not None:
        proportions = np.array(args.proportions, dtype=float)
        if len(proportions) != len(models):
            raise ValueError(
                f"Expected {len(models)} proportions, got {len(proportions)}. "
                "Provide one value per LoRA model."
            )
        if np.any(proportions < 0):
            raise ValueError("All proportions must be non-negative.")
        if proportions.sum() == 0:
            raise ValueError("Sum of proportions must be greater than 0.")
        proportions = proportions / proportions.sum()
        print("Using user-provided proportions.")
    else:
        if len(models) == 1:
            proportions = np.array([1.0])
            print("Using base model only.")
        else:
            if loaded_lora_indices and len(default_proportions) >= (max(loaded_lora_indices) + 1):
                selected_defaults = default_proportions[loaded_lora_indices]
                proportions = selected_defaults / selected_defaults.sum()
                print("Using built-in default proportions for selected LoRA indices.")
            else:
                proportions = np.ones(len(models), dtype=float) / len(models)
                print("Using uniform proportions (no matching built-in defaults available).")
    
    # Finalize model for inference after all adapters are loaded
    base_model.finalize_inference()

    config = None  # Replace with actual config
    entities = []  # Replace with actual list of entity_lib.Entity

    model_counts = {"Model_"+str(i): 0 for i in range(len(models))}

    attributed_names = set()
    for i in range(NUM_ENTITIES):
        name = get_unique_name(attributed_names)
        model_id = np.random.choice(len(models), p=proportions)
        model = models[model_id]
        model_counts["Model_"+str(model_id)] += 1
        user = User(name=name, model=model, model_id=model_id, add_survey_to_context=args.add_survey_to_context)
        entities.append(user)

    generic_news_source_names = [
        "Global News Network",
        "World News Desk",
        "The Daily Bulletin",
        "National News Service",
        "International News Wire",
        "Civic News Report",
        "Morning News Network",
        "Public Affairs Daily",
    ]

    tweet_files_pool = list(args.tweet_files) if args.tweet_files else ["trump_tweets.json"]
    if not tweet_files_pool:
        raise ValueError("At least one tweet file must be provided.")

    for _ in range(args.num_news_agents):
        tweet_file = random.choice(tweet_files_pool)
        tweet_file_path = tweet_file
        if not os.path.isabs(tweet_file_path):
            tweet_file_path = os.path.join(WORKSPACE_ROOT, tweet_file_path)

        with open(tweet_file_path, 'r') as f:
            news_feed = json.load(f)

        source_name = random.choice(generic_news_source_names)
        news_entity = NewsSource(name=source_name, news_feed=news_feed)
        entities.append(news_entity)
        print(f"Loaded news feed from: {tweet_file_path}. Assigned source name: {source_name}")

    print("Entity distribution among models:")
    for model_name, count in model_counts.items():
        print(f"{model_name}: {count} entities")

    # Initialize Optimized Engine
    classifier_template = os.path.expanduser("~/scratch/vinai/bertweet-base-action-classifier-{n}")
    print(f"Using classifier template: {classifier_template}")

    question, options, model_probabilities = load_questions_and_options(QUESTIONS_FILE_PATH, question_number=args.question_number)

    print(f"Loaded question: {question}")

    survey_config = {
        'interval': 250,
        'question': question,
        'options': options
    }

    if args.random_graph:
        G = nx.erdos_renyi_graph(NUM_ENTITIES + args.num_news_agents, 0.05, seed=args.array_id)
    else:
        G = nx.powerlaw_cluster_graph(NUM_ENTITIES + args.num_news_agents, 14, 0.4, seed=args.array_id)
    
    # sim_engine = OptimizedSimEngine(classifier_path_template=classifier_template)
    sim_engine = SimEngine(
        survey_config=survey_config,
        graph=G,
        homophily=args.homophily,
        model_probabilities=model_probabilities,
    )

    runnable_simulation = SocialMediaSim(
        config=config,
        model=base_model,
        embedder=lambda x: np.ones(3),
        entities=entities,
        engine=sim_engine
    )

    # Adjust duration to stop 5 minutes early
    effective_duration = args.duration - 300 if args.duration else None

    start = time.perf_counter()
    print(f"Starting at time: {datetime.datetime.now()}")
    results_log = runnable_simulation.play(max_steps=2500, start_time=args.start_time, duration=effective_duration)
    end = time.perf_counter()
    print(f"Simulation completed in {end - start:.2f} seconds.")

    run_parameters = {
        "start_time": args.start_time,
        "duration": args.duration,
        "effective_duration": effective_duration,
        "array_id": args.array_id,
        "job_id": args.job_id,
        "base_model": args.base_model,
        "loras_path": args.loras_path,
        "lora_name_template": args.lora_name_template,
        "num_loras": args.num_loras,
        "lora_indices": loaded_lora_indices,
        "num_agents": args.num_agents,
        "num_news_agents": args.num_news_agents,
        "graph_type": _infer_graph_type(args.random_graph),
        "homophily": args.homophily,
        "question_number": args.question_number,
        "tweet_files": args.tweet_files,
        "add_survey_to_context": args.add_survey_to_context,
        "proportions_option": args.proportions_option,
        "proportions": proportions.tolist(),
    }

    print("Resolved run parameters:")
    print(json.dumps(run_parameters, indent=2))

    survey_results_payload = {
        "run_parameters": run_parameters,
        "survey_results": sim_engine.get_survey_results(),
    }

    # Save survey results
    with open(args.survey_output, 'w') as f:
        json.dump(survey_results_payload, f, indent=2)
    print(f"Survey results saved to {args.survey_output}")

    for thread in runnable_simulation._engine._threads:
        print(f"================== Thread ID: {thread.id}")
        for message in thread.content:
            print(f"[{message['role']}]: {message['content']}")
        print()

    threads_to_save = []
    for thread in runnable_simulation._engine._threads:
        threads_to_save.append({
            "id": thread.id,
            "messages": thread.content
        })

    output_path = os.path.join(WORKSPACE_ROOT, f"simulation_threads_{args.job_id}_{args.array_id}.json")
    with open(output_path, 'w') as f:
        json.dump(threads_to_save, f, indent=4)
    print(f"Threads saved to {output_path}")

    if args.visualizer_output:
        visualizer_data_path = args.visualizer_output
    else:
        visualizer_data_path = os.path.join(WORKSPACE_ROOT, f"visualizer_data_{args.job_id}_{args.array_id}.json")
        
    visualizer_payload = sim_engine.get_visualizer_data()
    if isinstance(visualizer_payload, dict):
        visualizer_payload["run_parameters"] = run_parameters

    with open(visualizer_data_path, 'w') as f:
        json.dump(visualizer_payload, f, indent=4)
    print(f"Visualizer data saved to {visualizer_data_path}")

    # Echo chamber metrics (from latest survey + network + exposures):
    # network_assortativity
    # mean_local_agreement
    # cross_cutting_edge_fraction
    # mean_same_option_exposure_share
    # mean_exposure_diversity
    # plus counts/distribution metadata

    # Herd effect metrics (across consecutive surveys):
    # mean_opinion_shift_rate
    # mean_current_majority_follow_rate
    # mean_consensus_gain
    # initial_consensus, final_consensus, net_consensus_change
    # per-transition details in transitions

    behavioral_metrics = sim_engine.get_behavioral_metrics()
    echo_metrics = behavioral_metrics.get("echo_chamber_metrics", {})
    herd_metrics = behavioral_metrics.get("herd_effect_metrics", {})

    print("Behavioral metrics summary:")
    print(
        "Echo-chamber: "
        f"assortativity={echo_metrics.get('network_assortativity', 'n/a')}, "
        f"cross_cutting_edge_fraction={echo_metrics.get('cross_cutting_edge_fraction', 'n/a')}, "
        f"same_option_exposure={echo_metrics.get('mean_same_option_exposure_share', 'n/a')}"
    )
    print(
        "Herd effects: "
        f"mean_shift_rate={herd_metrics.get('mean_opinion_shift_rate', 'n/a')}, "
        f"majority_follow_rate={herd_metrics.get('mean_current_majority_follow_rate', 'n/a')}, "
        f"net_consensus_change={herd_metrics.get('net_consensus_change', 'n/a')}"
    )

    if args.metrics_output:
        metrics_path = args.metrics_output
    else:
        metrics_path = os.path.join(WORKSPACE_ROOT, f"behavioral_metrics_{args.job_id}_{args.array_id}.json")

    with open(metrics_path, 'w') as f:
        json.dump(behavioral_metrics, f, indent=4)
    print(f"Behavioral metrics saved to {metrics_path}")
