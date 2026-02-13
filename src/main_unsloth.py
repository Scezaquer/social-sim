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
        "--tweet_files",
        type=str,
        nargs="*",
        default=["trump_tweets.json"],
        help="Tweet JSON files to load as news feeds (space-separated).",
    )
    args = parser.parse_args()

    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))

    NUM_ENTITIES = 1_000
    VLLM_MODEL_NAME = "marcelbinz/Llama-3.1-Minitaur-8B"
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
    proportions = np.array([80578, 170583, 2225632, 107699, 257398, 406647, 1014601, 45071, 73774,
                            4668716, 116932, 304804, 104602, 814539, 70434, 32923, 149759, 422736,
                            900057, 43742, 62087, 739085, 164314, 212932, 491830])
    proportions = proportions / proportions.sum()

    models = []

    for i in range(25):
        # lora_path = args.loras_path + f"/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        lora_path = os.path.join(args.loras_path, f"Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}")
        # lora_path = f"/home/s4yor1/scratch/qwen-loras/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        print(f"Loading LoRA model from: {lora_path}")
        model_i = UnslothLora(
            model_name=VLLM_MODEL_NAME,
            lora_path=lora_path,
            unsloth_language_model=base_model,
        )
        models.append(model_i)
        print(f"LoRA model {i} initialized successfully!")
    
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
        user = User(name=name, model=model, model_id=model_id)
        entities.append(user)

    # with open(os.path.join(WORKSPACE_ROOT, 'poilievre_tweets.json'), 'r') as f:
    #     news_feed = json.load(f)
    # news_entity = NewsSource(name="Global News Wire", news_feed=news_feed)
    # entities.append(news_entity)

    # with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets.json'), 'r') as f:
    #     news_feed = json.load(f)
    # news_entity = NewsSource(name="Global News Wire", news_feed=news_feed)
    # entities.append(news_entity)

    # with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets2.json'), 'r') as f:
    #     news_feed2 = json.load(f)
    # news_entity2 = NewsSource(name="The Daily Chronicle", news_feed=news_feed2)
    # entities.append(news_entity2)

    # with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets3.json'), 'r') as f:
    #     news_feed3 = json.load(f)
    # news_entity3 = NewsSource(name="World Report", news_feed=news_feed3)
    # entities.append(news_entity3)

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

    for tweet_file in args.tweet_files:
        tweet_file_path = tweet_file
        if not os.path.isabs(tweet_file_path):
            tweet_file_path = os.path.join(WORKSPACE_ROOT, tweet_file_path)

        with open(tweet_file_path, 'r') as f:
            news_feed = json.load(f)

        source_name = random.choice(generic_news_source_names)
        news_entity = NewsSource(name=source_name, news_feed=news_feed)
        entities.append(news_entity)
        print(f"Loaded news feed from: {tweet_file_path}")

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
        G = nx.erdos_renyi_graph(NUM_ENTITIES + 1, 0.05, seed=args.array_id)
    else:
        G = nx.powerlaw_cluster_graph(NUM_ENTITIES + 1, 14, 0.4, seed=args.array_id)
    
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

    # Save survey results
    with open(args.survey_output, 'w') as f:
        json.dump(sim_engine.get_survey_results(), f, indent=2)
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
