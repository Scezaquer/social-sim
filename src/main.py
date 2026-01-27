import time
import datetime
import argparse
from concordia.language_model.vllm_model import VLLMLanguageModel, VLLMLora
from concordia.language_model.transformers_model import TransformersLanguageModel, TransformersLora
from concordia.language_model.no_language_model import NoLanguageModel
from concordia_components.simulation import SocialMediaSim
from concordia_components.entities import User, NewsSource
from concordia_components.engine import SimEngine
import numpy as np
import torch
import transformers
import names
import json

def get_unique_name(used_names):
    while True:
        name = names.get_full_name()
        if name not in used_names:
            used_names.add(name)
            return name

import os
from concordia_components.optimized_engine import OptimizedSimEngine

if __name__ == "__main__":
    # Fix paths to be workspace relative
    WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=float, help="Start time of the job")
    parser.add_argument("--loras_path", type=str, help="Path to the LoRA models directory")
    parser.add_argument("--duration", type=float, default=14400, help="Duration of the job in seconds")
    args = parser.parse_args()

    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))

    NUM_ENTITIES = 1_000
    VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    PREFIX_CACHING = False

    # model = NoLanguageModel()
    print(
        f"Initializing vLLM model: {VLLM_MODEL_NAME} (PREFIX_CACHING={PREFIX_CACHING})")
    base_model = VLLMLanguageModel(
        model_name=VLLM_MODEL_NAME,
        # Additional vLLM parameters can be passed as needed
        enable_lora=True,
        max_lora_rank=64,
        enable_prefix_caching=PREFIX_CACHING,
        # max_num_batched_tokens=8192,
    )

    # Number of total actions from each cluster in the training data
    # We create agents according to these proportions
    proportions = np.array([80578, 170583, 2225632, 107699, 257398, 406647, 1014601, 45071, 73774,
                            4668716, 116932, 304804, 104602, 814539, 70434, 32923, 149759, 422736,
                            900057, 43742, 62087, 739085, 164314, 212932, 491830])
    proportions = proportions / proportions.sum()

    models = []

    for i in range(25):
        lora_path = args.loras_path + f"/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        # lora_path = f"/home/s4yor1/scratch/qwen-loras/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        print(f"Loading LoRA model from: {lora_path}")
        model_i = VLLMLora(
            model_name=VLLM_MODEL_NAME,
            lora_path=lora_path,
            vllm_language_model=base_model,
        )
        models.append(model_i)
        print(f"LoRA model {i} initialized successfully!")

    instances = None  # Replace with actual list of prefab_lib.InstanceConfig
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

    with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets.json'), 'r') as f:
        news_feed = json.load(f)
    news_entity = NewsSource(name="Global News Wire", news_feed=news_feed)
    entities.append(news_entity)

    with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets2.json'), 'r') as f:
        news_feed2 = json.load(f)
    news_entity2 = NewsSource(name="The Daily Chronicle", news_feed=news_feed2)
    entities.append(news_entity2)

    with open(os.path.join(WORKSPACE_ROOT, 'maduro_tweets3.json'), 'r') as f:
        news_feed3 = json.load(f)
    news_entity3 = NewsSource(name="World Report", news_feed=news_feed3)
    entities.append(news_entity3)

    print("Entity distribution among models:")
    for model_name, count in model_counts.items():
        print(f"{model_name}: {count} entities")

    # Initialize Optimized Engine
    classifier_template = os.path.expanduser("~/scratch/vinai/bertweet-base-action-classifier-{n}")
    print(f"Using classifier template: {classifier_template}")
    
    # sim_engine = OptimizedSimEngine(classifier_path_template=classifier_template)
    sim_engine = SimEngine()

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
    results_log = runnable_simulation.play(max_steps=1000000, start_time=args.start_time, duration=effective_duration)
    end = time.perf_counter()
    print(f"Simulation completed in {end - start:.2f} seconds.")

    for thread in runnable_simulation._engine._threads:
        print(f"================== Thread ID: {thread.id}")
        for message in thread.content:
            print(f"[{message['role']}]: {message['content']}")
        print()
