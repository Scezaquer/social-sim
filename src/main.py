import time
import argparse
from concordia.language_model.vllm_model import VLLMLanguageModel, VLLMLora
from concordia.language_model.transformers_model import TransformersLanguageModel, TransformersLora
from concordia.language_model.no_language_model import NoLanguageModel
from concordia_components.simulation import SocialMediaSim
from concordia_components.entities import User
import numpy as np
import names

parser = argparse.ArgumentParser()
parser.add_argument("--start_time", type=float, help="Start time of the job")
parser.add_argument("--duration", type=float, default=14400, help="Duration of the job in seconds")
args = parser.parse_args()

NUM_ENTITIES = 1_000_000
VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PREFIX_CACHING = False

# model = NoLanguageModel()
print(
    f"Initializing vLLM model: {VLLM_MODEL_NAME} (PREFIX_CACHING={PREFIX_CACHING})")
base_model = TransformersLanguageModel(
    model_name=VLLM_MODEL_NAME,
    # Additional vLLM parameters can be passed as needed
    # enable_lora=True,
    # max_lora_rank=64,
    # enable_prefix_caching=PREFIX_CACHING,
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
    lora_path = f"/home/s4yor1/scratch/qwen-loras/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
    print(f"Loading LoRA model from: {lora_path}")
    model_i = TransformersLora(
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


def get_unique_name(used_names):
    while True:
        name = names.get_full_name()
        if name not in used_names:
            used_names.add(name)
            return name


attributed_names = set()
for i in range(NUM_ENTITIES):
    name = get_unique_name(attributed_names)
    model_id = np.random.choice(len(models), p=proportions)
    model = models[model_id]
    model_counts["Model_"+str(model_id)] += 1
    user = User(name=name, model=model)
    entities.append(user)

print("Entity distribution among models:")
for model_name, count in model_counts.items():
    print(f"{model_name}: {count} entities")

runnable_simulation = SocialMediaSim(
    config=config,
    model=base_model,
    embedder=lambda x: np.ones(3),
    entities=entities
)

# Adjust duration to stop 5 minutes early
effective_duration = args.duration - 300 if args.duration else None

start = time.perf_counter()
results_log = runnable_simulation.play(max_steps=1000000, start_time=args.start_time, duration=effective_duration)
end = time.perf_counter()
print(f"Simulation completed in {end - start:.2f} seconds.")

for thread in runnable_simulation._engine._threads:
    print(f"================== Thread ID: {thread.id}")
    for message in thread.content:
        print(f"[{message['role']}]: {message['content']}")
    print()
