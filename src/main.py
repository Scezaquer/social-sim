import time
from concordia.language_model.vllm_model import VLLMLanguageModel, VLLMLora
from concordia.language_model.no_language_model import NoLanguageModel
from concordia_components.simulation import SocialMediaSim
from concordia_components.entities import User
import numpy as np

NUM_ENTITIES = 25
VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PREFIX_CACHING = False

# model = NoLanguageModel()
print(f"Initializing vLLM model: {VLLM_MODEL_NAME} (PREFIX_CACHING={PREFIX_CACHING})")
base_model = VLLMLanguageModel(
    model_name=VLLM_MODEL_NAME,
    # Additional vLLM parameters can be passed as needed
    enable_lora=True,
    max_lora_rank=64,
    enable_prefix_caching=PREFIX_CACHING,
    # max_num_batched_tokens=8192,
)

models = []

for i in range(25):
    lora_path = f"/home/s4yor1/scratch/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
    print(f"Loading LoRA model from: {lora_path}")
    model_i = VLLMLora(
        model_name=VLLM_MODEL_NAME,
        lora_path=lora_path,
        vllm_language_model=base_model,
    )
    models.append(model_i)
    print(f"LoRA model {i} initialized successfully!")

instances = None # Replace with actual list of prefab_lib.InstanceConfig
config = None  # Replace with actual config
entities = []  # Replace with actual list of entity_lib.Entity

names = ["Theodore", "Mason", "Olivia", "Ethan", "Ava", "Liam", "Sophia", "Noah", "Isabella",
         "Lucas", "Mia", "Logan", "Charlotte", "James", "Amelia", "Benjamin", "Harper",
         "Elijah", "Evelyn", "Alexander", "Abigail"]
last_names = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
              "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee",
              "Walker", "Hall", "Allen"]

attributed_names = set()
for i in range(NUM_ENTITIES):
    while True:
        name = np.random.choice(names) + " " + np.random.choice(last_names)
        if name not in attributed_names:
            attributed_names.add(name)
            break
    user = User(
        name=name,
        model=models[NUM_ENTITIES % len(models)],
    )
    entities.append(user)

runnable_simulation = SocialMediaSim(
    config=config,
    model=base_model,
    embedder=lambda x: np.ones(3),
    entities=entities
)

start = time.perf_counter()
results_log = runnable_simulation.play(max_steps=200)
end = time.perf_counter()
print(f"Simulation completed in {end - start:.2f} seconds.")

for thread in runnable_simulation._engine._threads:
    print(f"================== Thread ID: {thread.id}")
    for message in thread.content:
        print(f"[{message['role']}]: {message['content']}")
    print()