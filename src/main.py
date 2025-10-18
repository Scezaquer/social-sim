from concordia.language_model.vllm_model import VLLMLanguageModel, VLLMLora
from concordia.language_model.no_language_model import NoLanguageModel
from concordia_components.simulation import SocialMediaSim
from concordia_components.entities import User
import numpy as np

NUM_ENTITIES = 15
VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# model = NoLanguageModel()
print(f"Initializing vLLM model: {VLLM_MODEL_NAME}")
base_model = VLLMLanguageModel(
    model_name=VLLM_MODEL_NAME,
    # Additional vLLM parameters can be passed as needed
    enable_lora=True,
    max_lora_rank=64
)

model = VLLMLora(
    model_name=VLLM_MODEL_NAME,
    lora_path="/home/s4yor1/scratch/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-6-no-focal",
    vllm_language_model=base_model,
)
print("vLLM model initialized successfully!")

instances = None # Replace with actual list of prefab_lib.InstanceConfig
config = None  # Replace with actual config
entities = []  # Replace with actual list of entity_lib.Entity

names = ["Theodore", "Mason", "Olivia", "Ethan", "Ava", "Liam", "Sophia", "Noah", "Isabella"]
last_names = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris"]

attributed_names = set()
for i in range(NUM_ENTITIES):
    while True:
        name = np.random.choice(names) + " " + np.random.choice(last_names)
        if name not in attributed_names:
            attributed_names.add(name)
            break
    user = User(
        name=name,
        model=model,
    )
    entities.append(user)

runnable_simulation = SocialMediaSim(
    config=config,
    model=model,
    embedder=lambda x: np.ones(3),
    entities=entities
)

results_log = runnable_simulation.play(max_steps=5)

for thread in runnable_simulation._engine._threads:
    print(f"Thread ID: {thread.id}")
    for message in thread.content:
        print(f"  {message['role']}: {message['content']}")
    print()