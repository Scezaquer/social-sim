import orjson
import os

count = 0
summary = []
for file in os.listdir('.'):
    if file.startswith('visualizer_randomized_emnlp2026') and file.endswith('.json'):
        with open(file, 'r') as f:
            data = f.read()
        json_data = orjson.loads(data)
        summary.append(
            {
                "file": file,
                "run_parameters": json_data["run_parameters"],
                "behavioral_metrics": json_data["behavioral_metrics"],
                "bert_real_vs_llm_classifier": json_data["bert_real_vs_llm_classifier"],
            }
        )
        count += 1
with open('summary.jsonl', 'w') as f:
    for item in summary:
        f.write(orjson.dumps(item).decode('utf-8') + '\n')
print(f'Total files copied: {count}')
