import os
import orjson

count = 0
for file in os.listdir('.'):
    # files matching visualizer_randomized_emnlp2026question*.json
    if file.startswith('visualizer_randomized_emnlp2026question') and file.endswith('.json'):
        with open(file, 'r') as f:
            data = f.read()
        json_data = orjson.loads(data)
        if json_data["run_parameters"]["question_number"] == 28:
            # copy file to emnlp_visualizer_baseline/
            with open(os.path.join('emnlp_visualizer_baseline', file), 'w') as f:
                f.write(data)
            print(f'Copied {file} to emnlp_visualizer_baseline/')
            count += 1
print(f'Total files copied: {count}')
