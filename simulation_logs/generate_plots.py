import os
import json
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter


def _extract_survey_entries(payload):
    if isinstance(payload, dict):
        return payload.get('survey_results', [])
    return payload

def parse_args():
    parser = argparse.ArgumentParser(description='Compare survey results between powerlaw and random groups.')
    parser.add_argument('folder', type=str, help='Folder containing the survey JSON files')
    parser.add_argument('--output', type=str, default='comparison_plot.png', help='Output plot filename')
    parser.add_argument('--target-option', type=str, help='Specific option to plot (default: first one found)')
    return parser.parse_args()

def load_data(file_path, target_option=None):
    with open(file_path, 'r') as f:
        payload = json.load(f)
        data = _extract_survey_entries(payload)
    
    results_per_step = []
    steps = []
    
    # Identify target option if not provided
    if target_option is None and data and 'results' in data[0]:
        first_step_results = list(data[0]['results'].values())
        if first_step_results:
            target_option = first_step_results[0]
    
    for entry in data:
        step = entry['step']
        results = entry['results']
        steps.append(step)
        
        counts = Counter(results.values())
        total = len(results)
        if total > 0:
            percentage = (counts.get(target_option, 0) / total) * 100
        else:
            percentage = 0
        results_per_step.append(percentage)
        
    return steps, results_per_step, target_option

def main():
    args = parse_args()
    folder = args.folder

    all_survey_files = sorted(glob.glob(os.path.join(folder, "survey_*.json")))
    group_pattern = re.compile(r"^survey_([^_]+)_(\d+(?:_\d+)*)\.json$")

    grouped_files = {}
    for file_path in all_survey_files:
        filename = os.path.basename(file_path)
        match = group_pattern.match(filename)
        if not match:
            continue
        group_name = match.group(1)
        grouped_files.setdefault(group_name, []).append(file_path)

    for group_name in grouped_files:
        grouped_files[group_name] = sorted(grouped_files[group_name])

    if not grouped_files:
        print(f"No matching files found in {folder}")
        return

    # Load all data
    def process_group(files, target_opt):
        group_data = [] # List of lists (one per file, containing percentages for each step)
        all_steps = []
        resolved_target_opt = target_opt
        
        for f in files:
            steps, percentages, opt = load_data(f, resolved_target_opt)
            if resolved_target_opt is None:
                resolved_target_opt = opt
            group_data.append(percentages)
            all_steps = steps # assuming all files have the same steps
            
        if not group_data:
            return None, None, None, resolved_target_opt
            
        # Transpose to get percentages per step across runs
        # group_data is [files][steps] -> we want [steps][files]
        num_steps = len(all_steps)
        per_step_values = [[] for _ in range(num_steps)]
        for run_percentages in group_data:
            for i, p in enumerate(run_percentages):
                if i < num_steps:
                    per_step_values[i].append(p)
        
        means = []
        cis = []
        maxs = []
        mins = []
        vars = []
        
        for values in per_step_values:
            arr = np.array(values)
            n = len(arr)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if n > 1 else 0
            variance = np.var(arr, ddof=1) if n > 1 else 0
            ci = 1.96 * (std / np.sqrt(n)) if n > 0 else 0
            
            means.append(mean)
            cis.append(ci)
            maxs.append(np.max(arr))
            mins.append(np.min(arr))
            vars.append(variance)
            
        return all_steps, (means, cis, maxs, mins, vars), resolved_target_opt

    # First determine target option from first available file
    target_option = args.target_option
    if not target_option:
        first_file = grouped_files[next(iter(grouped_files))][0]
        _, _, target_option = load_data(first_file)
    
    print(f"Analyzing percentages for option: {target_option}")

    group_results = {}
    for group_name, files in grouped_files.items():
        steps, stats, _ = process_group(files, target_option)
        if steps is not None and stats is not None:
            group_results[group_name] = (steps, stats)

    if not group_results:
        print(f"No valid survey groups found in {folder}")
        return

    # Prepare table data
    table_data = []

    # Use the longest available step list as the reference axis
    steps = max((result[0] for result in group_results.values()), key=len)
    for i, step in enumerate(steps):
        row = [step]
        for group_name in sorted(group_results.keys()):
            group_steps, group_stats = group_results[group_name]
            if i < len(group_steps):
                row.extend([
                    f"{group_stats[0][i]:.2f}",
                    f"{group_stats[1][i]:.2f}",
                    f"{group_stats[4][i]:.2f}",
                    f"{group_stats[2][i]:.2f}",
                    f"{group_stats[3][i]:.2f}",
                ])
            else:
                row.extend(["-", "-", "-", "-", "-"])
        table_data.append(row)

    headers = ["Step"]
    for group_name in sorted(group_results.keys()):
        headers.extend([
            f"{group_name} Mean",
            f"{group_name} CI",
            f"{group_name} Var",
            f"{group_name} Max",
            f"{group_name} Min",
        ])

    print("\nSurvey Results Comparison (%)")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Plotting
    plt.figure(figsize=(12, 7))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['blue', 'red', 'green', 'purple'])

    for idx, group_name in enumerate(sorted(group_results.keys())):
        group_steps, group_stats = group_results[group_name]
        means, cis, maxs, mins, vars = group_stats
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]

        plt.errorbar(
            group_steps,
            means,
            yerr=cis,
            fmt=f'-{marker}',
            label=f'{group_name} Group',
            color=color,
            capsize=5,
        )
        plt.fill_between(group_steps, mins, maxs, color=color, alpha=0.1, label=f'{group_name} Min-Max')

    plt.title(f"Comparison of Survey Evolution (Option: {target_option})")
    plt.xlabel("Simulation Step")
    plt.ylabel(f"Percentage of Responses for {target_option} (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(args.output)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
