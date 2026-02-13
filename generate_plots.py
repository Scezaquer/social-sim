import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description='Compare survey results between powerlaw and random groups.')
    parser.add_argument('folder', type=str, help='Folder containing the survey JSON files')
    parser.add_argument('--output', type=str, default='comparison_plot.png', help='Output plot filename')
    parser.add_argument('--target-option', type=str, help='Specific option to plot (default: first one found)')
    return parser.parse_args()

def load_data(file_path, target_option=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
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
    
    powerlaw_files = sorted(glob.glob(os.path.join(folder, "survey_powerlaw*.json")))
    random_files = sorted(glob.glob(os.path.join(folder, "survey_random*.json")))
    
    if not powerlaw_files and not random_files:
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
        first_file = (powerlaw_files + random_files)[0]
        _, _, target_option = load_data(first_file)
    
    print(f"Analyzing percentages for option: {target_option}")

    pl_steps, pl_stats, _ = process_group(powerlaw_files, target_option)
    rd_steps, rd_stats, _ = process_group(random_files, target_option)

    # Prepare table data
    table_data = []
    
    # We'll use pl_steps or rd_steps (assuming they are the same)
    steps = pl_steps if pl_steps is not None else rd_steps
    for i, step in enumerate(steps):
        row = [step]
        if pl_stats:
            row.extend([f"{pl_stats[0][i]:.2f}", f"{pl_stats[1][i]:.2f}", f"{pl_stats[4][i]:.2f}", f"{pl_stats[2][i]:.2f}", f"{pl_stats[3][i]:.2f}"])
        else:
            row.extend(["-", "-", "-", "-", "-"])
            
        if rd_stats:
            row.extend([f"{rd_stats[0][i]:.2f}", f"{rd_stats[1][i]:.2f}", f"{rd_stats[4][i]:.2f}", f"{rd_stats[2][i]:.2f}", f"{rd_stats[3][i]:.2f}"])
        else:
            row.extend(["-", "-", "-", "-", "-"])
        table_data.append(row)

    headers = ["Step", "PL Mean", "PL CI", "PL Var", "PL Max", "PL Min", "RD Mean", "RD CI", "RD Var", "RD Max", "RD Min"]
    print("\nSurvey Results Comparison (%)")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Plotting
    plt.figure(figsize=(12, 7))
    
    if pl_stats:
        means, cis, maxs, mins, vars = pl_stats
        plt.errorbar(pl_steps, means, yerr=cis, fmt='-o', label='Powerlaw Group', color='blue', capsize=5)
        plt.fill_between(pl_steps, mins, maxs, color='blue', alpha=0.1, label='Powerlaw Min-Max')

    if rd_stats:
        means, cis, maxs, mins, vars = rd_stats
        plt.errorbar(rd_steps, means, yerr=cis, fmt='-s', label='Random Group', color='red', capsize=5)
        plt.fill_between(rd_steps, mins, maxs, color='red', alpha=0.1, label='Random Min-Max')

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
