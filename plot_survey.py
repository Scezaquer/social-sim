import json
import argparse
import matplotlib.pyplot as plt
from collections import Counter
import sys

def main():
    parser = argparse.ArgumentParser(description='Analyze survey results chronologically.')
    parser.add_argument('input_file', type=str, help='Path to the survey results JSON file')
    parser.add_argument('--output', type=str, default='survey_plot.png', help='Path to save the plot')
    parser.add_argument('--group-mapping', type=str, help='Optional JSON file mapping entity names to groups')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {args.input_file}.")
        sys.exit(1)

    if not data:
        print("No data found in the survey result file.")
        return

    mapping = {}
    if args.group_mapping:
        try:
            with open(args.group_mapping, 'r') as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load group mapping: {e}")

    # Extract all unique options across all steps
    all_options = set()
    for step_data in data:
        all_options.update(step_data['results'].values())
    
    sorted_options = sorted(list(all_options))
    
    steps = []
    
    # Structure: stats[group][option] = [values_per_step]
    # If no mapping, group is 'All'
    groups = set(mapping.values()) if mapping else {'All'}
    stats = {group: {opt: [] for opt in sorted_options} for group in groups}

    for step_data in data:
        step = step_data['step']
        results = step_data['results']
        steps.append(step)
        
        if mapping:
            group_results = {group: [] for group in groups}
            for name, response in results.items():
                group = mapping.get(name)
                if group in group_results:
                    group_results[group].append(response)
            
            for group in groups:
                counts = Counter(group_results[group])
                total = len(group_results[group])
                for opt in sorted_options:
                    percentage = (counts.get(opt, 0) / total * 100) if total > 0 else 0
                    stats[group][opt].append(percentage)
        else:
            counts = Counter(results.values())
            total = len(results)
            for opt in sorted_options:
                percentage = (counts.get(opt, 0) / total * 100) if total > 0 else 0
                stats['All'][opt].append(percentage)

    # Print Table
    header = f"{'Step':<10} | "
    if len(groups) > 1:
        for group in sorted(list(groups)):
            header += f"Group {group:<10} | "
    else:
        for opt in sorted_options:
            header += f"{opt[:15]:<15} | "
    print(header)
    print("-" * len(header))

    for i, step in enumerate(steps):
        row = f"{str(step):<10} | "
        if len(groups) > 1:
            # For multiple groups, this table might get too big, so we'll just print a summary or pick the top group/opt
            # Let's just print the first option's percentage for each group as a demo
            opt0 = sorted_options[0]
            for group in sorted(list(groups)):
                perc = stats[group][opt0][i]
                row += f"{opt0[:5]}: {perc:>5.1f}% | "
        else:
            for opt in sorted_options:
                perc = stats['All'][opt][i]
                row += f"{perc:>14.1f}% | "
        print(row)

    # Plotting
    try:
        plt.figure(figsize=(12, 8))
        if len(groups) > 1:
            for group in sorted(list(groups)):
                for opt in sorted_options:
                    plt.plot(steps, stats[group][opt], linestyle='-', marker='.', label=f"Group {group}: {opt}")
        else:
            for opt in sorted_options:
                plt.plot(steps, stats['All'][opt], marker='o', linewidth=2, label=opt)
        
        plt.title(f"Survey Evolution: {data[0]['question']}")
        plt.xlabel("Simulation Step")
        plt.ylabel("Percentage of Responses (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"\nDetailed plot saved to {args.output}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()

