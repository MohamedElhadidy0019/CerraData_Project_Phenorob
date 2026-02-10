#!/usr/bin/env python3
"""
Extract test_f1_macro from tensorboard logs and create comparison plots and tables.
"""

import os
import re
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Base directory for logs
LOG_BASE = "/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_diff_percentages/logs"

# Experiment configurations
EXPERIMENTS = {
    "baseline": {
        "path": os.path.join(LOG_BASE, "baseline"),
        "name": "Baseline (No Pretraining, Random Start)"
    },
    "finetune_frozen": {
        "path": os.path.join(LOG_BASE, "finetune_frozen"),
        "name": "L1 Fine-tuning (L1-Pretrained)"
    },
    "frozen_moco_encoder": {
        "path": os.path.join(LOG_BASE, "frozen_moco_encoder"),
        "name": "MoCo (Self-Supervised Pretrained)"
    }
}

def extract_percentage_from_name(dirname):
    """Extract percentage from directory name."""
    match = re.search(r'(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)
    return None

def get_best_f1_from_tensorboard(log_dir):
    """Extract the best test_f1_macro from tensorboard logs."""
    try:
        # Find event files in the log directory
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                if f.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, f))

        if not event_files:
            print(f"  No event files found in {log_dir}")
            return None

        # Use the most recent event file
        event_file = max(event_files, key=os.path.getmtime)

        # Load tensorboard data
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Get available tags
        available_tags = ea.Tags()

        # Look for test_f1_macro
        if 'scalars' in available_tags and 'test_f1_macro' in available_tags['scalars']:
            f1_values = ea.Scalars('test_f1_macro')
            # Get the maximum f1 value
            best_f1 = max([x.value for x in f1_values])
            return best_f1
        else:
            print(f"  'test_f1_macro' not found. Available tags: {available_tags.get('scalars', [])}")
            return None

    except Exception as e:
        print(f"  Error reading {log_dir}: {e}")
        return None

def extract_results():
    """Extract results from all experiments."""
    results = defaultdict(dict)

    for exp_key, exp_config in EXPERIMENTS.items():
        exp_path = exp_config["path"]
        print(f"\nProcessing {exp_config['name']}...")

        if not os.path.exists(exp_path):
            print(f"  Path not found: {exp_path}")
            continue

        # List all subdirectories
        subdirs = [d for d in os.listdir(exp_path)
                   if os.path.isdir(os.path.join(exp_path, d))]

        for subdir in sorted(subdirs):
            log_dir = os.path.join(exp_path, subdir)
            percentage = extract_percentage_from_name(subdir)

            if percentage is None:
                continue

            print(f"  Processing {subdir} ({percentage}%)...")
            f1_macro = get_best_f1_from_tensorboard(log_dir)

            if f1_macro is not None:
                results[exp_key][percentage] = f1_macro
                print(f"    Best F1-Macro: {f1_macro:.4f}")

    return results

def create_markdown_table(results):
    """Create markdown tables for results."""
    md_content = "# L2 Classification Results - Data Scaling Experiments\n\n"
    md_content += "Comparison of different training approaches with varying amounts of labeled L2 data.\n\n"

    for exp_key, exp_config in EXPERIMENTS.items():
        if exp_key not in results or not results[exp_key]:
            continue

        md_content += f"## {exp_config['name']}\n\n"
        md_content += "| Data Percentage | Test F1-Macro |\n"
        md_content += "|----------------|---------------|\n"

        # Sort by percentage
        sorted_results = sorted(results[exp_key].items())
        for percentage, f1_macro in sorted_results:
            md_content += f"| {percentage}% | {f1_macro:.4f} |\n"

        md_content += "\n"

    # Combined comparison table
    md_content += "## Comparison Across All Experiments\n\n"
    md_content += "| Data % | Baseline | Fine-tuning (Frozen) | MoCo (Frozen) |\n"
    md_content += "|--------|----------|---------------------|---------------|\n"

    # Get all unique percentages
    all_percentages = set()
    for exp_results in results.values():
        all_percentages.update(exp_results.keys())

    for pct in sorted(all_percentages):
        baseline_f1 = results.get('baseline', {}).get(pct, None)
        finetune_f1 = results.get('finetune_frozen', {}).get(pct, None)
        moco_f1 = results.get('frozen_moco_encoder', {}).get(pct, None)

        baseline_str = f"{baseline_f1:.4f}" if baseline_f1 else "-"
        finetune_str = f"{finetune_f1:.4f}" if finetune_f1 else "-"
        moco_str = f"{moco_f1:.4f}" if moco_f1 else "-"

        md_content += f"| {pct}% | {baseline_str} | {finetune_str} | {moco_str} |\n"

    return md_content

def create_comparison_plot(results):
    """Create a comparison plot of all experiments."""
    plt.figure(figsize=(12, 7))

    colors = {
        'baseline': '#e74c3c',  # Red
        'finetune_frozen': '#3498db',  # Blue
        'frozen_moco_encoder': '#2ecc71'  # Green
    }

    markers = {
        'baseline': 'o',
        'finetune_frozen': 's',
        'frozen_moco_encoder': '^'
    }

    for exp_key, exp_config in EXPERIMENTS.items():
        if exp_key not in results or not results[exp_key]:
            continue

        # Sort by percentage
        sorted_results = sorted(results[exp_key].items())
        percentages = [x[0] for x in sorted_results]
        f1_scores = [x[1] for x in sorted_results]

        plt.plot(percentages, f1_scores,
                marker=markers[exp_key],
                color=colors[exp_key],
                linewidth=2.5,
                markersize=8,
                label=exp_config['name'])

    plt.xlabel('Data Percentage (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Test F1-Macro Score', fontsize=12, fontweight='bold')
    plt.title('L2 Classification Performance vs. Training Data Size\nComparison of Training Approaches',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis to log scale for better visualization
    plt.xscale('log')
    plt.xticks(percentages, [f'{p}%' for p in percentages], rotation=0)

    # Add minor gridlines
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')

    plt.tight_layout()

    # Save plot
    output_plot = 'l2_comparison_plot.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_plot}")

    # Also save as PDF for publication quality
    output_pdf = 'l2_comparison_plot.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✅ PDF saved to: {output_pdf}")

def main():
    print("=" * 70)
    print("Extracting Test F1-Macro from Tensorboard Logs")
    print("=" * 70)

    # Extract results
    results = extract_results()

    if not results:
        print("\n❌ No results found!")
        return

    # Create markdown table
    md_content = create_markdown_table(results)
    output_md = 'l2_results_comparison.md'
    with open(output_md, 'w') as f:
        f.write(md_content)
    print(f"\n✅ Markdown table saved to: {output_md}")

    # Create comparison plot
    create_comparison_plot(results)

    print("\n" + "=" * 70)
    print("✅ All done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
