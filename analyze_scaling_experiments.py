#!/usr/bin/env python3
"""
Comprehensive analysis script for data scaling experiments.
Extracts metrics from tensorboard logs and generates plots and tables.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# ============================================================================
# EXPERIMENT CONFIGURATION
# Comment out any experiment you want to exclude from analysis
# ============================================================================
EXPERIMENTS = {
    'baseline': 'Baseline (lr=1e-3)',
    'baseline_1e4': 'Baseline (lr=1e-4)',
    'finetuning': 'Hierarchical (L1→L2)',
    'self_supervised': 'Self-Supervised (MoCo→L2)',
}

# Example: To exclude an experiment, comment it out:
# EXPERIMENTS = {
#     'baseline': 'Baseline (lr=1e-3)',
#     'baseline_1e4': 'Baseline (lr=1e-4)',
#     # 'finetuning': 'Hierarchical (L1→L2)',  # EXCLUDED
#     'self_supervised': 'Self-Supervised (MoCo→L2)',
# }


def extract_metrics_from_tensorboard(log_dir):
    """
    Extract test_f1_macro and test_f1_weighted from tensorboard logs.

    Args:
        log_dir: Path to tensorboard log directory (contains version_0/)

    Returns:
        dict: {'test_f1_macro': value, 'test_f1_weighted': value}
    """
    version_dir = os.path.join(log_dir, 'version_0')

    if not os.path.exists(version_dir):
        print(f"Warning: {version_dir} does not exist")
        return None

    # Find event file
    event_files = [f for f in os.listdir(version_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"Warning: No event files found in {version_dir}")
        return None

    # Load the event file
    ea = event_accumulator.EventAccumulator(version_dir)
    ea.Reload()

    metrics = {}

    # Extract test_f1_macro
    if 'test_f1_macro' in ea.Tags()['scalars']:
        scalars = ea.Scalars('test_f1_macro')
        if scalars:
            # Get the last (best) value
            metrics['test_f1_macro'] = scalars[-1].value

    # Extract test_f1_weighted
    if 'test_f1_weighted' in ea.Tags()['scalars']:
        scalars = ea.Scalars('test_f1_weighted')
        if scalars:
            metrics['test_f1_weighted'] = scalars[-1].value

    return metrics if metrics else None


def parse_percentage_from_dirname(dirname):
    """
    Extract percentage from directory name.
    Examples:
        - 'l2_baseline_14classes_0_5percent_20260102_161127' -> 0.5
        - 'l2_baseline_14classes_lr1e-4_0_5percent_20260104' -> 0.5
        - 'l2_baseline_14classes_10percent_20251229_215538' -> 10.0
        - 'l2_finetune_14classes_from_l1_5percent_20251231' -> 5.0 (not 1.5!)
        - 'l2_from_simclr_5percent_20260101_235914' -> 5.0
    """
    # Try pattern 1: After "classes_" with optional lr specification (for baseline and finetuning)
    # This prevents matching "l1_5percent" as "1_5" (1.5) and handles "lr1e-4"
    match = re.search(r'classes_(?:lr\d+e-?\d+_)?(?:from_l\d+_)?(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)

    # Try pattern 2: After "simclr_" or "moco_" (for self-supervised)
    match = re.search(r'(?:simclr|moco)_(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)

    return None


def collect_all_metrics(base_dir='logs_scaling_experiments'):
    """
    Collect all metrics from all experiments.
    Uses the EXPERIMENTS configuration to determine which experiments to include.

    Returns:
        dict: {method: {percentage: {metric_name: value}}}
    """
    results = defaultdict(dict)

    for method_dir, method_name in EXPERIMENTS.items():
        method_path = os.path.join(base_dir, method_dir)

        if not os.path.exists(method_path):
            print(f"Warning: {method_path} does not exist")
            continue

        # Get all experiment directories
        exp_dirs = [d for d in os.listdir(method_path)
                   if os.path.isdir(os.path.join(method_path, d)) and not d.endswith('.md')]

        for exp_dir in exp_dirs:
            percentage = parse_percentage_from_dirname(exp_dir)
            if percentage is None:
                continue

            log_path = os.path.join(method_path, exp_dir)
            metrics = extract_metrics_from_tensorboard(log_path)

            if metrics:
                results[method_name][percentage] = metrics
                print(f"✓ {method_name} @ {percentage}%: F1-macro={metrics.get('test_f1_macro', 'N/A'):.4f}")
            else:
                print(f"✗ {method_name} @ {percentage}%: No metrics found")

    return results


def plot_individual_method(method_name, data, metric='test_f1_macro', output_dir='analysis_results'):
    """
    Create a plot for a single method showing percentage vs metric.
    """
    if not data:
        print(f"No data for {method_name}")
        return

    # Sort by percentage
    percentages = sorted(data.keys())
    values = [data[pct].get(metric, np.nan) for pct in percentages]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, values, 'o-', linewidth=2, markersize=8, label=method_name)

    # Add value labels on points
    for pct, val in zip(percentages, values):
        if not np.isnan(val):
            plt.annotate(f'{val:.3f}', (pct, val), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

    plt.xlabel('Data Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f'{method_name}: {metric.replace("_", " ").title()} vs Data Percentage',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Use log scale for x-axis to better show low percentages
    plt.xscale('log')
    plt.xticks(percentages, [f'{p}%' for p in percentages], rotation=45)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_method_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('→', 'to')
    filename = f'{safe_method_name}_{metric}.png'
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def plot_combined_comparison(all_results, metric='test_f1_macro', output_dir='analysis_results'):
    """
    Create a combined plot comparing all methods.
    """
    plt.figure(figsize=(12, 7))

    colors = {
        'Baseline (Random Init)': '#e74c3c',
        'Hierarchical (L1→L2)': '#3498db',
        'Self-Supervised (MoCo→L2)': '#2ecc71'
    }

    markers = {
        'Baseline (Random Init)': 'o',
        'Hierarchical (L1→L2)': 's',
        'Self-Supervised (MoCo→L2)': '^'
    }

    for method_name, data in all_results.items():
        if not data:
            continue

        percentages = sorted(data.keys())
        values = [data[pct].get(metric, np.nan) for pct in percentages]

        plt.plot(percentages, values,
                marker=markers.get(method_name, 'o'),
                linewidth=2.5,
                markersize=10,
                label=method_name,
                color=colors.get(method_name),
                alpha=0.8)

    plt.xlabel('Data Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f'Comparison of All Methods: {metric.replace("_", " ").title()} vs Data Percentage',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')

    # Use log scale for x-axis
    plt.xscale('log')

    # Get all unique percentages across all methods
    all_percentages = sorted(set(pct for data in all_results.values() for pct in data.keys()))
    plt.xticks(all_percentages, [f'{p}%' for p in all_percentages], rotation=45)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'combined_comparison_{metric}.png'
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_comparison_table(all_results, metric='test_f1_macro', output_dir='analysis_results'):
    """
    Create a comparison table with bold highlighting for best values.
    """
    # Get all unique percentages
    all_percentages = sorted(set(pct for data in all_results.values() for pct in data.keys()))

    # Create DataFrame
    table_data = []
    for pct in all_percentages:
        row = {'Data %': f'{pct}%'}
        for method_name in all_results.keys():
            value = all_results[method_name].get(pct, {}).get(metric, np.nan)
            row[method_name] = value
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save as CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f'table_{metric}.csv'
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False, float_format='%.4f')
    print(f"✓ Saved: {csv_filepath}")

    # Create formatted markdown table with bold for best values
    markdown_lines = []
    markdown_lines.append(f"# Comparison Table: {metric.replace('_', ' ').title()}\n")

    # Header
    headers = ['Data %'] + list(all_results.keys())
    markdown_lines.append('| ' + ' | '.join(headers) + ' |')
    markdown_lines.append('|' + '|'.join(['---' for _ in headers]) + '|')

    # Rows with bold for maximum value
    method_names = list(all_results.keys())
    for pct in all_percentages:
        values = []
        for method_name in method_names:
            val = all_results[method_name].get(pct, {}).get(metric, np.nan)
            values.append(val)

        # Find max value (ignoring NaN)
        valid_values = [v for v in values if not np.isnan(v)]
        max_val = max(valid_values) if valid_values else None

        # Build row
        row_parts = [f'{pct}%']
        for val in values:
            if np.isnan(val):
                row_parts.append('N/A')
            elif max_val is not None and abs(val - max_val) < 1e-6:
                row_parts.append(f'**{val:.4f}**')  # Bold for best
            else:
                row_parts.append(f'{val:.4f}')

        markdown_lines.append('| ' + ' | '.join(row_parts) + ' |')

    # Save markdown table
    md_filename = f'table_{metric}.md'
    md_filepath = os.path.join(output_dir, md_filename)
    with open(md_filepath, 'w') as f:
        f.write('\n'.join(markdown_lines))
    print(f"✓ Saved: {md_filepath}")

    # Print table to console
    print(f"\n{'='*80}")
    print(f"Table: {metric.replace('_', ' ').title()}")
    print('='*80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('='*80 + '\n')


def main():
    """Main analysis function."""
    print("="*80)
    print("SCALING EXPERIMENTS ANALYSIS")
    print("="*80)
    print()

    # Collect all metrics
    print("Step 1: Collecting metrics from tensorboard logs...")
    print("-"*80)
    all_results = collect_all_metrics('logs_scaling_experiments')
    print()

    if not all_results:
        print("ERROR: No results collected. Check tensorboard logs.")
        return

    # Create output directory
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots for each method
    print("\nStep 2: Generating individual method plots...")
    print("-"*80)
    for method_name, data in all_results.items():
        plot_individual_method(method_name, data, metric='test_f1_macro', output_dir=output_dir)
        plot_individual_method(method_name, data, metric='test_f1_weighted', output_dir=output_dir)
    print()

    # Generate combined comparison plots
    print("\nStep 3: Generating combined comparison plots...")
    print("-"*80)
    plot_combined_comparison(all_results, metric='test_f1_macro', output_dir=output_dir)
    plot_combined_comparison(all_results, metric='test_f1_weighted', output_dir=output_dir)
    print()

    # Create comparison tables
    print("\nStep 4: Creating comparison tables...")
    print("-"*80)
    create_comparison_table(all_results, metric='test_f1_macro', output_dir=output_dir)
    create_comparison_table(all_results, metric='test_f1_weighted', output_dir=output_dir)
    print()

    print("="*80)
    print(f"ANALYSIS COMPLETE! Results saved to: {output_dir}/")
    print("="*80)
    print("\nGenerated files:")
    print("  Individual plots:")
    print("    - Baseline_(Random_Init)_test_f1_macro.png")
    print("    - Baseline_(Random_Init)_test_f1_weighted.png")
    print("    - Hierarchical_(L1toL2)_test_f1_macro.png")
    print("    - Hierarchical_(L1toL2)_test_f1_weighted.png")
    print("    - Self-Supervised_(MoCotoL2)_test_f1_macro.png")
    print("    - Self-Supervised_(MoCotoL2)_test_f1_weighted.png")
    print("  Combined plots:")
    print("    - combined_comparison_test_f1_macro.png")
    print("    - combined_comparison_test_f1_weighted.png")
    print("  Tables:")
    print("    - table_test_f1_macro.csv / .md")
    print("    - table_test_f1_weighted.csv / .md")
    print()


if __name__ == '__main__':
    main()
