#!/usr/bin/env python3
"""
Script to check training progress by analyzing checkpoint files.
Shows last epoch reached and all saved checkpoints for each experiment.
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def parse_checkpoint_filename(filename):
    """
    Extract epoch number from checkpoint filename.

    Examples:
        - 'epoch=59-val_loss=0.5295-val_f1_macro=0.4479.ckpt' -> 59
        - 'last.ckpt' -> None

    Returns:
        int or None: Epoch number if found, None otherwise
    """
    if filename == 'last.ckpt':
        return None

    match = re.search(r'epoch=(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def parse_percentage_from_dirname(dirname):
    """
    Extract percentage from experiment directory name.

    Examples:
        - 'l2_baseline_14classes_0_5percent_20260102_161127' -> 0.5
        - 'l2_baseline_14classes_25percent_20251229_222207' -> 25.0
    """
    # Pattern with lr specification
    match = re.search(r'classes_(?:lr\d+e-?\d+_)?(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)

    # Pattern for self-supervised
    match = re.search(r'(?:simclr|moco)_(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)

    # Pattern for finetuning
    match = re.search(r'from_l\d+_(\d+(?:_\d+)?)percent', dirname)
    if match:
        pct_str = match.group(1).replace('_', '.')
        return float(pct_str)

    return None


def analyze_experiment_directory(exp_path):
    """
    Analyze a single experiment directory for checkpoints.

    Returns:
        dict: {
            'num_ckpts': int,
            'epochs': list of int,
            'last_epoch': int or None,
            'has_last_ckpt': bool
        }
    """
    ckpt_files = [f for f in os.listdir(exp_path) if f.endswith('.ckpt')]

    epochs = []
    has_last_ckpt = False

    for ckpt_file in ckpt_files:
        if ckpt_file == 'last.ckpt':
            has_last_ckpt = True
        else:
            epoch = parse_checkpoint_filename(ckpt_file)
            if epoch is not None:
                epochs.append(epoch)

    epochs.sort()
    last_epoch = max(epochs) if epochs else None

    return {
        'num_ckpts': len(ckpt_files),
        'epochs': epochs,
        'last_epoch': last_epoch,
        'has_last_ckpt': has_last_ckpt
    }


def collect_all_experiments(base_dir='checkpoints_scaling_experiments'):
    """
    Collect checkpoint info for all experiments.

    Returns:
        dict: {method_name: {percentage: experiment_info}}
    """
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return {}

    method_dirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d))
                   and not d.startswith('.')]

    results = defaultdict(dict)

    for method_dir in sorted(method_dirs):
        method_path = os.path.join(base_dir, method_dir)

        # Get all experiment directories
        exp_dirs = [d for d in os.listdir(method_path)
                   if os.path.isdir(os.path.join(method_path, d))
                   and not d.startswith('.')
                   and d != 'README.md']

        for exp_dir in sorted(exp_dirs):
            exp_path = os.path.join(method_path, exp_dir)
            percentage = parse_percentage_from_dirname(exp_dir)

            if percentage is not None:
                info = analyze_experiment_directory(exp_path)
                info['exp_name'] = exp_dir
                results[method_dir][percentage] = info

    return results


def format_epochs_list(epochs, max_display=5):
    """Format list of epochs for display, truncating if too long."""
    if not epochs:
        return "None"

    if len(epochs) <= max_display:
        return ', '.join(map(str, epochs))
    else:
        first_few = ', '.join(map(str, epochs[:max_display-1]))
        return f"{first_few}, ... ({len(epochs)} total)"


def print_results(results):
    """Print results in a nicely formatted table."""
    if not results:
        print("No experiments found.")
        return

    print("=" * 100)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 100)
    print()

    for method_name in sorted(results.keys()):
        experiments = results[method_name]

        if not experiments:
            continue

        # Print method header
        print(f"{'─' * 100}")
        print(f"Method: {method_name}")
        print(f"{'─' * 100}")

        # Print table header
        print(f"{'Data %':<10} │ {'Ckpts':<7} │ {'Last Epoch':<12} │ {'Status':<15} │ {'Epochs Saved':<40}")
        print(f"{'─' * 10}─┼─{'─' * 7}─┼─{'─' * 12}─┼─{'─' * 15}─┼─{'─' * 40}")

        # Print each experiment
        for pct in sorted(experiments.keys()):
            info = experiments[pct]

            # Format data percentage
            pct_str = f"{pct}%"

            # Format number of checkpoints
            num_ckpts = info['num_ckpts']

            # Format last epoch
            last_epoch = info['last_epoch']
            last_epoch_str = str(last_epoch) if last_epoch is not None else "N/A"

            # Determine status
            if num_ckpts == 0:
                status = "⚠️ NO CKPTS"
            elif info['has_last_ckpt']:
                status = "✓ Complete"
            elif last_epoch and last_epoch < 30:
                status = "⚠️ Early stop?"
            else:
                status = "✓ Has ckpts"

            # Format epochs list
            epochs_str = format_epochs_list(info['epochs'])
            if info['has_last_ckpt']:
                epochs_str += " + last"

            # Print row
            print(f"{pct_str:<10} │ {num_ckpts:<7} │ {last_epoch_str:<12} │ {status:<15} │ {epochs_str:<40}")

        print()

    print("=" * 100)
    print()

    # Print summary statistics
    total_experiments = sum(len(exps) for exps in results.values())
    complete_experiments = sum(
        1 for exps in results.values()
        for info in exps.values()
        if info['has_last_ckpt']
    )
    no_ckpts = sum(
        1 for exps in results.values()
        for info in exps.values()
        if info['num_ckpts'] == 0
    )

    print("SUMMARY:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Complete (has last.ckpt): {complete_experiments}")
    print(f"  No checkpoints: {no_ckpts}")
    print(f"  In progress: {total_experiments - complete_experiments - no_ckpts}")
    print()


def main():
    """Main function."""
    print()
    print("Analyzing checkpoint directories...")
    print()

    results = collect_all_experiments('checkpoints_scaling_experiments')
    print_results(results)


if __name__ == '__main__':
    main()
