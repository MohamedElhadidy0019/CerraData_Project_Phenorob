#!/usr/bin/env python3
"""Check integrity of CerraData-4MM dataset splits."""
import os
import sys

def count_tifs(path):
    """Count .tif files in a directory. Returns 0 if directory doesn't exist."""
    if not os.path.isdir(path):
        return -1  # signals missing directory
    return len([f for f in os.listdir(path) if f.endswith('.tif')])

def main():
    base_dir = '/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted'
    subfolders = ["msi_images", "sar_images", "semantic_7c", "semantic_14c", "edge_7c", "edge_14c"]
    splits = ["train", "val", "test"]

    # Also check raw source if available
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle_temp", "cerradata_4mm")

    # --- Raw source counts ---
    if os.path.isdir(raw_dir):
        print("=" * 70)
        print("RAW SOURCE (kaggle_temp/cerradata_4mm)")
        print("=" * 70)
        for sf in subfolders:
            c = count_tifs(os.path.join(raw_dir, sf))
            status = "OK" if c > 0 else "MISSING"
            print(f"  {sf:20s}: {c:>6} files  [{status}]")
        print()

    # --- Per-split counts ---
    print("=" * 70)
    print(f"SPLIT DIRECTORY: {base_dir}")
    print("=" * 70)

    totals = {sf: 0 for sf in subfolders}
    split_counts = {}

    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.isdir(split_path):
            print(f"\n  [{split.upper()}] — DIRECTORY MISSING")
            split_counts[split] = {sf: -1 for sf in subfolders}
            continue

        print(f"\n  [{split.upper()}]")
        split_counts[split] = {}
        for sf in subfolders:
            c = count_tifs(os.path.join(split_path, sf))
            split_counts[split][sf] = c
            if c == -1:
                print(f"    {sf:20s}: {'MISSING':>10}")
            else:
                totals[sf] += c
                print(f"    {sf:20s}: {c:>6} files")

    # --- Totals ---
    print()
    print("=" * 70)
    print("TOTALS (sum across splits)")
    print("=" * 70)
    for sf in subfolders:
        print(f"  {sf:20s}: {totals[sf]:>6}")

    # --- Consistency checks ---
    print()
    print("=" * 70)
    print("CONSISTENCY CHECKS")
    print("=" * 70)
    errors = 0

    # Check 1: within each split, all present subfolders should have the same count
    for split in splits:
        if split not in split_counts:
            continue
        counts = split_counts[split]
        present = {sf: c for sf, c in counts.items() if c > 0}
        if not present:
            continue
        expected = counts.get("msi_images", -1)
        if expected <= 0:
            continue
        for sf, c in counts.items():
            if c == -1:
                print(f"  WARN : {split}/{sf} — directory missing entirely")
                errors += 1
            elif c != expected:
                print(f"  ERROR: {split}/{sf} has {c} files, expected {expected} (matching msi_images)")
                errors += 1

    # Check 2: totals should match raw source
    if os.path.isdir(raw_dir):
        raw_count = count_tifs(os.path.join(raw_dir, "msi_images"))
        for sf in subfolders:
            if totals[sf] > 0 and totals[sf] != raw_count:
                print(f"  ERROR: {sf} total={totals[sf]}, raw source has {raw_count}")
                errors += 1
            elif totals[sf] == raw_count:
                print(f"  OK   : {sf} total={totals[sf]} matches raw source ({raw_count})")

    # Check 3: no overlap between splits (spot check via filenames)
    print()
    for sf in subfolders:
        sets = {}
        for split in splits:
            path = os.path.join(base_dir, split, sf)
            if os.path.isdir(path):
                sets[split] = set(os.listdir(path))
        if len(sets) >= 2:
            split_names = list(sets.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    overlap = sets[split_names[i]] & sets[split_names[j]]
                    if overlap:
                        print(f"  ERROR: {sf} — {len(overlap)} files overlap between {split_names[i]} and {split_names[j]}")
                        errors += 1
                    else:
                        print(f"  OK   : {sf} — no overlap between {split_names[i]} and {split_names[j]}")

    print()
    if errors == 0:
        print("All checks passed.")
    else:
        print(f"{errors} issue(s) found.")

if __name__ == "__main__":
    main()
