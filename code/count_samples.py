#!/usr/bin/env python3
"""
Script to count the number of samples in train, validation, and test splits.
"""

import os
from collections import defaultdict
from pathlib import Path


def count_samples_in_split(split_path: str) -> dict:
    """
    Count samples in each class directory within a split.

    Args:
        split_path: Path to the split directory (train/val/test)

    Returns:
        Dictionary with class names as keys and sample counts as values
    """
    class_counts = defaultdict(int)
    total_samples = 0

    # Get all subdirectories (classes)
    for class_dir in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_dir)

        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue

        # Count files in this class directory
        file_count = len([f for f in os.listdir(class_path)
                         if os.path.isfile(os.path.join(class_path, f))])

        class_counts[class_dir] = file_count
        total_samples += file_count

    class_counts['TOTAL'] = total_samples
    return class_counts


def main():
    """Main function to count samples across all splits."""
    base_path = "/Users/noahmv/Desktop/noah/studies/Master's/Chalmers/2nd year/sp_1/DML/Project/DL_Project_Processed_Data"

    splits = ['train', 'val', 'test']
    results = {}

    print("Dataset Sample Counts")
    print("=" * 50)

    for split in splits:
        split_path = os.path.join(base_path, split)

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            continue

        print(f"\n{split.upper()} Split:")
        print("-" * 20)

        class_counts = count_samples_in_split(split_path)
        results[split] = class_counts

        # Print class breakdown
        for class_name, count in class_counts.items():
            if class_name != 'TOTAL':
                print(f"  {class_name}: {count} samples")
            else:
                print(f"  {class_name}: {count} samples")

    # Print summary table
    print("\n\nSUMMARY TABLE")
    print("=" * 50)
    print(f"{'Split':<10} {'Samples':<10}")
    print("-" * 50)

    for split in splits:
        if split in results:
            print(f"{split.upper():<10} {results[split]['TOTAL']:<10}")

    print("-" * 50)
    total_all = sum(results[split]['TOTAL'] for split in splits if split in results)
    print(f"{'TOTAL':<10} {total_all:<10}")


if __name__ == "__main__":
    main()
