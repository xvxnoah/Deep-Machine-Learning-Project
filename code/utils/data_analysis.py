import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path


def analyze_dataset(data_root='processed_data'):
    # Load labels
    labels_path = Path(data_root) / 'labels.json'
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Convert to array
    label_array = np.array([labels[tid] for tid in sorted(labels.keys())])
    
    # Class names
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 
                   'subarachnoid', 'subdural']
    
    # Count class combinations
    label_tuples = [tuple(row) for row in label_array]
    combination_counts = Counter(label_tuples)
    
    # Calculate statistics
    total_samples = len(label_array)
    pos_counts = label_array.sum(axis=0)
    neg_counts = total_samples - pos_counts
    
    # Print analysis
    print("\nDataset Analysis")
    print("-" * 16)
    
    print(f"Total Samples: {total_samples}")
    print(f"Unique Class Combinations: {len(combination_counts)}\n")
    
    # Per-class statistics
    print("Per-Class Distribution:")
    print(f"{'Class':<25s} {'Positive':>10s} {'Negative':>10s} {'Pos %':>10s} {'Imbalance':>12s}")
    print("-" * 80)
    
    for i, name in enumerate(class_names):
        pos = int(pos_counts[i])
        neg = int(neg_counts[i])
        pos_pct = (pos / total_samples) * 100
        imbalance_ratio = neg / (pos + 1e-5)
        print(f"{name:<25s} {pos:>10d} {neg:>10d} {pos_pct:>9.2f}% {imbalance_ratio:>11.2f}:1")
    
    print("-" * 80)
    
    # Multi-label statistics
    labels_per_sample = label_array.sum(axis=1)
    print(f"\nMulti-Label Statistics:")
    print(f"  Samples with 0 labels (healthy): {(labels_per_sample == 0).sum()} ({(labels_per_sample == 0).sum() / total_samples * 100:.2f}%)")
    print(f"  Samples with 1 label:            {(labels_per_sample == 1).sum()} ({(labels_per_sample == 1).sum() / total_samples * 100:.2f}%)")
    print(f"  Samples with 2+ labels:          {(labels_per_sample >= 2).sum()} ({(labels_per_sample >= 2).sum() / total_samples * 100:.2f}%)")
    print(f"  Average labels per sample:       {labels_per_sample.mean():.2f}")
    
    # Top combinations
    print(f"\nTop 10 Class Combinations:")
    print(f"{'Combination':<35s} {'Count':>10s} {'Percentage':>12s}")
    print("-" * 60)
    
    for combo, count in combination_counts.most_common(10):
        combo_str = str(combo)
        pct = (count / total_samples) * 100
        print(f"{combo_str:<35s} {count:>10d} {pct:>11.2f}%")
    
    # Analyze splits
    print("\nDataset Splits:")
    splits_info = {}
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / split
        if split_path.exists():
            # Count files
            file_count = sum([len(list(d.glob('*.pt'))) 
                            for d in split_path.iterdir() 
                            if d.is_dir()])
            splits_info[split] = file_count
            print(f"  {split:5s}: {file_count:5d} samples ({file_count/total_samples*100:5.2f}%)")
    
    
    return {
        'total_samples': total_samples,
        'pos_counts': pos_counts,
        'neg_counts': neg_counts,
        'combination_counts': combination_counts,
        'labels_per_sample': labels_per_sample,
        'splits_info': splits_info,
        'label_array': label_array
    }

def compute_dataset_statistics(dataset):
    print("Computing dataset statistics")
    
    all_pixels = []
    
    for i in range(min(1000, len(dataset))):  # Sample first 1000 for speed
        image, _ = dataset[i]
        all_pixels.append(image.numpy().flatten())
    
    all_pixels = np.concatenate(all_pixels)
    
    stats = {
        'mean': float(all_pixels.mean()),
        'std': float(all_pixels.std()),
        'min': float(all_pixels.min()),
        'max': float(all_pixels.max()),
        'median': float(np.median(all_pixels))
    }
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key:10s}: {value:.6f}")
    
    return stats