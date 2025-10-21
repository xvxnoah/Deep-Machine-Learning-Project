#!/usr/bin/env python3
"""
Inference script for ViT ICH classification model.

Usage:
    python inference.py --model_path checkpoints/best_model.pt --config configs/base_config.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

# Add project root to path
sys.path.append('.')

from dataset import PreprocessedDataset
from models import ViTClassifier
from utils import MetricsCalculator
from utils.config_utils import load_config, get_device
from utils.metrics import log_classification_report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with trained ViT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for plots and results')
    return parser.parse_args()

def load_model_checkpoint(model_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("This might be due to:")
        print("1. Corrupted checkpoint file")
        print("2. Incompatible PyTorch version")
        print("3. File saved on different platform")
        raise

    # Extract config from checkpoint or load from file
    config = checkpoint.get('config', {})
    if not config:
        print("Warning: No config found in checkpoint, model may not match config")

    # Recreate model with same parameters
    model = ViTClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        input_channels=config['model']['input_channels'],
        slice_channels=config['model'].get('slice_channels', 3),
        dropout=config['model']['dropout'],
        backend=config['model']['backend'],
        unfreeze_layers=config['model'].get('unfreeze_layers', 0)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Best validation metric: {checkpoint.get('best_metric', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, config

def run_inference(model, test_loader, device):
    """Run inference on test set."""
    print("\n" + "="*60)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*60)

    model.eval()
    test_metrics = MetricsCalculator(num_classes=5)
    test_metrics.reset()

    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Get predictions (threshold 0.5)
            preds = (probs >= 0.5).cpu().numpy()
            targets = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_predictions.extend(preds)
            all_targets.extend(targets)
            all_probs.extend(probs_np)

            test_metrics.update(logits, labels)

    # Compute final metrics
    metrics = test_metrics.compute()

    return metrics, np.array(all_predictions), np.array(all_targets), np.array(all_probs)

def create_auc_roc_curves(metrics, targets, probabilities, output_dir):
    """Create AUC-ROC curves plot with 7 curves."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different classes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

    # Plot individual hemorrhage types (first 5 classes)
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    class_keys = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    for i, (name, key) in enumerate(zip(class_names, class_keys)):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=0.8,
                label=f'{name} (AUC = {roc_auc:.4f})')

    # Plot healthy cases (no hemorrhages)
    healthy_targets = (targets.sum(axis=1) == 0).astype(int)
    healthy_probs = 1 - probabilities.max(axis=1)  # Higher when all probs are low
    fpr, tpr, _ = roc_curve(healthy_targets, healthy_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[5], linewidth=2, alpha=0.8,
            label=f'Healthy (AUC = {roc_auc:.4f})')

    # Plot multiple hemorrhages
    multiple_targets = (targets.sum(axis=1) > 1).astype(int)
    multiple_probs = probabilities.max(axis=1)  # Higher when at least one prob is high
    fpr, tpr, _ = roc_curve(multiple_targets, multiple_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[6], linewidth=2, alpha=0.8,
            label=f'Multiple (AUC = {roc_auc:.4f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', alpha=0.7, label='Random')

    # Customize plot
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - All Categories', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'auc_roc_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ AUC-ROC curves plot saved to: {plot_path}")

def create_per_class_metrics_plot(metrics, output_dir):
    """Create bar chart for per-class precision/recall/f1."""
    class_names_plot = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    class_names_key = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    # Extract metrics
    precisions = [metrics[f'precision_{name}'] for name in class_names_key]
    recalls = [metrics[f'recall_{name}'] for name in class_names_key]
    f1_scores = [metrics[f'f1_{name}'] for name in class_names_key]

    # Plot
    x = np.arange(len(class_names_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Hemorrhage Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Per-class metrics plot saved to: {plot_path}")

def main():
    """Main inference function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Load config
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Get device
    device = get_device(config['hardware']['device'])
    print(f"Using device: {device}")

    # Load model
    model, checkpoint_config = load_model_checkpoint(args.model_path, device)

    # Create test dataset and loader
    data_root = config['data']['data_root']
    test_dataset = PreprocessedDataset(root_dir=os.path.join(data_root, 'test'))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    print(f"\nTest dataset: {len(test_dataset)} samples")
    print(f"Test batches: {len(test_loader)}")

    # Run inference
    metrics, predictions, targets, probabilities = run_inference(model, test_loader, device)

    # Print results
    print(f"\nInference completed on {len(test_dataset)} test samples")

    # Show classification report
    log_classification_report(metrics, prefix='TEST')

    # Create plots
    create_auc_roc_curves(metrics, targets, probabilities, str(output_dir))
    create_per_class_metrics_plot(metrics, str(output_dir))

    # Save results to JSON
    results = {
        'model_path': str(args.model_path),
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in metrics.items()},
        'config': config,
        'inference_time': str(output_dir)
    }

    results_path = output_dir / 'inference_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_path}")
    print(f"✓ All outputs saved to: {output_dir}")

    # Summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Samples: {len(test_dataset)}")
    print(f"AUC-ROC (Macro): {metrics['auc_roc_macro']:.4f}")
    print(f"AUC-ROC (Micro): {metrics['auc_roc_micro']:.4f}")
    print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"Exact Match Accuracy: {metrics['accuracy_exact']:.4f}")
    print(f"Hamming Accuracy: {metrics['accuracy_hamming']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
