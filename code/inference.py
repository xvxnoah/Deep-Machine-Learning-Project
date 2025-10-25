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
from tqdm import tqdm

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

# Add project root to path
sys.path.append('.')

from dataset import PreprocessedDataset
from models import ViTClassifier, EnsembleViTClassifier
from utils import MetricsCalculator, Trainer
from utils.config_utils import load_config, get_device
from utils.metrics import log_classification_report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with trained ViT model or ensemble')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model checkpoint (for single model evaluation)')
    parser.add_argument('--full_vit_path', type=str, default='best_model_full.pt',
                       help='Path to full ViT model checkpoint (for ensemble)')
    parser.add_argument('--rnn_vit_path', type=str, default='best_model_rnn.pt',
                       help='Path to ViT-RNN model checkpoint (for ensemble)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble evaluation combining full ViT and ViT-RNN models')
    parser.add_argument('--ensemble_method', type=str, default='average',
                       choices=['average', 'weighted', 'max_confidence', 'voting'],
                       help='Ensemble combination method')
    parser.add_argument('--ensemble_weights', type=float, nargs=2, default=[0.5, 0.5],
                       help='Weights for weighted ensemble [full_vit_weight, rnn_vit_weight]')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for plots and results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use for inference (default: cpu for compatibility)')
    return parser.parse_args()

def load_model_checkpoint(model_path, device):
    """Load model from checkpoint with robust device handling."""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint with proper device mapping
    # CUDA-trained models work best when loaded back to CUDA
    print("Attempting to load CUDA-trained checkpoint...")

    # Try to load to the target device first
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✓ Checkpoint loaded successfully to {device}")
    except Exception as e1:
        print(f"Failed to load directly to {device}: {e1}")

        # If that fails, try loading to CPU first (more compatible)
        try:
            print("Attempting to load via CPU first...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print("✓ Checkpoint loaded successfully via CPU")
        except Exception as e2:
            print(f"❌ Failed to load checkpoint: {e2}")
            print("This CUDA-trained checkpoint appears to be corrupted or incompatible.")
            print("Possible solutions:")
            print("1. Try --device cuda (if CUDA GPU available)")
            print("2. Use a different checkpoint file")
            print("3. Re-run training to generate new checkpoints")
            print("4. Check if CUDA was available during training")
            raise e2

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    if not config:
        print("⚠️  Warning: No config found in checkpoint, using default config")
        config = {'model': {'name': 'google/vit-base-patch16-224', 'num_classes': 5,
                           'pretrained': True, 'input_channels': 9, 'slice_channels': 3,
                           'dropout': 0.1, 'backend': 'torchvision', 'unfreeze_layers': 0}}

    # Recreate model with same parameters
    print("Recreating model architecture...")
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
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        print("The model architecture might not match the checkpoint.")
        raise

    # Move model to target device
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Best validation metric: {checkpoint.get('best_metric', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Model moved to device: {device}")

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
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Inference]"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Get predictions (threshold 0.5)
            preds = (probs >= 0.5).cpu().numpy()
            targets = labels.cpu().numpy()
            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_predictions.extend(preds)
            all_targets.extend(targets)
            all_logits.extend(logits_np)
            all_probs.extend(probs_np)

            test_metrics.update(logits, labels)

    # Compute final metrics
    metrics = test_metrics.compute()

    return metrics, np.array(all_predictions), np.array(all_targets), np.array(all_logits), np.array(all_probs)

def find_best_threshold(logits, labels, device):
    """Find the best threshold by optimizing Youden's J statistic (Sensitivity + Specificity - 1)."""
    print("\n" + "="*60)
    print("FINDING BEST THRESHOLD (Youden's J Optimization)")
    print("="*60)

    # Convert logits to probabilities
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()

    # Initialize best metrics
    best_threshold = 0.5
    best_youden = -1.0  # Youden's J ranges from -1 to 1
    best_acc, best_sens, best_spec, best_prec, best_npv, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Test all thresholds
    for threshold in tqdm(np.arange(0.0, 1.01, 0.01), desc="[Threshold Optimization]"):
        binary_outputs = (probabilities > threshold).astype(int)
        acc, sens, spec, prec, npv, f1, youden_scores = [], [], [], [], [], [], []

        for i in range(binary_outputs.shape[1]):
            outputs = binary_outputs[:, i]
            targets = labels[:, i]
            cm = confusion_matrix(targets, outputs)
            tn, fp, fn, tp = cm.ravel()
            acc.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0)
            sens.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
            spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
            prec.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
            npv.append(tn / (tn + fn) if (tn + fn) != 0 else 0)
            f1.append(2 * (prec[-1] * sens[-1]) / (prec[-1] + sens[-1]) if (prec[-1] + sens[-1]) != 0 else 0)

            # Calculate Youden's J for this class: sensitivity + specificity - 1
            youden = sens[-1] + spec[-1] - 1
            youden_scores.append(youden)

        mean_acc = np.mean(acc)
        mean_sens = np.mean(sens)
        mean_spec = np.mean(spec)
        mean_prec = np.mean(prec)
        mean_npv = np.mean(npv)
        mean_f1 = np.mean(f1)
        mean_youden = np.mean(youden_scores)

        # Use Youden's J statistic as optimization criterion
        if mean_youden > best_youden:
            best_youden = mean_youden
            best_acc = mean_acc
            best_sens = mean_sens
            best_spec = mean_spec
            best_prec = mean_prec
            best_npv = mean_npv
            best_f1 = mean_f1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.4f} | Youden's J: {best_youden:.4f}")
    print(f"Best Accuracy: {best_acc:.4f} | Best Sensitivity: {best_sens:.4f} | Best Specificity: {best_spec:.4f}")
    print(f"Best Precision: {best_prec:.4f} | Best F1 Score: {best_f1:.4f}")

    # Generate confusion matrices for each subtype using the best threshold
    binary_outputs = (probabilities > best_threshold).astype(int)
    print("\nConfusion Matrices for each subtype (Best Threshold = {:.2f}):".format(best_threshold))
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    for i in range(binary_outputs.shape[1]):
        outputs = binary_outputs[:, i]
        targets = labels[:, i]
        cm = confusion_matrix(targets, outputs)
        print(f"\nConfusion Matrix for {class_names[i]}:")
        print("Predicted |  0   |  1   ")
        print("Actual   |------|------")
        print(f"   0     | {cm[0,0]:4d} | {cm[0,1]:4d} ")
        print(f"   1     | {cm[1,0]:4d} | {cm[1,1]:4d} ")

    return best_threshold, binary_outputs

def calculate_threshold_metrics(predictions, targets, threshold_name=""):
    """Calculate metrics for given predictions and targets."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )

    # Macro averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    # Micro averages
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        targets, predictions, average='micro', zero_division=0
    )

    # Weighted averages
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )

    # Accuracy metrics
    exact_match_acc = accuracy_score(targets, predictions)
    hamming_acc = 1 - hamming_loss(targets, predictions)

    # Per-class metrics
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    metrics = {
        f'precision_macro{threshold_name}': precision_macro,
        f'recall_macro{threshold_name}': recall_macro,
        f'f1_macro{threshold_name}': f1_macro,
        f'precision_micro{threshold_name}': p_micro,
        f'recall_micro{threshold_name}': r_micro,
        f'f1_micro{threshold_name}': f1_micro,
        f'precision_weighted{threshold_name}': p_weighted,
        f'recall_weighted{threshold_name}': r_weighted,
        f'f1_weighted{threshold_name}': f1_weighted,
        f'accuracy_exact{threshold_name}': exact_match_acc,
        f'accuracy_hamming{threshold_name}': hamming_acc,
    }

    for i, name in enumerate(class_names):
        metrics[f'precision_{name}{threshold_name}'] = precision[i]
        metrics[f'recall_{name}{threshold_name}'] = recall[i]
        metrics[f'f1_{name}{threshold_name}'] = f1[i]

    return metrics

def create_simple_auc_roc_curves(metrics, targets, probabilities, output_dir):
    """Create simple AUC-ROC curves plot with 7 curves (no operating points)."""
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
    plot_path = os.path.join(output_dir, 'simple_auc_roc_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Simple AUC-ROC curves plot saved to: {plot_path}")

def create_auc_roc_curves_with_thresholds(metrics, targets, probabilities, output_dir, default_threshold=0.5, best_threshold=None):
    """Create AUC-ROC curves plot with operating points marked."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different classes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

    # Plot individual hemorrhage types (first 5 classes)
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    class_keys = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    for i, (name, key) in enumerate(zip(class_names, class_keys)):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(targets[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=0.8,
                label=f'{name} (AUC = {roc_auc:.4f})')

        # Mark default threshold operating point (red squares)
        threshold_idx = np.argmin(np.abs(thresholds - default_threshold))
        ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red',
                  marker='s', s=100, edgecolor='black', linewidth=2,
                  label=f'Default (0.5)' if i == 0 else "")

        # Mark best threshold operating point (blue circles)
        if best_threshold is not None:
            threshold_idx = np.argmin(np.abs(thresholds - best_threshold))
            ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='blue',
                      marker='o', s=100, edgecolor='black', linewidth=2,
                      label=f'Optimal ({best_threshold:.2f})' if i == 0 else "")

    # Plot healthy cases (no hemorrhages)
    healthy_targets = (targets.sum(axis=1) == 0).astype(int)
    healthy_probs = 1 - probabilities.max(axis=1)  # Higher when all probs are low
    fpr, tpr, thresholds = roc_curve(healthy_targets, healthy_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[5], linewidth=2, alpha=0.8,
            label=f'Healthy (AUC = {roc_auc:.4f})')

    # Plot multiple hemorrhages
    multiple_targets = (targets.sum(axis=1) > 1).astype(int)
    multiple_probs = probabilities.max(axis=1)  # Higher when at least one prob is high
    fpr, tpr, thresholds = roc_curve(multiple_targets, multiple_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[6], linewidth=2, alpha=0.8,
            label=f'Multiple (AUC = {roc_auc:.4f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', alpha=0.7, label='Random')

    # Customize plot
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    title = 'ROC Curves with Operating Points'
    if best_threshold is not None:
        title += f' (Default: {default_threshold}, Optimal: {best_threshold:.2f})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'auc_roc_curves_with_thresholds.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ AUC-ROC curves with thresholds plot saved to: {plot_path}")

def create_threshold_comparison_roc_curves(targets, probabilities, default_threshold, best_threshold, output_dir):
    """Create ROC curves showing operating points for both thresholds."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Colors for different classes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    # Plot individual hemorrhage types (first 5 classes) + healthy
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural', 'Healthy']
    plot_data = [
        (targets[:, 0], probabilities[:, 0]),  # Epidural
        (targets[:, 1], probabilities[:, 1]),  # Intraparenchymal
        (targets[:, 2], probabilities[:, 2]),  # Intraventricular
        (targets[:, 3], probabilities[:, 3]),  # Subarachnoid
        (targets[:, 4], probabilities[:, 4]),  # Subdural
        ((targets.sum(axis=1) == 0).astype(int), 1 - probabilities.max(axis=1))  # Healthy
    ]

    for i, (name, (y_true, y_scores)) in enumerate(zip(class_names, plot_data)):
        ax = axes[i]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=0.8,
                label=f'{name} (AUC = {roc_auc:.4f})')

        # Mark default threshold (0.5) operating point
        if i < 5:  # For hemorrhage classes
            threshold_idx = np.argmin(np.abs(thresholds - default_threshold))
            ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red',
                      marker='s', s=100, edgecolor='black', linewidth=2,
                      label=f'Default (0.5) TPR={tpr[threshold_idx]:.3f}, FPR={fpr[threshold_idx]:.3f}')
        else:  # For healthy class
            threshold_idx = np.argmin(np.abs(thresholds - (1-default_threshold)))
            ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red',
                      marker='s', s=100, edgecolor='black', linewidth=2,
                      label=f'Default (0.5) TPR={tpr[threshold_idx]:.3f}, FPR={fpr[threshold_idx]:.3f}')

        # Mark best threshold operating point
        if i < 5:  # For hemorrhage classes
            threshold_idx = np.argmin(np.abs(thresholds - best_threshold))
            ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='blue',
                      marker='o', s=100, edgecolor='black', linewidth=2,
                      label=f'Optimal ({best_threshold:.2f}) TPR={tpr[threshold_idx]:.3f}, FPR={fpr[threshold_idx]:.3f}')
        else:  # For healthy class
            threshold_idx = np.argmin(np.abs(thresholds - (1-best_threshold)))
            ax.scatter(fpr[threshold_idx], tpr[threshold_idx], color='blue',
                      marker='o', s=100, edgecolor='black', linewidth=2,
                      label=f'Optimal ({best_threshold:.2f}) TPR={tpr[threshold_idx]:.3f}, FPR={fpr[threshold_idx]:.3f}')

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', alpha=0.7)

        # Customize subplot
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{name} ROC Curve', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Hide the last subplot if we have 6 plots
    if len(class_names) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'threshold_comparison_roc.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Threshold comparison ROC curves saved to: {plot_path}")

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


def load_ensemble_model(full_vit_path, rnn_vit_path, ensemble_method, weights, device):
    """Load ensemble model combining full ViT and ViT-RNN models."""
    print("Loading ensemble model...")
    print(f"  Full ViT path: {full_vit_path}")
    print(f"  RNN ViT path: {rnn_vit_path}")
    print(f"  Ensemble method: {ensemble_method}")
    print(f"  Weights: {weights}")

    ensemble = EnsembleViTClassifier(
        full_vit_path=full_vit_path,
        rnn_vit_path=rnn_vit_path,
        ensemble_method=ensemble_method,
        weights=weights,
        device=device
    )

    return ensemble


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
    if args.device == 'auto':
        # For CUDA-trained models, prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Auto-selected CUDA device (recommended for CUDA-trained models)")
        else:
            device = get_device(config['hardware']['device'])
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Special handling for CUDA checkpoints
    if device.type == 'cpu':
        print("Note: Loading CUDA-trained checkpoint to CPU")
        print("If this fails, try --device cuda (if CUDA is available)")
    elif device.type == 'cuda':
        print("Loading CUDA checkpoint to CUDA device (recommended)")
    elif device.type == 'mps':
        print("Warning: Loading CUDA checkpoint to MPS may cause issues")
        print("Consider using --device cuda or --device cpu instead")

    # Load model or ensemble
    if args.ensemble:
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION MODE")
        print("="*60)

        # Validate required arguments for ensemble
        if not os.path.exists(args.full_vit_path):
            raise FileNotFoundError(f"Full ViT model not found: {args.full_vit_path}")
        if not os.path.exists(args.rnn_vit_path):
            raise FileNotFoundError(f"RNN ViT model not found: {args.rnn_vit_path}")

        model = load_ensemble_model(
            full_vit_path=args.full_vit_path,
            rnn_vit_path=args.rnn_vit_path,
            ensemble_method=args.ensemble_method,
            weights=args.ensemble_weights,
            device=device
        )
        checkpoint_config = None  # Ensemble doesn't have a single config
        model_name = f"Ensemble_{args.ensemble_method}"

    else:
        print("\n" + "="*60)
        print("SINGLE MODEL EVALUATION MODE")
        print("="*60)

        # Validate required arguments for single model
        if not args.model_path:
            raise ValueError("Must specify --model_path for single model evaluation")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        # Load single model
        model, checkpoint_config = load_model_checkpoint(args.model_path, device)
        model_name = checkpoint_config['model']['name'] if checkpoint_config else "Unknown"

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
    metrics, predictions, targets, logits, probabilities = run_inference(model, test_loader, device)

    # Calculate metrics for default threshold (0.5)
    default_predictions = (probabilities >= 0.5).astype(int)
    default_threshold_metrics = calculate_threshold_metrics(default_predictions, targets, threshold_name="_default")

    # Find best threshold
    best_threshold, best_predictions = find_best_threshold(logits, targets, device)

    # Calculate metrics for best threshold
    best_threshold_metrics = calculate_threshold_metrics(best_predictions, targets, threshold_name="_best")

    # Print results
    print(f"\nInference completed on {len(test_dataset)} test samples")
    print(f"Best threshold found: {best_threshold:.4f}")

    # Show classification report with default threshold
    print("\n" + "="*60)
    print("METRICS WITH DEFAULT THRESHOLD (0.5)")
    print("="*60)
    log_classification_report(metrics, prefix='DEFAULT THRESHOLD')

    # Show metrics comparison
    print("\n" + "="*80)
    print("METRICS COMPARISON: Default Threshold (0.5) vs Best Threshold")
    print("="*80)

    print(f"\n{'Metric':<25} {'Default (0.5)':>15} {f'Best ({best_threshold:.2f})':>15} {'Difference':>12}")
    print("-" * 80)

    # Calculate Youden's J for default threshold
    default_predictions = (probabilities >= 0.5).astype(int)
    default_youden_scores = []
    for i in range(default_predictions.shape[1]):
        outputs = default_predictions[:, i]
        targets_i = targets[:, i]
        cm = confusion_matrix(targets_i, outputs)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            default_youden_scores.append(sens + spec - 1)
    default_youden = np.mean(default_youden_scores) if default_youden_scores else 0

    # Calculate Youden's J for best threshold
    best_youden_scores = []
    for i in range(best_predictions.shape[1]):
        outputs = best_predictions[:, i]
        targets_i = targets[:, i]
        cm = confusion_matrix(targets_i, outputs)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) != 0 else 0
            spec = tn / (tn + fp) if (tn + fp) != 0 else 0
            best_youden_scores.append(sens + spec - 1)
    best_youden = np.mean(best_youden_scores) if best_youden_scores else 0

    key_metrics = [
        'f1_macro', 'precision_macro', 'recall_macro',
        'accuracy_exact', 'accuracy_hamming'
    ]

    for metric in key_metrics:
        default_val = metrics.get(metric, 0)
        best_val = best_threshold_metrics.get(f'{metric}_best', 0)
        diff = best_val - default_val
        print(f"{metric:<25} {default_val:>15.4f} {best_val:>15.4f} {diff:>+12.4f}")

    # Add Youden's J to the comparison
    print(f"{'youden_j':<25} {default_youden:>15.4f} {best_youden:>15.4f} {best_youden - default_youden:>+12.4f}")

    # Create plots (using original probabilities for AUC-ROC, which is threshold-independent)
    # Plot 1: Simple ROC curves without operating points
    create_simple_auc_roc_curves(metrics, targets, probabilities, str(output_dir))

    # Plot 2: ROC curves with operating points for both thresholds
    create_auc_roc_curves_with_thresholds(metrics, targets, probabilities, str(output_dir), 0.5, best_threshold)

    # Additional detailed threshold comparison plot
    create_threshold_comparison_roc_curves(targets, probabilities, 0.5, best_threshold, str(output_dir))
    create_per_class_metrics_plot(metrics, str(output_dir))

    # Save results to JSON
    if args.ensemble:
        results = {
            'model_type': 'ensemble',
            'ensemble_method': args.ensemble_method,
            'ensemble_weights': args.ensemble_weights,
            'full_vit_path': str(args.full_vit_path),
            'rnn_vit_path': str(args.rnn_vit_path),
            'best_threshold': float(best_threshold),
            'test_metrics_default_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                             for k, v in metrics.items()},
            'test_metrics_best_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                           for k, v in best_threshold_metrics.items()},
            'config': config,
            'inference_time': str(output_dir)
        }
    else:
        results = {
            'model_type': 'single',
            'model_path': str(args.model_path),
            'best_threshold': float(best_threshold),
            'test_metrics_default_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                             for k, v in metrics.items()},
            'test_metrics_best_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                           for k, v in best_threshold_metrics.items()},
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
    if args.ensemble:
        print(f"Model Type: Ensemble ({args.ensemble_method})")
        print(f"Full ViT: {args.full_vit_path}")
        print(f"RNN ViT: {args.rnn_vit_path}")
        print(f"Weights: {args.ensemble_weights}")
    else:
        print(f"Model: {args.model_path}")
    print(f"Samples: {len(test_dataset)}")
    print(f"Best Threshold: {best_threshold:.4f}")

    print(f"\nAUC-ROC (threshold-independent):")
    print(f"  Macro: {metrics['auc_roc_macro']:.4f}")
    print(f"  Micro: {metrics['auc_roc_micro']:.4f}")

    print(f"\nDefault Threshold (0.5):")
    print(f"  F1 (Macro):           {metrics['f1_macro']:.4f}")
    print(f"  Exact Match Accuracy: {metrics['accuracy_exact']:.4f}")
    print(f"  Hamming Accuracy:     {metrics['accuracy_hamming']:.4f}")
    print(f"  Youden's J:           {default_youden:.4f}")

    print(f"\nBest Threshold ({best_threshold:.2f}):")
    print(f"  F1 (Macro):           {best_threshold_metrics['f1_macro_best']:.4f}")
    print(f"  Exact Match Accuracy: {best_threshold_metrics['accuracy_exact_best']:.4f}")
    print(f"  Hamming Accuracy:     {best_threshold_metrics['accuracy_hamming_best']:.4f}")
    print(f"  Youden's J:           {best_youden:.4f}")

    print("="*60)

if __name__ == '__main__':
    main()
