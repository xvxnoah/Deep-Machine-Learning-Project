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
from models import ViTClassifier, ViTTripletBiRNNClassifier
from utils import MetricsCalculator, Trainer
from utils.config_utils import load_config, get_device
from utils.metrics import log_classification_report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with trained ViT model')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for plots and results (default: inference_<checkpoint_name>)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device to use for inference (default: cpu for compatibility)')
    return parser.parse_args()

def load_model_checkpoint(model_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try to load to the target device first
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Checkpoint loaded successfully to {device}")
    except Exception as e1:
        print(f"Failed to load directly to {device}: {e1}")

        # If that fails, try loading to CPU first
        try:
            print("Attempting to load via CPU...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print("Checkpoint loaded successfully via CPU")
        except Exception as e2:
            raise e2

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    if not config:
        print("Warning: No config found in checkpoint, using default config")
        config = {'model': {'name': 'google/vit-base-patch16-224', 'num_classes': 5,
                           'pretrained': True, 'input_channels': 9, 'slice_channels': 3,
                           'dropout': 0.1, 'backend': 'torchvision', 'unfreeze_layers': 0}}

    # Recreate model with same parameters
    print("Recreating model architecture...")

    # Check if this is a ViT-RNN model by checking the model path
    is_rnn_model = 'rnn' in model_path.lower()

    if is_rnn_model:
        # Override model_name to avoid relative path issues in stored config
        model_name_override = 'pre-trained-model'
        print(f"  Detected ViT-RNN model, overriding model_name to: {model_name_override}")

        model = ViTTripletBiRNNClassifier(
            model_name=model_name_override,  # Use override instead of config['model']['name']
            num_classes=config['model']['num_classes'],
            pretrained=False,  # Don't load pretrained weights when loading from checkpoint
            input_channels=config['model']['input_channels'],
            slice_channels=config['model'].get('slice_channels', 3),
            dropout=config['model']['dropout'],
            rnn_hidden_size=config['model'].get('rnn_hidden_size', 512),
            rnn_num_layers=config['model'].get('rnn_num_layers', 1),
            rnn_dropout=config['model'].get('rnn_dropout', 0.0),
            sequence_pooling=config['model'].get('sequence_pooling', 'last'),
            backend=config['model']['backend']
        )
    else:
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise

    # Move model to target device
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
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

    print(f"Simple AUC-ROC curves plot saved to: {plot_path}")

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

    print(f"Per-class metrics plot saved to: {plot_path}")


def main():
    args = parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        # Extract checkpoint name without .pt extension
        checkpoint_name = Path(args.model_path).stem  # Removes .pt extension
        args.output_dir = f"inference_{checkpoint_name}"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load config
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Get device
    if args.device == 'auto':
        # For CUDA-trained models, prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = get_device(config['hardware']['device'])
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print("\n" + "="*60)
    print("MODEL EVALUATION MODE")
    print("="*60)

    # Validate required arguments for single model
    if not args.model_path:
        raise ValueError("Must specify --model_path for model evaluation")
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

    # Print results
    print(f"\nInference completed on {len(test_dataset)} test samples")

    # Show classification report with default threshold
    print("\n" + "="*60)
    print("METRICS WITH DEFAULT THRESHOLD (0.5)")
    print("="*60)
    log_classification_report(metrics, prefix='DEFAULT THRESHOLD')


    # Create plots
    create_simple_auc_roc_curves(metrics, targets, probabilities, str(output_dir))
    create_per_class_metrics_plot(metrics, str(output_dir))

    # Save results to JSON
    results = {
        'model_type': 'single',
        'model_path': str(args.model_path),
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()},
        'config': config,
        'inference_time': str(output_dir)
    }

    results_path = output_dir / 'inference_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"All outputs saved to: {output_dir}")

    # Summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Samples: {len(test_dataset)}")

    print(f"\nAUC-ROC (threshold-independent):")
    print(f"  Macro: {metrics['auc_roc_macro']:.4f}")
    print(f"  Micro: {metrics['auc_roc_micro']:.4f}")

    print(f"\nDefault Threshold (0.5):")
    print(f"  F1 (Macro):           {metrics['f1_macro']:.4f}")
    print(f"  Exact Match Accuracy: {metrics['accuracy_exact']:.4f}")
    print(f"  Hamming Accuracy:     {metrics['accuracy_hamming']:.4f}")

    print("="*60)

if __name__ == '__main__':
    main()
