import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append('.')

from dataset import PreprocessedDataset
from models import ViTClassifier, EnsembleViTClassifier
from utils import MetricsCalculator
from utils.config_utils import load_config, get_device
from utils.metrics import log_classification_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ensemble ViT models for ICH classification')
    parser.add_argument('--full_vit_path', type=str, required=True,
                       help='Path to full ViT model checkpoint')
    parser.add_argument('--rnn_vit_path', type=str, required=True,
                       help='Path to ViT-RNN model checkpoint')
    parser.add_argument('--method', type=str, default='average',
                       choices=['average', 'weighted', 'max_confidence', 'voting'],
                       help='Ensemble combination method')
    parser.add_argument('--weights', type=float, nargs=2, default=[0.5, 0.5],
                       help='Weights for weighted ensemble [full_vit_weight, rnn_vit_weight]')
    parser.add_argument('--config', type=str, default='configs/vit_base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='ensemble_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device to use for evaluation')
    return parser.parse_args()


def evaluate_individual_models(full_vit_path, rnn_vit_path, test_loader, device):
    """Evaluate individual models for comparison."""
    print("\n" + "="*60)
    print("EVALUATING INDIVIDUAL MODELS")
    print("="*60)

    results = {}

    # Evaluate Full ViT
    print("\nEvaluating Full ViT model...")
    try:
        full_vit_metrics = evaluate_single_model(full_vit_path, ViTClassifier, test_loader, device)
        results['full_vit'] = full_vit_metrics
        print(f"Full ViT AUC-ROC Macro: {full_vit_metrics['auc_roc_macro']:.4f}")
    except Exception as e:
        print(f"Failed to evaluate Full ViT: {e}")
        results['full_vit'] = None

    # Evaluate ViT-RNN
    print("\nEvaluating ViT-RNN model...")
    try:
        from models import ViTTripletBiRNNClassifier
        rnn_vit_metrics = evaluate_single_model(rnn_vit_path, ViTTripletBiRNNClassifier, test_loader, device)
        results['rnn_vit'] = rnn_vit_metrics
        print(f"ViT-RNN AUC-ROC Macro: {rnn_vit_metrics['auc_roc_macro']:.4f}")
    except Exception as e:
        print(f"Failed to evaluate ViT-RNN: {e}")
        results['rnn_vit'] = None

    return results


def evaluate_single_model(model_path, model_class, test_loader, device):
    """Evaluate a single model."""
    from models import ViTTripletBiRNNClassifier

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    if model_class == ViTTripletBiRNNClassifier:
        # Override model_name to avoid relative path issues in stored config
        model_name_override = 'pre-trained-model'
        print(f"  ✓ Overriding model_name to: {model_name_override}")

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

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Evaluate
    metrics_calculator = MetricsCalculator(num_classes=5)
    metrics_calculator.reset()

    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)

            outputs = model(images)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            metrics_calculator.update(logits.cpu(), targets.cpu())

    metrics = metrics_calculator.compute()
    return metrics


def evaluate_ensemble(ensemble, test_loader, device):
    """Evaluate ensemble model."""
    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*60)

    ensemble.eval()
    metrics_calculator = MetricsCalculator(num_classes=5)
    metrics_calculator.reset()

    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)

            logits = ensemble(images)

            metrics_calculator.update(logits.cpu(), targets.cpu())

    metrics = metrics_calculator.compute()
    return metrics


def create_comparison_plot(individual_results, ensemble_results, output_dir):
    """Create comparison plot between individual models and ensemble."""
    models = []
    auc_scores = []

    if individual_results['full_vit']:
        models.append('Full ViT')
        auc_scores.append(individual_results['full_vit']['auc_roc_macro'])

    if individual_results['rnn_vit']:
        models.append('ViT-RNN')
        auc_scores.append(individual_results['rnn_vit']['auc_roc_macro'])

    models.append('Ensemble')
    # Handle both old format (auc_roc_macro) and new format (auc_roc_macro_default)
    ensemble_auc = (ensemble_results.get('auc_roc_macro') or
                   ensemble_results.get('auc_roc_macro_default') or
                   0.0)
    auc_scores.append(ensemble_auc)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, auc_scores, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

    # Add value labels
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC-ROC Macro Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim([min(auc_scores) - 0.01, max(auc_scores) + 0.02])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plot saved to: {os.path.join(output_dir, 'ensemble_comparison.png')}")


def run_comprehensive_inference(ensemble, test_loader, device, output_dir):
    """Run comprehensive inference with threshold optimization and detailed plots."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE INFERENCE ON TEST SET")
    print("="*60)

    ensemble.eval()

    # Initialize MetricsCalculator for threshold-independent metrics (AUC-ROC)
    metrics_calculator = MetricsCalculator(num_classes=5)
    metrics_calculator.reset()

    all_predictions = []
    all_targets = []
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Ensemble Inference]"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = ensemble(images)
            probs = torch.sigmoid(logits)

            # Update MetricsCalculator for threshold-independent metrics
            metrics_calculator.update(logits.cpu(), labels.cpu())

            # Convert to numpy
            preds = (probs >= 0.5).cpu().numpy()
            targets = labels.cpu().numpy()
            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_predictions.extend(preds)
            all_targets.extend(targets)
            all_logits.extend(logits_np)
            all_probs.extend(probs_np)

    # Compute threshold-independent metrics (AUC-ROC)
    threshold_independent_metrics = metrics_calculator.compute()

    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    logits = np.array(all_logits)
    probabilities = np.array(all_probs)

    return predictions, targets, logits, probabilities, threshold_independent_metrics


def find_best_threshold(logits, targets, device):
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
            targets_i = targets[:, i]
            cm = confusion_matrix(targets_i, outputs)
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
        targets_i = targets[:, i]
        cm = confusion_matrix(targets_i, outputs)
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


def create_simple_auc_roc_curves(targets, probabilities, output_dir):
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


def create_auc_roc_curves_with_thresholds(targets, probabilities, output_dir, default_threshold=0.5, best_threshold=None):
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


def create_per_class_metrics_plot(metrics, output_dir):
    """Create bar chart for per-class precision/recall/f1."""
    class_names_plot = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    class_names_key = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    # Extract metrics (handle both formats: with and without suffix)
    def get_metric_value(metric_name):
        # Try with suffix first (e.g., precision_epidural_default)
        if f'{metric_name}_default' in metrics:
            return metrics[f'{metric_name}_default']
        # Fall back to without suffix (e.g., precision_epidural)
        elif metric_name in metrics:
            return metrics[metric_name]
        else:
            return 0.0  # Default fallback

    precisions = [get_metric_value(f'precision_{name}') for name in class_names_key]
    recalls = [get_metric_value(f'recall_{name}') for name in class_names_key]
    f1_scores = [get_metric_value(f'f1_{name}') for name in class_names_key]

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
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Ensemble Evaluation for ICH Classification")
    print("=" * 50)
    print(f"Full ViT model: {args.full_vit_path}")
    print(f"ViT-RNN model: {args.rnn_vit_path}")
    print(f"Ensemble method: {args.method}")
    print(f"Weights: {args.weights}")
    print(f"Output directory: {output_dir}")

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Get device
    device = get_device(args.device) if args.device == 'auto' else torch.device(args.device)
    print(f"Using device: {device}")

    # Create test dataset and loader
    data_root = config['data']['data_root']
    test_dataset = PreprocessedDataset(root_dir=os.path.join(data_root, 'test'))

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # Evaluate individual models
    individual_results = evaluate_individual_models(
        args.full_vit_path, args.rnn_vit_path, test_loader, device
    )

    # Create and evaluate ensemble
    print("\nCreating ensemble model...")
    ensemble = EnsembleViTClassifier(
        full_vit_path=args.full_vit_path,
        rnn_vit_path=args.rnn_vit_path,
        ensemble_method=args.method,
        weights=args.weights,
        device=device
    )

    # Run comprehensive inference on ensemble
    predictions, targets, logits, probabilities, threshold_independent_metrics = run_comprehensive_inference(ensemble, test_loader, device, output_dir)

    # Calculate metrics for default threshold (0.5)
    default_predictions = (probabilities >= 0.5).astype(int)
    default_threshold_metrics = calculate_threshold_metrics(default_predictions, targets, threshold_name="_default")

    # Find best threshold using Youden's J optimization
    best_threshold, best_predictions = find_best_threshold(logits, targets, device)

    # Calculate metrics for best threshold
    best_threshold_metrics = calculate_threshold_metrics(best_predictions, targets, threshold_name="_best")

    # Create comparison plot with individual models
    create_comparison_plot(individual_results, default_threshold_metrics, output_dir)

    # Print results comparison
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS COMPARISON")
    print("="*80)

    if individual_results['full_vit']:
        print(f"Full ViT AUC-ROC Macro: {individual_results['full_vit']['auc_roc_macro']:.4f}")
    if individual_results['rnn_vit']:
        print(f"ViT-RNN AUC-ROC Macro: {individual_results['rnn_vit']['auc_roc_macro']:.4f}")

    print(f"\nEnsemble (threshold-independent):")
    print(f"  AUC-ROC Macro: {threshold_independent_metrics['auc_roc_macro']:.4f}")
    print(f"  AUC-ROC Micro: {threshold_independent_metrics['auc_roc_micro']:.4f}")

    print(f"\nEnsemble Default Threshold (0.5):")
    print(f"  F1 (Macro):           {default_threshold_metrics['f1_macro_default']:.4f}")
    print(f"  Exact Match Accuracy: {default_threshold_metrics['accuracy_exact_default']:.4f}")
    print(f"  Hamming Accuracy:     {default_threshold_metrics['accuracy_hamming_default']:.4f}")

    print(f"\nEnsemble Best Threshold ({best_threshold:.2f}):")
    print(f"  F1 (Macro):           {best_threshold_metrics['f1_macro_best']:.4f}")
    print(f"  Exact Match Accuracy: {best_threshold_metrics['accuracy_exact_best']:.4f}")
    print(f"  Hamming Accuracy:     {best_threshold_metrics['accuracy_hamming_best']:.4f}")

    # Calculate improvement over best individual model
    individual_scores = [r['auc_roc_macro'] for r in individual_results.values() if r is not None]
    if individual_scores:
        best_individual = max(individual_scores)
        improvement = threshold_independent_metrics['auc_roc_macro'] - best_individual
        print(f"\nImprovement over best individual: {improvement:+.4f}")

    # Create comprehensive plots
    print("\nCreating comprehensive plots...")
    create_simple_auc_roc_curves(targets, probabilities, output_dir)
    create_auc_roc_curves_with_thresholds(targets, probabilities, output_dir, 0.5, best_threshold)
    create_per_class_metrics_plot(default_threshold_metrics, output_dir)

    # Save comprehensive results
    results = {
        'ensemble_config': {
            'method': args.method,
            'weights': args.weights,
            'full_vit_path': args.full_vit_path,
            'rnn_vit_path': args.rnn_vit_path
        },
        'individual_results': {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in (v.items() if v else {})}
            for k, v in individual_results.items()
        },
        'ensemble_threshold_independent_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                                 for k, v in threshold_independent_metrics.items()},
        'ensemble_metrics_default_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                             for k, v in default_threshold_metrics.items()},
        'ensemble_metrics_best_threshold': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                           for k, v in best_threshold_metrics.items()},
        'best_threshold': float(best_threshold),
        'config': config
    }

    results_path = os.path.join(output_dir, 'ensemble_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Comprehensive results saved to: {results_path}")
    print(f"✓ All outputs saved to: {output_dir}")

    # Final summary
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION SUMMARY")
    print("="*80)
    print(f"Model Type: Ensemble ({args.method})")
    print(f"Full ViT: {args.full_vit_path}")
    print(f"RNN ViT: {args.rnn_vit_path}")
    print(f"Weights: {args.weights}")
    print(f"Samples: {len(test_dataset)}")
    print(f"Best Threshold: {best_threshold:.4f}")

    print(f"\nAUC-ROC (threshold-independent):")
    print(f"  Macro: {threshold_independent_metrics['auc_roc_macro']:.4f}")
    print(f"  Micro: {threshold_independent_metrics['auc_roc_micro']:.4f}")

    print(f"\nDefault Threshold (0.5):")
    print(f"  F1 (Macro):           {default_threshold_metrics['f1_macro_default']:.4f}")
    print(f"  Exact Match Accuracy: {default_threshold_metrics['accuracy_exact_default']:.4f}")
    print(f"  Hamming Accuracy:     {default_threshold_metrics['accuracy_hamming_default']:.4f}")

    print(f"\nBest Threshold ({best_threshold:.2f}):")
    print(f"  F1 (Macro):           {best_threshold_metrics['f1_macro_best']:.4f}")
    print(f"  Exact Match Accuracy: {best_threshold_metrics['accuracy_exact_best']:.4f}")
    print(f"  Hamming Accuracy:     {best_threshold_metrics['accuracy_hamming_best']:.4f}")

    print("="*80)


if __name__ == '__main__':
    main()
