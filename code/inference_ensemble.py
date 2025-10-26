import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (roc_curve, auc, confusion_matrix, roc_auc_score,
                             precision_recall_fscore_support)

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
    print("\nEvaluating Individual Models")
    print("-" * 30)

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
        print(f"  Overriding model_name to: {model_name_override}")

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
    print("\nEvaluating Ensemble Model")
    print("-" * 25)

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


def run_comprehensive_inference(ensemble, test_loader, device, output_dir):
    """Run comprehensive inference with threshold optimization and detailed plots."""
    print("\nRunning Comprehensive Inference on Test Set")
    print("-" * 42)

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




def calculate_threshold_metrics(predictions, targets):
    """Calculate metrics for given predictions and targets, including all 7 categories."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

    # Calculate metrics for all 7 categories as separate binary classification problems
    category_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy', 'multiple']
    category_targets = []
    category_predictions = []
    category_supports = []

    # Original 5 hemorrhage types
    for i in range(5):
        binary_targets = targets[:, i]
        binary_predictions = predictions[:, i]
        category_targets.append(binary_targets)
        category_predictions.append(binary_predictions)
        category_supports.append(binary_targets.sum())

    # Healthy (no hemorrhages)
    healthy_targets = (targets.sum(axis=1) == 0).astype(int)
    healthy_predictions = (predictions.sum(axis=1) == 0).astype(int)
    category_targets.append(healthy_targets)
    category_predictions.append(healthy_predictions)
    category_supports.append(healthy_targets.sum())

    # Multiple hemorrhages
    multiple_targets = (targets.sum(axis=1) > 1).astype(int)
    multiple_predictions = (predictions.sum(axis=1) > 1).astype(int)
    category_targets.append(multiple_targets)
    category_predictions.append(multiple_predictions)
    category_supports.append(multiple_targets.sum())

    # Calculate precision, recall, f1 for each category
    precisions = []
    recalls = []
    f1s = []

    for targets_cat, preds_cat in zip(category_targets, category_predictions):
        p, r, f1, _ = precision_recall_fscore_support(targets_cat, preds_cat, average='binary', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # Convert to numpy arrays for easier calculations
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(category_supports)

    # Macro averages (simple average across all 7 categories)
    precision_macro = precisions.mean()
    recall_macro = recalls.mean()
    f1_macro = f1s.mean()

    # Micro averages (weighted by support, then averaged)
    total_support = supports.sum()
    if total_support > 0:
        precision_micro = (precisions * supports).sum() / total_support
        recall_micro = (recalls * supports).sum() / total_support
        f1_micro = (f1s * supports).sum() / total_support
    else:
        precision_micro = recall_micro = f1_micro = 0.0

    # Weighted averages (weighted by support)
    if supports.sum() > 0:
        precision_weighted = (precisions * supports).sum() / supports.sum()
        recall_weighted = (recalls * supports).sum() / supports.sum()
        f1_weighted = (f1s * supports).sum() / supports.sum()
    else:
        precision_weighted = recall_weighted = f1_weighted = 0.0

    # Accuracy metrics (keep the original multi-class accuracies)
    exact_match_acc = accuracy_score(targets, predictions)
    hamming_acc = 1 - hamming_loss(targets, predictions)

    # Build metrics dictionary
    metrics = {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'accuracy_exact': exact_match_acc,
        'accuracy_hamming': hamming_acc,
    }

    # Add per-category metrics
    for i, name in enumerate(category_names):
        metrics[f'precision_{name}'] = precisions[i]
        metrics[f'recall_{name}'] = recalls[i]
        metrics[f'f1_{name}'] = f1s[i]

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
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'simple_auc_roc_curves.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Simple AUC-ROC curves plot saved to: {plot_path}")




def create_per_class_metrics_plot(metrics, output_dir):
    """Create bar chart for per-class precision/recall/f1."""
    class_names_plot = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
    class_names_key = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    # Extract metrics
    precisions = [metrics.get(f'precision_{name}', 0.0) for name in class_names_key]
    recalls = [metrics.get(f'recall_{name}', 0.0) for name in class_names_key]
    f1_scores = [metrics.get(f'f1_{name}', 0.0) for name in class_names_key]

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
    plot_path = os.path.join(output_dir, 'per_class_metrics.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Per-class metrics plot saved to: {plot_path}")


def create_confusion_matrix_heatmap(predictions, targets, output_dir):
    """Create a confusion matrix heatmap for multi-class classification."""
    class_names = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']

    # Calculate confusion matrix
    cm = confusion_matrix(targets.argmax(axis=1), predictions.argmax(axis=1))

    # Normalize by true labels (rows)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, annot_kws={"fontsize": 12})

    # Increase font size for tick labels
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'confusion_matrix_heatmap.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix heatmap saved to: {plot_path}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Ensemble Evaluation for ICH Classification")
    print("-" * 42)
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
    default_threshold_metrics = calculate_threshold_metrics(default_predictions, targets)



    # Print results comparison
    print("\nEnsemble Results Comparison")
    print("-" * 26)

    if individual_results['full_vit']:
        print(f"Full ViT AUC-ROC Macro: {individual_results['full_vit']['auc_roc_macro']:.4f}")
    if individual_results['rnn_vit']:
        print(f"ViT-RNN AUC-ROC Macro: {individual_results['rnn_vit']['auc_roc_macro']:.4f}")

    print(f"\nEnsemble (threshold-independent):")
    print(f"  AUC-ROC Macro: {threshold_independent_metrics['auc_roc_macro']:.4f}")
    print(f"  AUC-ROC Micro: {threshold_independent_metrics['auc_roc_micro']:.4f}")

    print(f"\nEnsemble Default Threshold (0.5):")
    print(f"  F1 (Macro):           {default_threshold_metrics['f1_macro']:.4f}")
    print(f"  Exact Match Accuracy: {default_threshold_metrics['accuracy_exact']:.4f}")
    print(f"  Hamming Accuracy:     {default_threshold_metrics['accuracy_hamming']:.4f}")

    # Calculate improvement over best individual model
    individual_scores = [r['auc_roc_macro'] for r in individual_results.values() if r is not None]
    if individual_scores:
        best_individual = max(individual_scores)
        improvement = threshold_independent_metrics['auc_roc_macro'] - best_individual
        print(f"\nImprovement over best individual: {improvement:+.4f}")

    # Create comprehensive plots
    print("\nCreating comprehensive plots...")
    create_simple_auc_roc_curves(targets, probabilities, output_dir)
    create_per_class_metrics_plot(default_threshold_metrics, output_dir)

    # Additional insightful visualizations for report
    print("\nCreating additional visualizations...")
    create_confusion_matrix_heatmap(default_predictions, targets, output_dir)

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
        'config': config
    }

    results_path = os.path.join(output_dir, 'ensemble_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComprehensive results saved to: {results_path}")
    print(f"All outputs saved to: {output_dir}")

    # Final summary
    print("\nEnsemble Evaluation Summary")
    print("-" * 27)
    print(f"Model Type: Ensemble ({args.method})")
    print(f"Full ViT: {args.full_vit_path}")
    print(f"RNN ViT: {args.rnn_vit_path}")
    print(f"Weights: {args.weights}")
    print(f"Samples: {len(test_dataset)}")

    print(f"\nAUC-ROC (threshold-independent):")
    print(f"  Macro: {threshold_independent_metrics['auc_roc_macro']:.4f}")
    print(f"  Micro: {threshold_independent_metrics['auc_roc_micro']:.4f}")

    print(f"\nDefault Threshold (0.5):")
    print(f"  F1 (Macro):           {default_threshold_metrics['f1_macro']:.4f}")
    print(f"  Exact Match Accuracy: {default_threshold_metrics['accuracy_exact']:.4f}")
    print(f"  Hamming Accuracy:     {default_threshold_metrics['accuracy_hamming']:.4f}")


if __name__ == '__main__':
    main()
