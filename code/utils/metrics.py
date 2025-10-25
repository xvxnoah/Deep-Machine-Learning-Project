import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
    confusion_matrix
)
import warnings


class MetricsCalculator:
    def __init__(self, num_classes=5, class_names=None, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        
        if class_names is None:
            self.class_names = [
                'epidural',
                'intraparenchymal', 
                'intraventricular',
                'subarachnoid',
                'subdural'
            ]
        else:
            self.class_names = class_names
            
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_logits = []
    
    def update(self, logits, targets):
        # Convert to numpy
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Get predictions
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        preds = (probs >= self.threshold).astype(int)
        
        self.all_logits.append(logits)
        self.all_predictions.append(preds)
        self.all_targets.append(targets)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary of metrics
        """
        if len(self.all_predictions) == 0:
            return {}
        
        # Concatenate all batches
        logits = np.concatenate(self.all_logits, axis=0)
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        # Get probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        metrics = {}
        
        try:
            # Per-class AUC
            auc_per_class = []
            for i in range(self.num_classes):
                if len(np.unique(targets[:, i])) > 1:  # Need both classes present
                    auc = roc_auc_score(targets[:, i], probs[:, i])
                    auc_per_class.append(auc)
                    metrics[f'auc_roc_{self.class_names[i]}'] = auc
                else:
                    auc_per_class.append(np.nan)

            # Macro AUC (average of per-class)
            valid_aucs = [x for x in auc_per_class if not np.isnan(x)]
            if valid_aucs:
                metrics['auc_roc_macro'] = np.mean(valid_aucs)

            # Micro AUC (global)
            if targets.sum() > 0:  # At least one positive
                metrics['auc_roc_micro'] = roc_auc_score(
                    targets.ravel(), probs.ravel()
                )

            # Weighted AUC
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    metrics['auc_roc_weighted'] = roc_auc_score(
                        targets, probs, average='weighted'
                    )
                except:
                    pass

            # Healthy cases: cases with no hemorrhages (sum of all labels == 0)
            healthy_targets = (targets.sum(axis=1) == 0).astype(int)
            if len(np.unique(healthy_targets)) > 1:
                # Use the minimum probability across all classes as the "healthy" score
                healthy_probs = 1 - probs.max(axis=1)  # Higher when all probs are low
                metrics['auc_roc_healthy'] = roc_auc_score(healthy_targets, healthy_probs)

            # Multiple hemorrhages: cases with more than one hemorrhage (sum > 1)
            multiple_targets = (targets.sum(axis=1) > 1).astype(int)
            if len(np.unique(multiple_targets)) > 1:
                # Use the maximum probability across all classes as the "multiple" score
                multiple_probs = probs.max(axis=1)
                metrics['auc_roc_multiple'] = roc_auc_score(multiple_targets, multiple_probs)

        except Exception as e:
            print(f"Warning: Could not compute AUC-ROC: {e}")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Per-class metrics
        for i in range(self.num_classes):
            metrics[f'precision_{self.class_names[i]}'] = precision[i]
            metrics[f'recall_{self.class_names[i]}'] = recall[i]
            metrics[f'f1_{self.class_names[i]}'] = f1[i]
        
        # Macro averages (unweighted mean)
        metrics['precision_macro'] = precision.mean()
        metrics['recall_macro'] = recall.mean()
        metrics['f1_macro'] = f1.mean()
        
        # Micro averages (global)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            targets, predictions, average='micro', zero_division=0
        )
        metrics['precision_micro'] = p_micro
        metrics['recall_micro'] = r_micro
        metrics['f1_micro'] = f1_micro
        
        # Weighted averages (by support)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = p_weighted
        metrics['recall_weighted'] = r_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # Exact match accuracy (all labels must match)
        metrics['accuracy_exact'] = accuracy_score(targets, predictions)
        
        # Hamming accuracy (per-label accuracy)
        metrics['accuracy_hamming'] = 1 - hamming_loss(targets, predictions)
        
        # Per-sample metrics
        metrics['avg_num_positives_pred'] = predictions.sum(axis=1).mean()
        metrics['avg_num_positives_true'] = targets.sum(axis=1).mean()
        
        return metrics
    
    def get_confusion_matrices(self):
        if len(self.all_predictions) == 0:
            return {}
        
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        
        cms = {}
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(targets[:, i], predictions[:, i])
            cms[class_name] = cm
        
        return cms


def compute_class_weights(dataset, strategy='inverse_freq'):
    print("Computing class weights from dataset...")
    
    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)
    
    all_labels = np.array(all_labels)
    
    # Calculate weights
    from .losses import compute_pos_weights
    weights = compute_pos_weights(all_labels, strategy=strategy)
    
    # Print class distribution
    pos_counts = all_labels.sum(axis=0)
    total = len(all_labels)
    
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 
                   'subarachnoid', 'subdural']
    
    print("\nClass Distribution:")
    print("-" * 60)
    for i, name in enumerate(class_names):
        pos = int(pos_counts[i])
        neg = total - pos
        pos_pct = (pos / total) * 100
        print(f"{name:20s}: {pos:5d} pos ({pos_pct:5.2f}%), {neg:5d} neg, weight: {weights[i]:.4f}")
    print("-" * 60)
    
    return weights


def log_classification_report(metrics, prefix=''):
    class_names = ['epidural', 'intraparenchymal', 'intraventricular',
                   'subarachnoid', 'subdural']

    print(f"\n{'='*80}")
    print(f"{prefix.upper()} CLASSIFICATION REPORT")
    print(f"{'='*80}")

    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Exact Match Accuracy: {metrics.get(f'accuracy_exact', 0):.4f}")
    print(f"  Hamming Accuracy:     {metrics.get(f'accuracy_hamming', 0):.4f}")
    print(f"  AUC-ROC (Macro):      {metrics.get(f'auc_roc_macro', 0):.4f}")
    print(f"  AUC-ROC (Micro):      {metrics.get(f'auc_roc_micro', 0):.4f}")
    print(f"  F1 (Macro):           {metrics.get(f'f1_macro', 0):.4f}")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'AUC-ROC':>10s}")
    print("-" * 70)

    for name in class_names:
        prec = metrics.get(f'precision_{name}', 0)
        rec = metrics.get(f'recall_{name}', 0)
        f1 = metrics.get(f'f1_{name}', 0)
        auc = metrics.get(f'auc_roc_{name}', 0)
        print(f"{name:<20s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    # Additional AUC metrics
    print(f"\nSpecial Case AUC-ROC:")
    print(f"  Healthy (No Hemorrhages): {metrics.get('auc_roc_healthy', 0):.4f}")
    print(f"  Multiple Hemorrhages:      {metrics.get('auc_roc_multiple', 0):.4f}")

    print(f"{'='*80}\n")

