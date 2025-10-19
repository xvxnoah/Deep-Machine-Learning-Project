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
    def __init__(self, num_classes=5, class_names=None, threshold=0.5, thresholds=None):
        self.num_classes = num_classes
        self.threshold = threshold
        # Optional per-class thresholds. If provided, overrides single threshold.
        self.thresholds = None if thresholds is None else np.array(thresholds, dtype=float)
        
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
        if self.thresholds is not None:
            thr = self.thresholds.reshape(1, -1)
            preds = (probs >= thr).astype(int)
        else:
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
        targets = np.concatenate(self.all_targets, axis=0)
        
        # Get probabilities and derive predictions using current thresholds
        probs = 1 / (1 + np.exp(-logits))
        if self.thresholds is not None:
            thr = self.thresholds.reshape(1, -1)
            predictions = (probs >= thr).astype(int)
        else:
            predictions = (probs >= self.threshold).astype(int)
        
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

    def set_thresholds(self, thresholds):
        """Set per-class thresholds to use for binarization.
        Args:
            thresholds (array-like): Shape (num_classes,)
        """
        thresholds = np.array(thresholds, dtype=float)
        if thresholds.shape[0] != self.num_classes:
            raise ValueError(f"thresholds must have length {self.num_classes}")
        self.thresholds = thresholds

    def optimize_thresholds(self, method='f1', num_points=101, per_class=True, min_thr=0.05, max_thr=0.95):
        """Optimize thresholds based on accumulated logits/targets.
        Args:
            method (str): 'f1' (currently supported)
            num_points (int): Number of grid points between min_thr and max_thr
            per_class (bool): If True, optimize each class independently
            min_thr (float): Minimum threshold to consider
            max_thr (float): Maximum threshold to consider
        Returns:
            np.ndarray: Optimal thresholds of shape (num_classes,)
        """
        if len(self.all_logits) == 0:
            raise RuntimeError("No data accumulated. Call update() before optimizing thresholds.")
        logits = np.concatenate(self.all_logits, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)
        probs = 1 / (1 + np.exp(-logits))

        grid = np.linspace(min_thr, max_thr, num_points)
        best_thresholds = np.full(self.num_classes, 0.5, dtype=float)

        if method != 'f1':
            raise ValueError(f"Unsupported method: {method}")

        if per_class:
            for i in range(self.num_classes):
                # Skip if only one class present to avoid undefined F1
                if len(np.unique(targets[:, i])) < 2:
                    best_thresholds[i] = 0.5
                    continue
                best_f1 = -1.0
                best_t = 0.5
                y_true = targets[:, i]
                p = probs[:, i]
                for t in grid:
                    y_pred = (p >= t).astype(int)
                    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
                best_thresholds[i] = best_t
        else:
            # Single global threshold optimized for macro F1
            best_f1 = -1.0
            best_t = 0.5
            for t in grid:
                y_pred = (probs >= t).astype(int)
                _, _, f1_macro, _ = precision_recall_fscore_support(targets, y_pred, average='macro', zero_division=0)
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_t = t
            best_thresholds.fill(best_t)

        return best_thresholds

    def compute_with_thresholds(self, thresholds):
        """Convenience to compute metrics using provided thresholds without mutating state."""
        old = self.thresholds
        try:
            self.set_thresholds(thresholds)
            return self.compute()
        finally:
            self.thresholds = old
    
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
    
    print(f"{'='*80}\n")

