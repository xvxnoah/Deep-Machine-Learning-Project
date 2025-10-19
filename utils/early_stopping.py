"""
Early stopping utility to prevent overfitting.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience=7,
        min_delta=0.0,
        mode='max',
        verbose=True,
        restore_best_weights=True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored metric to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy, AUC), 'min' for metrics to minimize (loss)
            verbose: Whether to print messages
            restore_best_weights: Whether to restore model to best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        elif mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
    
    def __call__(self, current_score, model, epoch):
        """
        Check if training should stop.
        
        Args:
            current_score: Current value of monitored metric
            model: Model to save state from
            epoch: Current epoch number
            
        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            return False
        
        # Check if improved
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            # Improvement
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f"  → Validation metric improved to {current_score:.4f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  → No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"Early stopping triggered!")
                    print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                    print(f"{'='*80}\n")
                
                # Restore best weights if requested
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print(f"Restored model to best weights from epoch {self.best_epoch}")
                
                return True
        
        return False
    
    def save_checkpoint(self, model):
        """Save model state."""
        if self.restore_best_weights:
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None

