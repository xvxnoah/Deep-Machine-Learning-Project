import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for handling class imbalance."""
    
    def __init__(self, pos_weights=None, reduction='mean'):
        super().__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
        
    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weights,
            reduction=self.reduction
        )
        return loss


def compute_pos_weights(labels, strategy='inverse_freq', beta=0.999):
    # Calculate positive sample counts per class
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    if strategy == 'inverse_freq':
        # Inverse frequency: neg_count / pos_count
        weights = neg_counts / (pos_counts + 1e-5)
        
    elif strategy == 'effective_samples':
        # Effective number of samples: (1-β) / (1-β^n)
        effective_num = 1.0 - np.power(beta, pos_counts)
        weights = (1.0 - beta) / (effective_num + 1e-5)
        weights = weights / weights.sum() * len(labels)
        
    elif strategy == 'sqrt_inverse_freq':
        # Square root of inverse frequency (less aggressive)
        weights = np.sqrt(neg_counts / (pos_counts + 1e-5))
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return torch.FloatTensor(weights)
