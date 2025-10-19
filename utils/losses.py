"""
Loss Functions for Multi-Label Classification

This module implements a combined loss function that merges:
- Binary Cross-Entropy Loss (with class weights)
- Focal Loss (with per-class alpha for rare classes)
- Asymmetric Loss (with per-class gamma for handling false negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-label classification.
    
    Combines three loss components:
    1. Weighted BCE: Handles class imbalance with pos_weight
    2. Focal Loss: Focuses on hard examples, with per-class alpha for rare classes
    3. Asymmetric Loss: Reduces false negatives with per-class gamma
    
    Args:
        loss_weights: Dictionary with weights for each component {'bce': float, 'focal': float, 'asymmetric': float}
        pos_weight: Tensor of positive class weights for BCE
        focal_alpha: Alpha parameter for Focal Loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0)
        per_class_focal_alpha: Tensor of per-class alpha values (optional)
        label_smoothing: Label smoothing factor (default: 0.0)
        asymmetric_gamma_neg: Gamma for negative samples in Asymmetric Loss (default: 4)
        asymmetric_gamma_pos: Gamma for positive samples in Asymmetric Loss (default: 1)
        per_class_gamma_neg: Tensor of per-class gamma_neg values (optional)
        per_class_gamma_pos: Tensor of per-class gamma_pos values (optional)
        class_weights: Tensor of class weights for Asymmetric Loss (optional)
    """
    
    def __init__(
        self,
        loss_weights={'bce': 0.4, 'focal': 0.3, 'asymmetric': 0.3},
        pos_weight=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
        per_class_focal_alpha=None,
        label_smoothing=0.0,
        asymmetric_gamma_neg=4,
        asymmetric_gamma_pos=1,
        per_class_gamma_neg=None,
        per_class_gamma_pos=None,
        class_weights=None
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.per_class_focal_alpha = per_class_focal_alpha
        self.label_smoothing = label_smoothing
        self.asymmetric_gamma_neg = asymmetric_gamma_neg
        self.asymmetric_gamma_pos = asymmetric_gamma_pos
        self.per_class_gamma_neg = per_class_gamma_neg
        self.per_class_gamma_pos = per_class_gamma_pos
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        """
        Compute combined loss.
        
        Args:
            logits: Model outputs (before sigmoid) [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        
        Returns:
            Combined weighted loss
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        total_loss = 0.0
        
        # 1. Binary Cross-Entropy Loss
        if self.loss_weights.get('bce', 0) > 0:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='mean'
            )
            total_loss += self.loss_weights['bce'] * bce_loss
        
        # 2. Focal Loss
        if self.loss_weights.get('focal', 0) > 0:
            focal_loss = self._focal_loss(logits, targets, probs)
            total_loss += self.loss_weights['focal'] * focal_loss
        
        # 3. Asymmetric Loss
        if self.loss_weights.get('asymmetric', 0) > 0:
            asymmetric_loss = self._asymmetric_loss(logits, targets, probs)
            total_loss += self.loss_weights['asymmetric'] * asymmetric_loss
        
        return total_loss
    
    def _focal_loss(self, logits, targets, probs):
        """
        Compute Focal Loss.
        
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        where p_t = p if y=1, else 1-p
        """
        # Compute cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.focal_gamma
        
        # Apply alpha weighting (per-class if specified)
        if self.per_class_focal_alpha is not None:
            alpha_t = self.per_class_focal_alpha * targets + (1 - self.per_class_focal_alpha) * (1 - targets)
        else:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_term * bce_loss
        
        return focal_loss.mean()
    
    def _asymmetric_loss(self, logits, targets, probs):
        """
        Compute Asymmetric Loss.
        
        Asymmetric focusing: different gamma for positive and negative samples.
        Reduces false negatives by down-weighting easy negatives more than easy positives.
        """
        # Split into positive and negative samples
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)
        
        # Get gamma values (per-class if specified)
        if self.per_class_gamma_neg is not None:
            gamma_neg = self.per_class_gamma_neg
        else:
            gamma_neg = self.asymmetric_gamma_neg
            
        if self.per_class_gamma_pos is not None:
            gamma_pos = self.per_class_gamma_pos
        else:
            gamma_pos = self.asymmetric_gamma_pos
        
        # Compute loss for positive samples
        pos_loss = -(1 - probs) ** gamma_pos * torch.log(probs.clamp(min=1e-8))
        pos_loss = pos_loss * pos_mask
        
        # Compute loss for negative samples
        neg_loss = -probs ** gamma_neg * torch.log((1 - probs).clamp(min=1e-8))
        neg_loss = neg_loss * neg_mask
        
        # Combine
        loss = pos_loss + neg_loss
        
        # Apply class weights if specified
        if self.class_weights is not None:
            loss = loss * self.class_weights
        
        return loss.mean()
