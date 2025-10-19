"""
Imbalanced Sampling for Multi-Label Classification

This module implements sampling strategies to handle class imbalance
by oversampling examples with rare classes.
"""

import torch
import numpy as np
from torch.utils.data import Sampler


class ImbalancedMultiLabelSampler(Sampler):
    """
    Sampler that handles class imbalance in multi-label classification.
    
    Computes sampling weights based on the frequency of each class,
    with special boosting for rare classes.
    
    Args:
        dataset: PyTorch dataset with multi-label targets
        strategy: Weighting strategy ('inverse_freq' or 'sqrt_inverse_freq')
        rare_class_boost: Additional multiplier for rare classes (default: 1.0)
        num_samples: Number of samples per epoch (default: len(dataset))
        replacement: Sample with replacement (default: True)
    """
    
    def __init__(
        self,
        dataset,
        strategy='sqrt_inverse_freq',
        rare_class_boost=1.0,
        num_samples=None,
        replacement=True
    ):
        self.dataset = dataset
        self.strategy = strategy
        self.rare_class_boost = rare_class_boost
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        
        # Compute sample weights
        self.weights = self._compute_weights()
    
    def _compute_weights(self):
        """Compute sampling weight for each sample based on class frequencies."""
        # Collect all labels
        labels = []
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            labels.append(label)
        labels = np.array(labels)
        
        # Compute class frequencies
        num_samples = len(labels)
        num_classes = labels.shape[1]
        class_counts = labels.sum(axis=0)
        class_frequencies = class_counts / num_samples
        
        # Compute class weights based on strategy
        if self.strategy == 'inverse_freq':
            # Inverse frequency: w_c = 1 / freq_c
            class_weights = 1.0 / (class_frequencies + 1e-8)
        elif self.strategy == 'sqrt_inverse_freq':
            # Square root of inverse frequency (softer)
            class_weights = np.sqrt(1.0 / (class_frequencies + 1e-8))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Normalize class weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        # Apply rare class boost
        if self.rare_class_boost > 1.0:
            # Identify rare classes (below 10% frequency)
            rare_threshold = 0.1
            rare_classes = class_frequencies < rare_threshold
            class_weights[rare_classes] *= self.rare_class_boost
        
        # Compute sample weights
        # For multi-label: weight = average of weights for all positive classes
        sample_weights = np.zeros(num_samples)
        for i in range(num_samples):
            positive_classes = labels[i] == 1
            if positive_classes.sum() > 0:
                sample_weights[i] = class_weights[positive_classes].mean()
            else:
                # For samples with no positive labels, use minimum weight
                sample_weights[i] = class_weights.min()
        
        return torch.from_numpy(sample_weights).double()
    
    def __iter__(self):
        """Generate indices for sampling."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples
