"""
Test-Time Augmentation (TTA) for Robust Predictions

This module implements test-time augmentation by applying multiple
transformations to test images and averaging the predictions.
"""

import torch
import torch.nn.functional as F


class TTAWrapper:
    """
    Test-Time Augmentation wrapper for a model.
    
    Applies multiple augmentations to input images and combines predictions
    for more robust results.
    
    Args:
        model: PyTorch model to wrap
        num_classes: Number of output classes
        device: Device to run inference on
    """
    
    def __init__(self, model, num_classes, device='cpu'):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        
    def predict_with_tta(self, x, augmentations='standard', combine='mean'):
        """
        Predict with test-time augmentation.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            augmentations: TTA strategy ('standard')
            combine: How to combine predictions ('mean', 'median', or 'vote')
        
        Returns:
            Combined predictions [batch_size, num_classes]
        """
        self.model.eval()
        
        # Get augmentation list
        aug_list = self._get_augmentations(augmentations)
        
        # Collect predictions from all augmentations
        all_predictions = []
        
        with torch.no_grad():
            # Original
            logits_orig = self.model(x)
            all_predictions.append(torch.sigmoid(logits_orig))
            
            # Augmented versions
            for name, aug_fn in aug_list:
                x_aug = aug_fn(x.clone())
                logits_aug = self.model(x_aug)
                all_predictions.append(torch.sigmoid(logits_aug))
        
        # Combine predictions
        all_predictions = torch.stack(all_predictions, dim=0)  # [num_augs, batch, num_classes]
        
        if combine == 'mean':
            final_pred = all_predictions.mean(dim=0)
        elif combine == 'median':
            final_pred = all_predictions.median(dim=0)[0]
        elif combine == 'vote':
            # Hard voting: threshold at 0.5, then take majority
            binary_preds = (all_predictions > 0.5).float()
            final_pred = (binary_preds.mean(dim=0) > 0.5).float()
        else:
            raise ValueError(f"Unknown combine method: {combine}")
        
        return final_pred
    
    def _get_augmentations(self, strategy):
        """
        Get list of augmentation functions for the given strategy.
        
        Args:
            strategy: TTA strategy name
        
        Returns:
            List of (name, augmentation_function) tuples
        """
        if strategy == 'standard':
            return [
                ('horizontal_flip', self._horizontal_flip),
                ('vertical_flip', self._vertical_flip),
                ('rotate_90', lambda x: self._rotate(x, 90)),
                ('rotate_270', lambda x: self._rotate(x, 270)),
            ]
        else:
            raise ValueError(f"Unknown TTA strategy: {strategy}")
    
    def _horizontal_flip(self, x):
        """Flip image horizontally."""
        return torch.flip(x, dims=[-1])
    
    def _vertical_flip(self, x):
        """Flip image vertically."""
        return torch.flip(x, dims=[-2])
    
    def _rotate(self, x, angle):
        """
        Rotate image by specified angle (90, 180, or 270 degrees).
        
        Args:
            x: Input tensor [batch, channels, height, width]
            angle: Rotation angle in degrees (90, 180, 270)
        
        Returns:
            Rotated tensor
        """
        if angle == 90:
            return torch.rot90(x, k=1, dims=[-2, -1])
        elif angle == 180:
            return torch.rot90(x, k=2, dims=[-2, -1])
        elif angle == 270:
            return torch.rot90(x, k=3, dims=[-2, -1])
        else:
            raise ValueError(f"Unsupported rotation angle: {angle}")
