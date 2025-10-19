"""
Data Augmentation for Brain Hemorrhage Classification

This module implements data augmentation strategies including:
- Spatial transformations (rotation, translation, scaling, flips)
- Intensity transformations (shift, scale, noise)
- Mixup and CutMix for regularization
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np


def create_transforms(config, split='train'):
    """
    Create transforms from configuration.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', or 'test')
    
    Returns:
        Transform function
    """
    if split == 'train':
        aug_config = config.get('augmentation', {})
        return get_train_augmentation(
            rotation_degrees=aug_config.get('rotation_degrees', 15),
            translate=aug_config.get('translate', [0.15, 0.15]),
            scale_range=aug_config.get('scale_range', [0.85, 1.15]),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            vertical_flip=aug_config.get('vertical_flip', False),
            intensity_shift=aug_config.get('intensity_shift', 0.15),
            intensity_scale=aug_config.get('intensity_scale', 0.15),
            noise_std=aug_config.get('noise_std', 0.02),
            aug_prob=aug_config.get('aug_prob', 0.6),
            use_random_erasing=aug_config.get('use_random_erasing', True),
            random_erasing_prob=aug_config.get('random_erasing_prob', 0.3)
        )
    else:
        # Validation and test: no augmentation
        return T.Compose([T.ToTensor()])


def get_train_augmentation(
    rotation_degrees=15,
    translate=[0.15, 0.15],
    scale_range=[0.85, 1.15],
    horizontal_flip=True,
    vertical_flip=False,
    intensity_shift=0.15,
    intensity_scale=0.15,
    noise_std=0.02,
    aug_prob=0.6,
    use_random_erasing=True,
    random_erasing_prob=0.3
):
    """
    Get training augmentation pipeline.
    
    Returns:
        Composed transform
    """
    transforms = []
    
    # Spatial augmentations
    transforms.append(T.RandomAffine(
        degrees=rotation_degrees,
        translate=translate,
        scale=scale_range,
        interpolation=T.InterpolationMode.BILINEAR
    ))
    
    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    
    if vertical_flip:
        transforms.append(T.RandomVerticalFlip(p=0.5))
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Intensity augmentations
    transforms.append(T.RandomApply([
        IntensityAugmentation(
            intensity_shift=intensity_shift,
            intensity_scale=intensity_scale,
            noise_std=noise_std
        )
    ], p=aug_prob))
    
    # Random erasing
    if use_random_erasing:
        transforms.append(T.RandomErasing(
            p=random_erasing_prob,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random'
        ))
    
    return T.Compose(transforms)


class IntensityAugmentation(nn.Module):
    """
    Intensity-based augmentation for medical images.
    
    Applies random intensity shift, scaling, and Gaussian noise.
    """
    
    def __init__(self, intensity_shift=0.1, intensity_scale=0.1, noise_std=0.01):
        super().__init__()
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std = noise_std
    
    def forward(self, img):
        """
        Apply intensity augmentation.
        
        Args:
            img: Image tensor [C, H, W]
        
        Returns:
            Augmented image tensor
        """
        # Random intensity shift
        if self.intensity_shift > 0:
            shift = torch.rand(1).item() * 2 * self.intensity_shift - self.intensity_shift
            img = img + shift
        
        # Random intensity scaling
        if self.intensity_scale > 0:
            scale = 1 + (torch.rand(1).item() * 2 * self.intensity_scale - self.intensity_scale)
            img = img * scale
        
        # Random Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
        
        # Clip to valid range
        img = torch.clamp(img, 0, 1)
        
        return img


class MixupCutmix(nn.Module):
    """
    Combined Mixup and CutMix augmentation.
    
    Randomly applies either Mixup or CutMix to a batch of images.
    
    Args:
        mixup_alpha: Alpha parameter for Mixup (default: 0.3)
        cutmix_alpha: Alpha parameter for CutMix (default: 1.0)
        num_classes: Number of classes (default: 5)
        prob: Probability of applying augmentation (default: 0.6)
    """
    
    def __init__(self, mixup_alpha=0.3, cutmix_alpha=1.0, num_classes=5, prob=0.6):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        self.prob = prob
    
    def __call__(self, images, labels):
        """
        Apply Mixup or CutMix.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B, num_classes]
        
        Returns:
            Augmented images and labels
        """
        # Decide whether to apply augmentation
        if torch.rand(1).item() > self.prob:
            return images, labels
        
        # Randomly choose Mixup or CutMix
        if torch.rand(1).item() < 0.5:
            return self._mixup(images, labels)
        else:
            return self._cutmix(images, labels)
    
    def _mixup(self, images, labels):
        """Apply Mixup augmentation."""
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels
    
    def _cutmix(self, images, labels):
        """Apply CutMix augmentation."""
        batch_size = images.size(0)
        _, _, H, W = images.shape
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Generate random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels


# Backward compatibility - old function names
class Mixup(MixupCutmix):
    """Backward compatibility wrapper for Mixup."""
    def __init__(self, alpha=0.3, num_classes=5):
        super().__init__(mixup_alpha=alpha, cutmix_alpha=0, num_classes=num_classes, prob=1.0)


class CutMix(MixupCutmix):
    """Backward compatibility wrapper for CutMix."""
    def __init__(self, alpha=1.0, num_classes=5):
        super().__init__(mixup_alpha=0, cutmix_alpha=alpha, num_classes=num_classes, prob=1.0)
