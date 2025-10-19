"""
Channel Adapter for Multi-Channel Medical Images

This module adapts multi-channel inputs (e.g., 9-channel CT scans)
to the 3-channel RGB format expected by pre-trained vision models.
"""

import torch
import torch.nn as nn


class DeepConvAdapter(nn.Module):
    """
    Deep convolutional channel adapter.
    
    Uses multiple convolutional layers with batch normalization and ReLU
    to progressively reduce channels while learning spatial patterns.
    
    Architecture:
        input_channels -> 64 -> 32 -> 16 -> 3 channels
    
    Args:
        input_channels: Number of input channels (e.g., 9 for CT triplets)
    """
    
    def __init__(self, input_channels=9):
        super().__init__()
        
        self.adapter = nn.Sequential(
            # Layer 1: input_channels -> 64
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2: 64 -> 32
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 3: 32 -> 16
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Layer 4: 16 -> 3 (RGB)
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
        
        Returns:
            Adapted tensor [batch_size, 3, height, width]
        """
        return self.adapter(x)
