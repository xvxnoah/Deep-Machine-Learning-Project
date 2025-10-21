"""
Simple MLP Classification Head

A straightforward multi-layer perceptron matching the ResNet colleague's
successful architecture.
"""

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple MLP classification head.
    
    Architecture matches the successful ResNet implementation:
        input_dim -> 128 -> 32 -> num_classes
    
    Args:
        input_dim: Dimension of input features (768 for ViT-base)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dim=768, num_classes=5, dropout=0.3):
        super().__init__()
        
        # No Flatten needed - ViT output is already [batch, 768]
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        return self.classifier(x)

