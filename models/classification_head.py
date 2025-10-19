"""
Classification Head with Attention and MLP

This module implements an advanced classification head that combines
multi-head attention with a multi-layer perceptron for improved
feature learning.
"""

import torch
import torch.nn as nn


class AttentionMLPHead(nn.Module):
    """
    Attention-based MLP classification head.
    
    Combines multi-head self-attention with an MLP to process
    features before classification. Includes residual connections.
    
    Architecture:
        input -> attention -> residual -> MLP layers -> dropout -> output
    
    Args:
        input_dim: Dimension of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
        num_attention_heads: Number of attention heads (default: 8)
        attention_dropout: Dropout rate for attention (default: 0.1)
        dropout: Dropout rate for MLP (default: 0.1)
        use_residual: Whether to use residual connections (default: True)
    """
    
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dims=[512, 256, 128],
        num_attention_heads=8,
        attention_dropout=0.1,
        dropout=0.1,
        use_residual=True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_residual = use_residual
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Layer normalization after attention
        self.norm = nn.LayerNorm(input_dim)
        
        # MLP layers
        mlp_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)
        
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
        # Reshape for attention: [batch, seq_len=1, embed_dim]
        x_reshaped = x.unsqueeze(1)
        
        # Apply self-attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Squeeze back to [batch, embed_dim]
        attn_out = attn_out.squeeze(1)
        
        # Residual connection and layer norm
        if self.use_residual:
            x = self.norm(x + attn_out)
        else:
            x = self.norm(attn_out)
        
        # MLP layers
        x = self.mlp(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits
