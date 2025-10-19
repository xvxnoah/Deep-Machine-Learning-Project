"""
Vision Transformer for Brain Hemorrhage Classification

This module implements a ViT-based classifier with:
- Deep convolutional channel adapter (9 -> 3 channels)
- Pre-trained ViT backbone (frozen during training)
- Attention-MLP classification head
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from .channel_adapter import DeepConvAdapter
from .classification_head import AttentionMLPHead


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier for intracranial hemorrhage detection.
    
    Architecture:
        Input (9 channels) -> DeepConvAdapter (3 channels) -> ViT (frozen) -> AttentionMLPHead -> Output (5 classes)
    
    Args:
        model_name: HuggingFace ViT model name (default: 'google/vit-base-patch16-224')
        num_classes: Number of output classes (default: 5)
        pretrained: Use pretrained weights (default: True)
        input_channels: Number of input channels (default: 9)
        dropout: Dropout rate for classification head (default: 0.1)
        head_hidden_dims: Hidden dimensions for MLP head (default: [512, 256, 128])
        head_num_attention_heads: Number of attention heads (default: 8)
        head_attention_dropout: Dropout for attention (default: 0.1)
        head_use_residual: Use residual connections (default: True)
    """
    
    def __init__(
        self,
        model_name='google/vit-base-patch16-224',
        num_classes=5,
        pretrained=True,
        input_channels=9,
        dropout=0.1,
        head_hidden_dims=[512, 256, 128],
        head_num_attention_heads=8,
        head_attention_dropout=0.1,
        head_use_residual=True,
        # Unused legacy parameters (kept for backward compatibility with old configs)
        channel_adaptation=None,
        backend=None,
        head_type=None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Channel adapter (9 -> 3 channels)
        if input_channels != 3:
            self.channel_adapter = DeepConvAdapter(input_channels=input_channels)
        else:
            self.channel_adapter = nn.Identity()
        
        # Load pretrained ViT backbone
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            from transformers import ViTConfig
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        
        self.embed_dim = self.vit.config.hidden_size
        
        # Classification head
        self.head = AttentionMLPHead(
            input_dim=self.embed_dim,
            num_classes=num_classes,
            hidden_dims=head_hidden_dims,
            num_attention_heads=head_num_attention_heads,
            attention_dropout=head_attention_dropout,
            dropout=dropout,
            use_residual=head_use_residual
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Adapt channels if needed
        x = self.channel_adapter(x)
        
        # ViT backbone
        outputs = self.vit(pixel_values=x)
        
        # Get [CLS] token embedding
        features = outputs.last_hidden_state[:, 0]
        
        # Classification head
        logits = self.head(features)
        
        return logits
    
    def freeze_backbone(self):
        """Freeze ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("ViT backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("ViT backbone unfrozen")


def create_vit_model(config):
    """
    Create ViT model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'model' key
    
    Returns:
        ViTClassifier instance
    """
    model_config = config['model']
    
    model = ViTClassifier(
        model_name=model_config['name'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        input_channels=model_config['input_channels'],
        dropout=model_config['dropout'],
        head_hidden_dims=model_config.get('head_hidden_dims', [512, 256, 128]),
        head_num_attention_heads=model_config.get('head_num_attention_heads', 8),
        head_attention_dropout=model_config.get('head_attention_dropout', 0.1),
        head_use_residual=model_config.get('head_use_residual', True)
    )
    
    return model
