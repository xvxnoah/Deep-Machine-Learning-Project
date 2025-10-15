import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights
from transformers import ViTForImageClassification, ViTConfig
from .channel_adapter import ChannelAdapter


class ViTClassifier(nn.Module):
    """Vision Transformer for Intracranial Hemorrhage Classification."""
    
    def __init__(
        self,
        model_name='google/vit-base-patch16-224',
        num_classes=5,
        pretrained=True,
        input_channels=9,
        channel_adaptation='conv1x1',
        dropout=0.1,
        backend='huggingface'
    ):
        """
        Args:
            model_name (str): Name of ViT model
            num_classes (int): Number of output classes (5 hemorrhage types)
            pretrained (bool): Use ImageNet pretrained weights
            input_channels (int): Number of input channels (9 for our case)
            channel_adaptation (str): Strategy for 9→3 channel adaptation
            dropout (float): Dropout rate for classification head
            backend (str): 'huggingface', 'torchvision', or 'timm'
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backend = backend
        
        # Channel adapter (9 channels → 3 channels)
        if input_channels != 3:
            self.channel_adapter = ChannelAdapter(
                in_channels=input_channels,
                out_channels=3,
                strategy=channel_adaptation
            )
        else:
            self.channel_adapter = nn.Identity()
        
        # Load pretrained ViT
        if backend == 'huggingface':
            if pretrained:
                self.vit = ViTForImageClassification.from_pretrained(
                    model_name, 
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True  # Allow different number of classes
                )
            else:
                config = ViTConfig.from_pretrained(model_name)
                config.num_labels = num_classes
                self.vit = ViTForImageClassification(config)
            
            self.embed_dim = self.vit.config.hidden_size
            self.use_builtin_head = True  # Flag to use built-in classifier
            
        elif backend == 'torchvision':
            # Use torchvision ViT
            if pretrained:
                weights = ViT_B_16_Weights.IMAGENET1K_V1
                self.vit = tv_models.vit_b_16(weights=weights)
            else:
                self.vit = tv_models.vit_b_16(weights=None)
            
            # Remove the classification head
            self.embed_dim = self.vit.heads.head.in_features
            self.vit.heads = nn.Identity()
            
        elif backend == 'timm':
            # Use timm ViT
            import timm
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
            )
            self.embed_dim = self.vit.num_features
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'huggingface', 'torchvision', or 'timm'")
        
        # Custom classification head for multi-label classification (only for non-huggingface backends)
        if backend != 'huggingface':
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes)
            )
        else:
            # HuggingFace has built-in classifier - no need for custom head
            self.classifier = None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # Adapt channels: (B, 9, H, W) → (B, 3, H, W)
        x = self.channel_adapter(x)
        
        # Ensure contiguous for MPS compatibility
        x = x.contiguous()
        
        # Get predictions based on backend
        if self.backend == 'huggingface':
            # HuggingFace ViTForImageClassification handles everything internally (MPS-safe!)
            # Just like in the TFG project: predictions = net(images).logits
            outputs = self.vit(pixel_values=x)
            logits = outputs.logits  # (B, num_classes) - already processed!
        else:
            # torchvision and timm: extract features then classify
            features = self.vit(x)  # (B, embed_dim)
            features = features.contiguous()
            logits = self.classifier(features)  # (B, num_classes)
        
        # Ensure output is contiguous
        logits = logits.contiguous()
        
        return logits
    
    def freeze_backbone(self):
        """Freeze ViT backbone parameters (keep classifier trainable)."""
        if self.backend == 'huggingface':
            # Freeze only the ViT encoder, keep classifier trainable
            if hasattr(self.vit, 'vit'):
                for param in self.vit.vit.parameters():
                    param.requires_grad = False
            # Channel adapter stays trainable
        else:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def get_attention_maps(self, x):
        """
        Extract attention maps from ViT for visualization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
            
        Returns:
            list: Attention maps from each transformer block
        """
        x = self.channel_adapter(x).contiguous()
        
        if self.backend == 'huggingface':
            # HuggingFace ViTForImageClassification also supports output_attentions
            outputs = self.vit(pixel_values=x, output_attentions=True)
            return outputs.attentions  # Tuple of attention tensors from each layer
        
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Extract attention weights
            if hasattr(output, 'attn'):
                attention_maps.append(output.attn.detach())
        
        # Register hooks on transformer blocks
        hooks = []
        if self.backend == 'torchvision':
            # torchvision uses encoder.layers
            for block in self.vit.encoder.layers:
                hooks.append(block.self_attention.register_forward_hook(hook_fn))
        elif self.backend == 'timm':
            # timm uses blocks
            for block in self.vit.blocks:
                hooks.append(block.attn.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = self.vit(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps

