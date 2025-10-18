import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ViT_B_16_Weights
from transformers import ViTForImageClassification, ViTConfig, ViTModel
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


class ViTTripletBiRNNClassifier(nn.Module):
    """Processes triplets of CT slices with a shared ViT encoder and a bidirectional RNN."""

    def __init__(
        self,
        model_name='google/vit-base-patch16-224',
        num_classes=5,
        pretrained=True,
        input_channels=9,
        slice_channels=3,
        dropout=0.1,
        rnn_hidden_size=512,
        rnn_num_layers=1,
        rnn_dropout=0.0,
        sequence_pooling='last',
        backend='huggingface'
    ):
        """Initialise the hybrid ViT-RNN classifier.

        Args:
            model_name (str): ViT backbone identifier.
            num_classes (int): Number of output labels.
            pretrained (bool): Whether to load ImageNet-pretrained weights.
            input_channels (int): Total channel count (e.g. 9 for 3 slices × 3 channels).
            slice_channels (int): Number of channels per slice (defaults to windowed RGB triplets).
            dropout (float): Dropout applied before the classifier head.
            rnn_hidden_size (int): Hidden size of the bidirectional RNN.
            rnn_num_layers (int): Number of RNN layers.
            rnn_dropout (float): Dropout between stacked RNN layers (ignored if one layer).
            sequence_pooling (str): Aggregation strategy, supports 'last' or 'mean'.
            backend (str): Currently only 'huggingface' is supported.
        """
        super().__init__()

        if backend != 'huggingface':
            raise ValueError("ViTTripletBiRNNClassifier currently supports only the 'huggingface' backend")

        if input_channels % slice_channels != 0:
            raise ValueError("input_channels must be divisible by slice_channels so slices can be reconstructed")

        if sequence_pooling not in {'last', 'mean'}:
            raise ValueError("sequence_pooling must be either 'last' or 'mean'")

        self.num_classes = num_classes
        self.slice_channels = slice_channels
        self.num_slices = input_channels // slice_channels
        self.sequence_pooling = sequence_pooling

        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            vit_config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(vit_config)

        self.embed_dim = self.vit.config.hidden_size

        # Bidirectional RNN that fuses per-slice embeddings into a sequence representation
        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0
        )

        self.rnn_output_dim = rnn_hidden_size * 2  # bidirectional
        self.head = nn.Sequential(
            nn.LayerNorm(self.rnn_output_dim),
            nn.Dropout(dropout),
            nn.Linear(self.rnn_output_dim, num_classes)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Tensor shaped (B, num_slices * slice_channels, H, W).

        Returns:
            torch.Tensor: Logits shaped (B, num_classes).
        """
        b, c, h, w = x.shape

        if c != self.num_slices * self.slice_channels:
            raise ValueError(
                f"Expected {self.num_slices * self.slice_channels} channels (got {c}); "
                "check slice_channels or input preprocessing."
            )

        # Group channels back into (B, num_slices, slice_channels, H, W)
        slices = x.view(b, self.num_slices, self.slice_channels, h, w).contiguous()

        # Flatten slice dimension into the batch to reuse the ViT in a single call
        slices = slices.view(b * self.num_slices, self.slice_channels, h, w).contiguous()

        vit_outputs = self.vit(pixel_values=slices)
        if vit_outputs.pooler_output is not None:
            slice_embeddings = vit_outputs.pooler_output
        else:
            # Fallback to class token representation if pooler is disabled
            slice_embeddings = vit_outputs.last_hidden_state[:, 0]

        slice_embeddings = slice_embeddings.view(b, self.num_slices, self.embed_dim).contiguous()

        sequence_output, (h_n, _) = self.rnn(slice_embeddings)

        if self.sequence_pooling == 'mean':
            seq_repr = sequence_output.mean(dim=1)
        else:
            # Concatenate final hidden states from both directions
            forward_final = h_n[-2]
            backward_final = h_n[-1]
            seq_repr = torch.cat([forward_final, backward_final], dim=-1)

        logits = self.head(seq_repr)
        return logits

    def freeze_backbone(self):
        """Freeze the ViT encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the ViT encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True

    def get_slice_embeddings(self, x):
        """Return per-slice embeddings before sequence fusion (useful for debugging)."""
        self.eval()
        with torch.no_grad():
            b, c, h, w = x.shape
            slices = x.view(b, self.num_slices, self.slice_channels, h, w).contiguous()
            slices = slices.view(b * self.num_slices, self.slice_channels, h, w).contiguous()
            vit_outputs = self.vit(pixel_values=slices)
            if vit_outputs.pooler_output is not None:
                embeddings = vit_outputs.pooler_output
            else:
                embeddings = vit_outputs.last_hidden_state[:, 0]
            embeddings = embeddings.view(b, self.num_slices, self.embed_dim).contiguous()
        return embeddings

