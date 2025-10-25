import os
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
        backend='huggingface',
        slice_channels=3,  # For sequence processing
        unfreeze_layers=0  # Number of transformer layers to unfreeze (from end)
    ):
        """
        Args:
            model_name (str): Name of ViT model
            num_classes (int): Number of output classes (5 hemorrhage types)
            pretrained (bool): Use ImageNet pretrained weights
            input_channels (int): Number of input channels (9 for our case)
            channel_adaptation (str): Strategy for 9→3 channel adaptation
            dropout (float): Dropout rate for classification head
            backend (str): 'huggingface' or 'torchvision'
            slice_channels (int): Channels per slice for sequence processing (3)
            unfreeze_layers (int): Number of transformer layers to unfreeze from end (0 = freeze all, -1 = unfreeze all)
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backend = backend
        self.slice_channels = slice_channels
        self.unfreeze_layers = unfreeze_layers

        self.process_sequences = (input_channels > 3 and input_channels % slice_channels == 0)
        if self.process_sequences:
            self.num_slices = input_channels // slice_channels
        else:
            self.num_slices = 1

        # Channel adapter - adapt per slice if processing sequences, otherwise adapt all channels
        if self.process_sequences:
            # For sequences: adapt each slice from slice_channels to 3
            # (ViT expects 3 channels, slice_channels might be 3 already)
            if slice_channels != 3:
                self.channel_adapter = ChannelAdapter(
                    in_channels=slice_channels,
                    out_channels=3,
                    strategy=channel_adaptation
                )
            else:
                self.channel_adapter = nn.Identity()
        else:
            # Single image: adapt all input_channels to 3
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
                self.vit = ViTModel.from_pretrained(model_name)
            else:
                config = ViTConfig.from_pretrained(model_name)
                self.vit = ViTModel(config)

            self.embed_dim = self.vit.config.hidden_size

            # Custom multi-label classification head
            # Input dimension depends on whether we process sequences
            classifier_input_dim = self.embed_dim * self.num_slices if self.process_sequences else self.embed_dim

            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Dropout(dropout),
                nn.Linear(classifier_input_dim, num_classes)
            )

            self.use_builtin_head = False  # Use our custom head
            
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
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'huggingface' or 'torchvision'")
        
        # Custom classification head for multi-label classification
        # Input dimension depends on whether we process sequences
        classifier_input_dim = self.embed_dim * self.num_slices if self.process_sequences else self.embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        b, c, h, w = x.shape

        if self.process_sequences:
            # Process as sequences (like RNN version but with ViT)
            if c != self.num_slices * self.slice_channels:
                raise ValueError(
                    f"Expected {self.num_slices * self.slice_channels} channels for "
                    f"{self.num_slices} slices (got {c})"
                )

            # Split into slices: (B, num_slices, slice_channels, H, W)
            slices = x.view(b, self.num_slices, self.slice_channels, h, w).contiguous()

            # Process each slice through channel adapter
            slice_features = []
            for i in range(self.num_slices):
                slice_i = slices[:, i]  # (B, slice_channels, H, W)
                slice_i = self.channel_adapter(slice_i)  # (B, 3, H, W)
                slice_i = slice_i.contiguous()

                # Process through ViT
                if self.backend == 'huggingface':
                    outputs = self.vit(pixel_values=slice_i)
                    if outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        features = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    features = self.vit(slice_i)  # (B, embed_dim)

                slice_features.append(features)

            # Concatenate all slice features: (B, num_slices * embed_dim)
            all_features = torch.cat(slice_features, dim=-1)
            all_features = all_features.contiguous()

            # Final classification
            logits = self.classifier(all_features)  # (B, num_classes)

        else:
            # Single image processing (original behavior)
            x = self.channel_adapter(x)
            x = x.contiguous()

            # Get predictions based on backend
            if self.backend == 'huggingface':
                # Use ViTModel to extract features, then our custom classifier
                outputs = self.vit(pixel_values=x)
                # Use pooler_output if available, otherwise use CLS token
                if outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0]  # CLS token
                features = features.contiguous()
                logits = self.classifier(features)  # (B, num_classes)
            else:
                # torchvision extract features then classify
                features = self.vit(x)  # (B, embed_dim)
                features = features.contiguous()
                logits = self.classifier(features)  # (B, num_classes)

        # Ensure output is contiguous
        logits = logits.contiguous()

        return logits
    
    def freeze_backbone(self):
        """Freeze ViT backbone parameters (keep classifier trainable), with optional partial unfreezing."""
        # First freeze everything in the ViT backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the last N transformer layers
        if self.unfreeze_layers != 0:  # 0 = freeze all, -1 = unfreeze all, >0 = unfreeze N layers
            if self.unfreeze_layers == -1:
                print(f"Unfreezing all transformer layers (full fine-tuning)...")
            else:
                print(f"Unfreezing {self.unfreeze_layers} layers...")

            if self.backend == 'huggingface':
                # For HuggingFace: unfreeze last N layers in the encoder
                if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layer'):
                    layers = self.vit.encoder.layer
                    num_layers = len(layers)

                    if self.unfreeze_layers == -1:
                        # Unfreeze all layers
                        start_unfreeze = 0
                        layers_to_unfreeze = num_layers
                    else:
                        # Unfreeze last N layers
                        start_unfreeze = max(0, num_layers - self.unfreeze_layers)
                        layers_to_unfreeze = self.unfreeze_layers

                    print(f"Unfreezing {layers_to_unfreeze} transformer layers "
                          f"({start_unfreeze} to {num_layers-1})")

                    unfrozen_count = 0
                    for i in range(start_unfreeze, num_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = True
                            unfrozen_count += param.numel()
                    print(f" Unfrozen {unfrozen_count:,} parameters in HuggingFace layers")

            elif self.backend == 'torchvision':
                # For torchvision: unfreeze last N layers in encoder.layers
                print(f"Checking torchvision structure...")
                print(f"    Has encoder: {hasattr(self.vit, 'encoder')}")
                if hasattr(self.vit, 'encoder'):
                    print(f"    Encoder has layers: {hasattr(self.vit.encoder, 'layers')}")

                if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layers'):
                    layers = self.vit.encoder.layers
                    num_layers = len(layers)

                    if self.unfreeze_layers == -1:
                        # Unfreeze all layers
                        start_unfreeze = 0
                        layers_to_unfreeze = num_layers
                    else:
                        # Unfreeze last N layers
                        start_unfreeze = max(0, num_layers - self.unfreeze_layers)
                        layers_to_unfreeze = min(self.unfreeze_layers, num_layers)

                    print(f"Unfreezing {layers_to_unfreeze} transformer layers "
                          f"({start_unfreeze} to {num_layers-1}) out of {num_layers} total")

                    unfrozen_count = 0
                    for i in range(start_unfreeze, num_layers):
                        layer_params = 0
                        for param in layers[i].parameters():
                            param.requires_grad = True
                            layer_params += param.numel()
                        print(f"    Layer {i}: unfroze {layer_params:,} parameters")
                        unfrozen_count += layer_params
                    print(f"  Total unfrozen in torchvision layers: {unfrozen_count:,} parameters")
                else:
                    print(f"    Could not find encoder.layers in torchvision model")
                    print(f"    Available attributes: {dir(self.vit)}")

        # Count final trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Final parameter counts: {trainable_params:,} trainable / {total_params:,} total")

        # Channel adapter always stays trainable
        if hasattr(self, 'channel_adapter'):
            for param in self.channel_adapter.parameters():
                param.requires_grad = True
    
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
        model_name='pre-trained-model',
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

        self.num_classes = num_classes
        self.slice_channels = slice_channels
        self.num_slices = input_channels // slice_channels
        self.sequence_pooling = sequence_pooling

        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name, local_files_only=True, use_safetensors=True)
        else:
            vit_config = ViTConfig.from_pretrained(model_name, local_files_only=True, use_safetensors=False)
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


class EnsembleViTClassifier(nn.Module):
    """Ensemble model combining a full ViT classifier and a ViT-RNN classifier."""

    def __init__(
        self,
        full_vit_path: str,
        rnn_vit_path: str,
        ensemble_method: str = 'average',
        weights: list = None,
        num_classes: int = 5,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble model.

        Args:
            full_vit_path (str): Path to the full ViT model checkpoint
            rnn_vit_path (str): Path to the ViT-RNN model checkpoint
            ensemble_method (str): How to combine predictions ('average', 'weighted', 'max_confidence')
            weights (list): Weights for weighted averaging [full_vit_weight, rnn_vit_weight]
            num_classes (int): Number of output classes
            device (str): Device to load models on
        """
        super().__init__()

        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        self.device = device

        # Set default weights if not provided
        if weights is None:
            self.weights = [0.5, 0.5]  # Equal weighting by default
        else:
            self.weights = weights

        # Load full ViT model
        print(f"Loading full ViT model from: {full_vit_path}")
        self.full_vit_model = self._load_model_checkpoint(full_vit_path, ViTClassifier)
        self.full_vit_model.to(device)
        self.full_vit_model.eval()

        # Load ViT-RNN model
        print(f"Loading ViT-RNN model from: {rnn_vit_path}")
        self.rnn_vit_model = self._load_model_checkpoint(rnn_vit_path, ViTTripletBiRNNClassifier)
        self.rnn_vit_model.to(device)
        self.rnn_vit_model.eval()

        print(f"Ensemble initialized with {ensemble_method} method")
        print(f"  Weights: {self.weights}")

    def _load_model_checkpoint(self, model_path: str, model_class):
        """Load a model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"  Checkpoint loaded successfully")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
            raise

        # Extract config from checkpoint
        config = checkpoint.get('config', {})
        if not config:
            print("  Warning: No config found in checkpoint, using default config")
            config = {
                'model': {
                    'name': 'google/vit-base-patch16-224',
                    'num_classes': 5,
                    'pretrained': True,
                    'input_channels': 9,
                    'slice_channels': 3,
                    'dropout': 0.1,
                    'backend': 'torchvision',
                    'unfreeze_layers': 0
                }
            }

        # Recreate model with same parameters
        if model_class == ViTTripletBiRNNClassifier:
            # Override model_name to avoid relative path issues in stored config
            model_name_override = 'pre-trained-model'
            print(f"  Overriding model_name to: {model_name_override}")

            # For loading from checkpoin
            model = ViTTripletBiRNNClassifier(
                model_name=model_name_override,  # Use override instead of config['model']['name']
                num_classes=config['model']['num_classes'],
                pretrained=False,
                input_channels=config['model']['input_channels'],
                slice_channels=config['model'].get('slice_channels', 3),
                dropout=config['model']['dropout'],
                rnn_hidden_size=config['model'].get('rnn_hidden_size', 512),
                rnn_num_layers=config['model'].get('rnn_num_layers', 1),
                rnn_dropout=config['model'].get('rnn_dropout', 0.0),
                sequence_pooling=config['model'].get('sequence_pooling', 'last'),
                backend=config['model']['backend']
            )
        else:  # ViTClassifier
            model = ViTClassifier(
                model_name=config['model']['name'],
                num_classes=config['model']['num_classes'],
                pretrained=config['model']['pretrained'],
                input_channels=config['model']['input_channels'],
                slice_channels=config['model'].get('slice_channels', 3),
                dropout=config['model']['dropout'],
                backend=config['model']['backend'],
                unfreeze_layers=config['model'].get('unfreeze_layers', 0)
            )

        # Load state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"  Model state dict loaded successfully (strict=False)")
        except Exception as e:
            print(f"  Failed to load model state dict: {e}")
            raise

        return model

    def forward(self, x):
        """
        Forward pass through ensemble.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Ensemble predictions
        """
        # Get predictions from both models
        with torch.no_grad():
            full_vit_logits = self.full_vit_model(x)
            rnn_vit_logits = self.rnn_vit_model(x)

        # Apply ensemble method
        if self.ensemble_method == 'average':
            # Simple averaging of logits
            ensemble_logits = (full_vit_logits + rnn_vit_logits) / 2

        elif self.ensemble_method == 'weighted':
            # Weighted averaging of logits
            w1, w2 = self.weights
            ensemble_logits = w1 * full_vit_logits + w2 * rnn_vit_logits

        elif self.ensemble_method == 'max_confidence':
            # Use predictions from model with higher confidence
            full_vit_probs = torch.sigmoid(full_vit_logits)
            rnn_vit_probs = torch.sigmoid(rnn_vit_logits)

            # Calculate confidence as mean probability across classes
            full_confidence = full_vit_probs.mean(dim=1, keepdim=True)
            rnn_confidence = rnn_vit_probs.mean(dim=1, keepdim=True)

            # Select logits based on higher confidence
            ensemble_logits = torch.where(
                full_confidence > rnn_confidence,
                full_vit_logits,
                rnn_vit_logits
            )

        elif self.ensemble_method == 'voting':
            # Convert to binary predictions and vote
            full_vit_preds = (torch.sigmoid(full_vit_logits) > 0.5).float()
            rnn_vit_preds = (torch.sigmoid(rnn_vit_logits) > 0.5).float()

            # Majority voting (average predictions)
            ensemble_preds = (full_vit_preds + rnn_vit_preds) / 2
            # Convert back to logits (rough approximation)
            ensemble_logits = torch.log(ensemble_preds / (1 - ensemble_preds + 1e-8))

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        return ensemble_logits

    def to(self, device):
        """Move ensemble to device."""
        self.device = device
        self.full_vit_model.to(device)
        self.rnn_vit_model.to(device)
        return self

    def eval(self):
        """Set ensemble to evaluation mode."""
        self.full_vit_model.eval()
        self.rnn_vit_model.eval()
        return self

    def train(self, mode=True):
        """Set ensemble to training mode (not typically used for inference)."""
        self.full_vit_model.train(mode)
        self.rnn_vit_model.train(mode)
        return self

