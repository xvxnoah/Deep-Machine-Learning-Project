import torch
import torch.nn as nn


class ChannelAdapter(nn.Module):
    """Adapts input channels from 9 to 3 for ViT compatibility."""
    
    def __init__(self, in_channels=9, out_channels=3, strategy='conv1x1'):
        """
        Args:
            in_channels (int): Number of input channels (default: 9)
            out_channels (int): Number of output channels (default: 3)
            strategy (str): Adaptation strategy - 'conv1x1', 'linear_projection', or 'avg_pool'
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strategy = strategy
        
        if strategy == 'conv1x1':
            # Learnable 1×1 convolution to project channels
            self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'conv1x1'.")
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, 3, H, W)
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        if self.strategy == 'conv1x1':
            return self.adapter(x)
        
        elif self.strategy == 'linear_projection':
            # Rearrange: (B, C, H, W) -> (B, H, W, C) -> linear -> (B, H, W, C') -> (B, C', H, W)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x = self.adapter(x)  # (B, H, W, out_channels)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, out_channels, H, W)
            return x
        
        elif self.strategy == 'avg_pool':
            # Average pool over channel groups
            # Reshape (B, 9, H, W) -> (B, 3, 3, H, W) -> average -> (B, 3, H, W)
            B, C, H, W = x.shape
            x = x.contiguous().reshape(B, self.out_channels, self.group_size, H, W)
            x = x.mean(dim=2)  # Average over group dimension
            return x.contiguous()