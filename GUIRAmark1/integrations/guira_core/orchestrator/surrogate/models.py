"""
FireSpreadNet - Surrogate Model for PhysX Fire Spread Simulation

MODEL: Encoder-decoder CNN (U-Net) for fast fire spread emulation
DATA: Raster stacks from PhysX simulations
TRAINING: See train.py and README.md
EVAL: MSE (intensity), BCE (ignition), Brier score (probabilistic calibration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FireSpreadNet(nn.Module):
    """
    Encoder-decoder CNN for learning fire spread dynamics from PhysX simulations.
    
    Architecture: U-Net style with skip connections
    Input: Raster stack [fire_t0, wind_u, wind_v, humidity, fuel_density, slope] shape (C, H, W)
    Output: 
        - ignition_prob: Ignition probability map at t1 (1, H, W)
        - intensity: Fire intensity map at t1 (1, H, W)
    """
    
    def __init__(self, 
                 in_channels: int = 6,
                 base_filters: int = 32,
                 num_levels: int = 4):
        """
        Initialize FireSpreadNet.
        
        Args:
            in_channels: Number of input channels (default: 6 for fire, wind_u, wind_v, humidity, fuel, slope)
            base_filters: Base number of filters in first layer (doubles at each level)
            num_levels: Number of encoder/decoder levels
        """
        super(FireSpreadNet, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.num_levels = num_levels
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        current_channels = in_channels
        for i in range(num_levels):
            out_channels = base_filters * (2 ** i)
            self.encoder_blocks.append(self._conv_block(current_channels, out_channels))
            current_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = base_filters * (2 ** num_levels)
        self.bottleneck = self._conv_block(current_channels, bottleneck_channels)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        current_channels = bottleneck_channels
        for i in range(num_levels - 1, -1, -1):
            out_channels = base_filters * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(current_channels, out_channels, 2, stride=2))
            # *2 for concatenation with skip connection
            self.decoder_blocks.append(self._conv_block(out_channels * 2, out_channels))
            current_channels = out_channels
        
        # Output heads
        self.ignition_head = nn.Conv2d(base_filters, 1, 1)
        self.intensity_head = nn.Conv2d(base_filters, 1, 1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            ignition_prob: Ignition probability map (B, 1, H, W)
            intensity: Fire intensity map (B, 1, H, W)
        """
        # Encoder with skip connections
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            # Concatenate with skip connection
            skip = skip_connections[i]
            # Handle size mismatch due to padding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Output heads
        ignition_prob = torch.sigmoid(self.ignition_head(x))
        intensity = torch.relu(self.intensity_head(x))
        
        return ignition_prob, intensity


class FireSpreadNetLite(nn.Module):
    """
    Lightweight version of FireSpreadNet for faster inference.
    
    Uses fewer filters and levels while maintaining reasonable accuracy.
    """
    
    def __init__(self, in_channels: int = 6):
        """Initialize lightweight FireSpreadNet."""
        super(FireSpreadNetLite, self).__init__()
        
        # Simpler encoder
        self.enc1 = self._conv_block(in_channels, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)
        
        # Output heads
        self.ignition_head = nn.Conv2d(16, 1, 1)
        self.intensity_head = nn.Conv2d(16, 1, 1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a lightweight convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Encoder
        e1 = self.enc1(x)
        x = self.pool(e1)
        
        e2 = self.enc2(x)
        x = self.pool(e2)
        
        e3 = self.enc3(x)
        x = self.pool(e3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up3(x)
        if x.shape != e3.shape:
            x = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        if x.shape != e2.shape:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        if x.shape != e1.shape:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Output
        ignition_prob = torch.sigmoid(self.ignition_head(x))
        intensity = torch.relu(self.intensity_head(x))
        
        return ignition_prob, intensity


def brier_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Brier score for probabilistic calibration.
    
    Args:
        predictions: Predicted probabilities (0-1)
        targets: Binary targets (0 or 1)
    
    Returns:
        Brier score (lower is better)
    """
    return torch.mean((predictions - targets) ** 2)


def combined_loss(
    pred_ignition: torch.Tensor,
    pred_intensity: torch.Tensor,
    target_ignition: torch.Tensor,
    target_intensity: torch.Tensor,
    bce_weight: float = 1.0,
    mse_weight: float = 1.0,
    brier_weight: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss for fire spread prediction.
    
    Args:
        pred_ignition: Predicted ignition probabilities
        pred_intensity: Predicted fire intensity
        target_ignition: Target ignition binary mask
        target_intensity: Target fire intensity
        bce_weight: Weight for BCE loss
        mse_weight: Weight for MSE loss
        brier_weight: Weight for Brier score
    
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary with individual loss components
    """
    # Binary cross-entropy for ignition
    bce_loss = F.binary_cross_entropy(pred_ignition, target_ignition)
    
    # MSE for intensity
    mse_loss = F.mse_loss(pred_intensity, target_intensity)
    
    # Brier score for probabilistic calibration
    brier = brier_score(pred_ignition, target_ignition)
    
    # Combined loss
    total_loss = (bce_weight * bce_loss + 
                  mse_weight * mse_loss + 
                  brier_weight * brier)
    
    loss_dict = {
        'bce': bce_loss.item(),
        'mse': mse_loss.item(),
        'brier': brier.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict
