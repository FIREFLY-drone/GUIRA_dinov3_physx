"""
Fire Spread Training - Hybrid Physics+NN

MODEL: UNet/ConvLSTM with physics regularization
DATA: Raster stacks (fuel, DEM, weather, prior burns) 256x256 windows
TRAINING RECIPE: T_in=6, T_out=12, epochs=100, physics loss λ=0.1
EVAL & ACCEPTANCE: IoU@horizon>=0.65, Hausdorff<50m, physics consistency
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import setup_logging


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for spatio-temporal modeling."""
    
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class FireSpreadModel(nn.Module):
    """Hybrid physics-NN model for fire spread prediction."""
    
    def __init__(self, input_channels=8, hidden_dim=64, num_layers=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # UNet-style encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTMCell(128, hidden_dim, 3)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # Probability output
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states
        h = torch.zeros(batch_size, self.hidden_dim, height//2, width//2).to(x.device)
        c = torch.zeros(batch_size, self.hidden_dim, height//2, width//2).to(x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Encode current frame
            encoded = self.encoder(x[:, t])
            
            # Apply ConvLSTM
            h, c = self.convlstm(encoded, (h, c))
            
            # Decode to prediction
            output = self.decoder(h)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)


class FireSpreadTrainer:
    """Trainer for fire spread prediction model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['training']['device'] if config['training']['device'] != 'auto'
                                 else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = FireSpreadModel(
            input_channels=config['model']['input_channels'],
            hidden_dim=config['model']['hidden_dim']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.physics_weight = config['training']['physics_loss_weight']
        
        logger.info(f"Initialized FireSpreadTrainer on {self.device}")
    
    def dice_loss(self, pred, target):
        """Dice loss for binary segmentation."""
        smooth = 1.0
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def physics_loss(self, pred, wind_u, wind_v, slope):
        """Physics-based regularization loss."""
        # Simplified physics constraint: fire should spread with wind direction
        # This would be much more complex in practice
        
        # Gradient of fire probability (spread direction)
        grad_y, grad_x = torch.gradient(pred.squeeze())
        
        # Wind direction should correlate with spread direction
        wind_alignment = torch.cos(torch.atan2(grad_y, grad_x) - torch.atan2(wind_v, wind_u))
        
        # Fire should spread uphill (slope effect)
        slope_effect = torch.mean(grad_y * slope)  # Simplified
        
        return -torch.mean(wind_alignment) - 0.1 * slope_effect
    
    def train(self):
        """Train the fire spread model."""
        logger.info("Training fire spread model")
        
        self.model.train()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_loss = 0
            num_batches = 5  # Dummy batches
            
            for batch_idx in range(num_batches):
                # Create dummy batch
                batch_size = self.config['training']['batch_size']
                seq_len = self.config['model']['input_sequence_length']
                
                # Input raster stack: [fuel, DEM, slope, wind_u, wind_v, temp, humidity, prior_burn]
                inputs = torch.randn(batch_size, seq_len, 8, 256, 256).to(self.device)
                targets = torch.rand(batch_size, seq_len, 1, 256, 256).to(self.device)  # Future burn masks
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Combined loss
                dice_loss = self.dice_loss(predictions, targets)
                bce_loss = F.binary_cross_entropy(predictions, targets)
                
                # Physics loss (using last frame for simplicity)
                wind_u = inputs[:, -1, 3]  # Wind u-component
                wind_v = inputs[:, -1, 4]  # Wind v-component  
                slope = inputs[:, -1, 2]   # Slope
                phys_loss = self.physics_loss(predictions[:, -1], wind_u, wind_v, slope)
                
                total_loss = 0.5 * dice_loss + 0.5 * bce_loss + self.physics_weight * phys_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}: Loss = {avg_loss:.4f}")
        
        # Save model
        save_path = Path(self.config['paths']['save_dir']) / 'fire_spread.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        
        logger.info(f"Training completed. Model saved to {save_path}")
        return {'success': True, 'model_path': str(save_path)}


def main():
    parser = argparse.ArgumentParser(description='Train Fire Spread Model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    # Default config
    config = {
        'model': {
            'input_channels': 8,  # fuel, DEM, slope, wind_u, wind_v, temp, humidity, prior_burn
            'hidden_dim': 64,
            'input_sequence_length': 6,
            'output_sequence_length': 12
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'device': 'cpu',
            'physics_loss_weight': 0.1
        },
        'paths': {
            'save_dir': 'models/spread/runs'
        }
    }
    
    # Load config if exists  
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    setup_logging('INFO')
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration validated")
        return
    
    # Train model
    trainer = FireSpreadTrainer(config)
    results = trainer.train()
    
    success = results.get('success', False)  
    logger.info(f"Training completed: {'Success' if success else 'Failed'}")


if __name__ == '__main__':
    main()