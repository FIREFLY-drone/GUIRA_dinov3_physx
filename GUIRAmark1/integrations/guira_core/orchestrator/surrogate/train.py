"""
Training Script for FireSpreadNet Surrogate Model

Trains encoder-decoder CNN to emulate PhysX fire spread simulations.
Uses MLflow for experiment tracking.

Usage:
    python train.py --data-dir ../physx_dataset --epochs 50 --batch-size 8 --exp-name physx-surrogate
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import logging

from models import FireSpreadNet, FireSpreadNetLite, combined_loss, brier_score
from dataset_builder import load_sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FireSpreadDataset(Dataset):
    """PyTorch Dataset for fire spread surrogate training."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load metadata
        metadata_file = self.data_dir / 'metadata' / f'{split}.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded {split} split with {len(self.metadata)} samples")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            input_stack: Input features, shape (C, H, W)
            target_ignition: Target ignition binary mask, shape (1, H, W)
            target_intensity: Target intensity, shape (1, H, W)
        """
        sample_meta = self.metadata[idx]
        sample_id = sample_meta['sample_id']
        sample_path = self.data_dir / 'samples' / f'{sample_id}.npz'
        
        input_stack, target_ignition, target_intensity = load_sample(str(sample_path))
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_stack).float()
        target_ignition_tensor = torch.from_numpy(target_ignition).float().unsqueeze(0)
        target_intensity_tensor = torch.from_numpy(target_intensity).float().unsqueeze(0)
        
        return input_tensor, target_ignition_tensor, target_intensity_tensor


class SurrogateTrainer:
    """Trainer for FireSpreadNet surrogate model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 lr: float = 1e-3,
                 bce_weight: float = 1.0,
                 mse_weight: float = 1.0,
                 brier_weight: float = 0.5):
        """
        Initialize trainer.
        
        Args:
            model: FireSpreadNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            bce_weight: Weight for BCE loss
            mse_weight: Weight for MSE loss
            brier_weight: Weight for Brier score
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.brier_weight = brier_weight
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_bce = 0.0
        total_mse = 0.0
        total_brier = 0.0
        n_batches = 0
        
        for inputs, target_ignition, target_intensity in tqdm(self.train_loader, desc='Training'):
            inputs = inputs.to(self.device)
            target_ignition = target_ignition.to(self.device)
            target_intensity = target_intensity.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_ignition, pred_intensity = self.model(inputs)
            
            # Calculate loss
            loss, loss_dict = combined_loss(
                pred_ignition, pred_intensity,
                target_ignition, target_intensity,
                self.bce_weight, self.mse_weight, self.brier_weight
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total']
            total_bce += loss_dict['bce']
            total_mse += loss_dict['mse']
            total_brier += loss_dict['brier']
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'bce': total_bce / n_batches,
            'mse': total_mse / n_batches,
            'brier': total_brier / n_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        total_bce = 0.0
        total_mse = 0.0
        total_brier = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for inputs, target_ignition, target_intensity in tqdm(self.val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                target_ignition = target_ignition.to(self.device)
                target_intensity = target_intensity.to(self.device)
                
                # Forward pass
                pred_ignition, pred_intensity = self.model(inputs)
                
                # Calculate loss
                loss, loss_dict = combined_loss(
                    pred_ignition, pred_intensity,
                    target_ignition, target_intensity,
                    self.bce_weight, self.mse_weight, self.brier_weight
                )
                
                # Accumulate metrics
                total_loss += loss_dict['total']
                total_bce += loss_dict['bce']
                total_mse += loss_dict['mse']
                total_brier += loss_dict['brier']
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'bce': total_bce / n_batches,
            'mse': total_mse / n_batches,
            'brier': total_brier / n_batches
        }
    
    def train(self, epochs: int, save_dir: str) -> None:
        """
        Train model for specified epochs.
        
        Args:
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"BCE: {train_metrics['bce']:.4f}, "
                       f"MSE: {train_metrics['mse']:.4f}, "
                       f"Brier: {train_metrics['brier']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"BCE: {val_metrics['bce']:.4f}, "
                       f"MSE: {val_metrics['mse']:.4f}, "
                       f"Brier: {val_metrics['brier']:.4f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_bce': train_metrics['bce'],
                'train_mse': train_metrics['mse'],
                'train_brier': train_metrics['brier'],
                'val_loss': val_metrics['loss'],
                'val_bce': val_metrics['bce'],
                'val_mse': val_metrics['mse'],
                'val_brier': val_metrics['brier'],
                'lr': self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                best_path = save_path / 'fire_spreadnet.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }, best_path)
                logger.info(f"✓ Saved best model to {best_path}")
                
                # Log model to MLflow
                mlflow.pytorch.log_model(self.model, "model")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss']
                }, checkpoint_path)
        
        logger.info(f"\n✓ Training complete. Best val loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train FireSpreadNet surrogate model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--exp-name', type=str, default='physx-surrogate',
                       help='MLflow experiment name')
    parser.add_argument('--model-type', type=str, default='full',
                       choices=['full', 'lite'],
                       help='Model architecture (full or lite)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda, cpu, or auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load dataset info
    data_dir = Path(args.data_dir)
    info_file = data_dir / 'dataset_info.json'
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create datasets
    train_dataset = FireSpreadDataset(args.data_dir, split='train')
    val_dataset = FireSpreadDataset(args.data_dir, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    in_channels = dataset_info['input_channels']
    if args.model_type == 'full':
        model = FireSpreadNet(in_channels=in_channels)
    else:
        model = FireSpreadNetLite(in_channels=in_channels)
    
    logger.info(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup MLflow
    mlflow.set_experiment(args.exp_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': str(device),
            'in_channels': in_channels,
            'grid_size': str(dataset_info['grid_size']),
            'train_samples': dataset_info['splits']['train'],
            'val_samples': dataset_info['splits']['val']
        })
        
        # Create trainer
        trainer = SurrogateTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr
        )
        
        # Train
        trainer.train(epochs=args.epochs, save_dir=args.save_dir)
        
        # Log final best metrics
        mlflow.log_metric('best_val_loss', trainer.best_val_loss)
        
        logger.info("✓ Training completed successfully")


if __name__ == '__main__':
    main()
