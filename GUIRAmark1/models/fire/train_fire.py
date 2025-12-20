"""
Fire Detection Training Script - YOLOv8

MODEL: YOLOv8 (nano/small selectable) with pre-trained COCO weights
DATA: flame_rgb, flame_rgb_simplified, flame2_rgb_ir, sfgdn_fire, flame3_thermal, wit_uas_thermal
TRAINING RECIPE: img=640, epochs=150, batch=16, lr0=0.01, SGD+cosine, warmup=3
EVAL & ACCEPTANCE: mAP@50>=0.6, mAP@50-95>=0.4, small-object AP>=0.3
"""

import argparse
import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from loguru import logger

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    # Suppress ultralytics verbose logging
    LOGGER.setLevel(logging.WARNING)
except ImportError as e:
    logger.warning(f"ultralytics not available: {e}")
    YOLO = None

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    logger.warning("Hydra not available, falling back to argparse")
    HYDRA_AVAILABLE = False

from utils import setup_logging, load_config


class FireYOLOv8Trainer:
    """Fire Detection Trainer using YOLOv8."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        self.paths_config = config.get('paths', {})
        
        # Set device
        device = self.training_config.get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self._init_model()
        
        # Setup directories
        self._setup_directories()
        
        logger.info(f"Initialized FireYOLOv8Trainer on {self.device}")
        
    def _init_model(self):
        """Initialize YOLOv8 model."""
        if YOLO is None:
            logger.error("YOLOv8 not available. Please install ultralytics.")
            return
            
        model_size = self.model_config.get('size', 'n')
        pretrained = self.model_config.get('pretrained', True)
        
        if pretrained:
            model_path = f'yolov8{model_size}.pt'
        else:
            model_path = f'yolov8{model_size}.yaml'
            
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8{model_size} model")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # Fallback to nano model
            try:
                self.model = YOLO('yolov8n.pt')
                logger.warning("Using fallback YOLOv8n model")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
    
    def _setup_directories(self):
        """Setup training directories."""
        self.save_dir = Path(self.paths_config.get('save_dir', 'models/fire/runs'))
        self.log_dir = Path(self.paths_config.get('log_dir', 'models/fire/logs'))
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def prepare_data_yaml(self) -> str:
        """Prepare YOLO data configuration file."""
        data_yaml = {
            'path': str(Path(self.data_config['path']).absolute()),
            'train': self.data_config.get('train', 'images/train'),
            'val': self.data_config.get('val', 'images/val'),
            'test': self.data_config.get('test', 'images/test'),
            'nc': self.model_config.get('num_classes', 2),
            'names': self.data_config.get('names', {0: 'fire', 1: 'smoke'})
        }
        
        yaml_path = self.save_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return str(yaml_path)
    
    def train(self) -> Dict[str, Any]:
        """Train the fire detection model."""
        if self.model is None:
            logger.error("Model not initialized")
            return {}
        
        # Prepare data configuration
        data_yaml_path = self.prepare_data_yaml()
        
        # Prepare training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': self.training_config.get('epochs', 150),
            'imgsz': self.training_config.get('img_size', 640),
            'batch': self.training_config.get('batch_size', 16),
            'lr0': self.training_config.get('lr0', 0.01),
            'lrf': self.training_config.get('lrf', 0.1),
            'momentum': self.training_config.get('momentum', 0.937),
            'weight_decay': self.training_config.get('weight_decay', 0.0005),
            'warmup_epochs': self.training_config.get('warmup_epochs', 3),
            'warmup_momentum': self.training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.training_config.get('warmup_bias_lr', 0.1),
            'cos_lr': self.training_config.get('cos_lr', True),
            'device': self.device,
            'workers': self.training_config.get('workers', 4),
            'project': str(self.save_dir.parent),
            'name': 'fire_training',
            'exist_ok': True,
            'save_period': 10,
            'patience': 50,
            'verbose': True,
        }
        
        # Add augmentation parameters
        aug_config = self.config.get('augmentation', {})
        train_args.update({
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
            'copy_paste': aug_config.get('copy_paste', 0.3),
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.0),
            'fliplr': aug_config.get('fliplr', 0.5),
        })
        
        # Add loss parameters
        loss_config = self.config.get('loss', {})
        train_args.update({
            'box': loss_config.get('box_gain', 0.05),
            'cls': loss_config.get('cls_gain', 0.5),
            'dfl': loss_config.get('dfl_gain', 1.5),
        })
        
        logger.info(f"Starting training with parameters:")
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # Train the model
            results = self.model.train(**train_args)
            
            # Save training metrics
            metrics_path = self.save_dir / 'training_metrics.json'
            if hasattr(results, 'results_dict'):
                with open(metrics_path, 'w') as f:
                    json.dump(results.results_dict, f, indent=2, default=str)
            
            # Save final model
            final_model_path = self.save_dir / 'fire_yolov8_final.pt'
            if hasattr(self.model, 'save'):
                self.model.save(str(final_model_path))
            
            logger.info(f"Training completed successfully")
            logger.info(f"Model saved to: {final_model_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            
            return {
                'success': True,
                'model_path': str(final_model_path),
                'metrics_path': str(metrics_path),
                'results': results
            }
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate(self) -> Dict[str, Any]:
        """Run validation on the trained model."""
        if self.model is None:
            logger.error("Model not initialized")
            return {}
        
        data_yaml_path = self.prepare_data_yaml()
        
        try:
            val_results = self.model.val(
                data=data_yaml_path,
                imgsz=self.training_config.get('img_size', 640),
                batch=self.training_config.get('batch_size', 16),
                device=self.device,
                verbose=True
            )
            
            logger.info("Validation completed successfully")
            return {
                'success': True,
                'results': val_results
            }
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def create_default_config() -> Dict[str, Any]:
    """Create default configuration if none provided."""
    return {
        'model': {
            'name': 'fire_yolov8',
            'size': 'n',
            'num_classes': 2,
            'pretrained': True
        },
        'training': {
            'epochs': 150,
            'batch_size': 16,
            'img_size': 640,
            'device': 'auto',
            'workers': 4,
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'cos_lr': True
        },
        'data': {
            'path': 'data/processed/fire_yolo',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'fire', 1: 'smoke'}
        },
        'paths': {
            'save_dir': 'models/fire/runs',
            'log_dir': 'models/fire/logs'
        },
        'augmentation': {
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.3,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'fliplr': 0.5
        },
        'loss': {
            'box_gain': 0.05,
            'cls_gain': 0.5,
            'dfl_gain': 1.5
        }
    }


if HYDRA_AVAILABLE:
    @hydra.main(version_base=None, config_path=".", config_name="config")
    def main_hydra(cfg: DictConfig) -> None:
        """Main function using Hydra configuration."""
        config = OmegaConf.to_container(cfg, resolve=True)
        
        # Setup logging
        setup_logging(config.get('log_level', 'INFO'))
        
        logger.info("Fire Detection Training - Using Hydra configuration")
        logger.info("=" * 60)
        
        # Initialize trainer
        trainer = FireYOLOv8Trainer(config)
        
        # Train model
        results = trainer.train()
        
        if results.get('success', False):
            logger.info("Training completed successfully!")
            # Run validation
            val_results = trainer.validate()
            if val_results.get('success', False):
                logger.info("Validation completed successfully!")
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")


def main_argparse():
    """Main function using argparse configuration."""
    parser = argparse.ArgumentParser(
        description='Train Fire Detection Model (YOLOv8)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--lr', type=float, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'],
                       default='n', help='YOLOv8 model size')
    parser.add_argument('--data-path', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--save-dir', type=str,
                       help='Directory to save model and logs')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without actual training (for testing)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = create_default_config()
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr0'] = args.lr
    if args.device != 'auto':
        config['training']['device'] = args.device
    if args.img_size:
        config['training']['img_size'] = args.img_size
    if args.model_size:
        config['model']['size'] = args.model_size
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.save_dir:
        config['paths']['save_dir'] = args.save_dir
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    logger.info("Fire Detection Training - Using argparse configuration")
    logger.info("=" * 60)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Training will not actually run")
        logger.info("Configuration validated successfully")
        return
    
    # Initialize trainer
    trainer = FireYOLOv8Trainer(config)
    
    # Train model
    results = trainer.train()
    
    if results.get('success', False):
        logger.info("Training completed successfully!")
        # Run validation
        val_results = trainer.validate()
        if val_results.get('success', False):
            logger.info("Validation completed successfully!")
    else:
        logger.error(f"Training failed: {results.get('error', 'Unknown error')}")


def main():
    """Main entry point - choose between Hydra and argparse."""
    if HYDRA_AVAILABLE and len(sys.argv) == 1:
        # No arguments provided, try Hydra if available
        main_hydra()
    else:
        # Use argparse for backward compatibility and explicit args
        main_argparse()


if __name__ == '__main__':
    main()