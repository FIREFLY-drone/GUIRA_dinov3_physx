"""
Training script for vegetation health assessment model.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from vegetation.vegetation_health import VegetationHealthTrainer
from utils import setup_logging, load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train vegetation health model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    parser.add_argument('--backbone', choices=['resnet50', 'resnet34'], 
                      default='resnet50', help='Backbone architecture')
    parser.add_argument('--use_vari', action='store_true', default=True, 
                      help='Use VARI index as additional channel')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    veg_config = config['vegetation']
    
    # Override config with command line arguments
    if args.device != 'auto':
        veg_config['device'] = args.device
    if args.epochs:
        veg_config['epochs'] = args.epochs
    if args.batch_size:
        veg_config['batch_size'] = args.batch_size
    if args.lr:
        veg_config['lr'] = args.lr
    
    veg_config['vari_enabled'] = args.use_vari
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting vegetation health training")
    logger.info(f"Configuration: {veg_config}")
    logger.info(f"Using VARI: {args.use_vari}")
    logger.info(f"Backbone: {args.backbone}")
    
    # Initialize trainer
    trainer = VegetationHealthTrainer(veg_config)
    
    # Prepare data
    trainer.prepare_data(veg_config['data_dir'])
    
    # Train model
    trainer.train(
        epochs=veg_config.get('epochs', 35),
        save_dir=args.save_dir
    )
    
    logger.info("Vegetation health training completed successfully")
    

if __name__ == "__main__":
    main()
