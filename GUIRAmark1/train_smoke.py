"""
Training script for smoke detection model.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from smoke.smoke_detection import SmokeDetectionTrainer
from utils import setup_logging, load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train smoke detection model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    smoke_config = config['smoke']
    
    # Override config with command line arguments
    if args.device != 'auto':
        smoke_config['device'] = args.device
    if args.epochs:
        smoke_config['epochs'] = args.epochs
    if args.batch_size:
        smoke_config['batch_size'] = args.batch_size
    if args.lr:
        smoke_config['lr'] = args.lr
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting smoke detection training")
    logger.info(f"Configuration: {smoke_config}")
    
    # Initialize trainer
    trainer = SmokeDetectionTrainer(smoke_config)
    
    # Prepare data
    trainer.prepare_data(smoke_config['data_dir'])
    
    # Train model
    trainer.train(
        epochs=smoke_config.get('epochs', 30),
        save_dir=args.save_dir
    )
    
    logger.info("Smoke detection training completed successfully")
    

if __name__ == "__main__":
    main()
