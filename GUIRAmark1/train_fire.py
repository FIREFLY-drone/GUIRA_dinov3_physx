"""
Training script for fire detection model.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fire.fire_detection import FireDetectionTrainer
from utils import setup_logging, load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    fire_config = config['fire']
    
    # Override config with command line arguments
    if args.device != 'auto':
        fire_config['device'] = args.device
    if args.epochs:
        fire_config['epochs'] = args.epochs
    if args.batch_size:
        fire_config['batch_size'] = args.batch_size
    if args.lr:
        fire_config['lr'] = args.lr
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting fire detection training")
    logger.info(f"Configuration: {fire_config}")
    
    # Initialize trainer
    trainer = FireDetectionTrainer(fire_config)
    
    # Prepare data
    trainer.prepare_data(fire_config['data_dir'])
    
    # Train model
    results = trainer.train(
        epochs=fire_config.get('epochs', 50),
        save_dir=args.save_dir
    )
    
    logger.info("Fire detection training completed successfully")
    

if __name__ == "__main__":
    main()
