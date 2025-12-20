"""
Training script for fire spread simulation model.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from spread.fire_spread_simulation import FireSpreadTrainer
from utils import setup_logging, load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train fire spread simulation model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    parser.add_argument('--sequence_length', type=int, default=5, help='Training sequence length')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    spread_config = config['spread']
    
    # Override config with command line arguments
    if args.device != 'auto':
        spread_config['device'] = args.device
    if args.epochs:
        spread_config['epochs'] = args.epochs
    if args.batch_size:
        spread_config['batch_size'] = args.batch_size
    if args.lr:
        spread_config['lr'] = args.lr
    
    spread_config['sequence_length'] = args.sequence_length
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting fire spread simulation training")
    logger.info(f"Configuration: {spread_config}")
    logger.info(f"Sequence length: {args.sequence_length}")
    
    # Initialize trainer
    trainer = FireSpreadTrainer(spread_config)
    
    # Prepare data
    trainer.prepare_data(spread_config['data_dir'])
    
    # Train model
    trainer.train(
        epochs=spread_config.get('epochs', 50),
        save_dir=args.save_dir
    )
    
    logger.info("Fire spread simulation training completed successfully")
    

if __name__ == "__main__":
    main()
