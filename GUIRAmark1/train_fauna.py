"""
Training script for fauna detection model.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fauna.fauna_detection import FaunaDetectionTrainer
from utils import setup_logging, load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train fauna detection model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    parser.add_argument('--model_type', choices=['detection', 'density', 'both'], 
                      default='both', help='Which model to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    fauna_config = config['fauna']
    
    # Override config with command line arguments
    if args.device != 'auto':
        fauna_config['device'] = args.device
    if args.epochs:
        fauna_config['epochs'] = args.epochs
    if args.batch_size:
        fauna_config['batch_size'] = args.batch_size
    if args.lr:
        fauna_config['lr'] = args.lr
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting fauna detection training")
    logger.info(f"Configuration: {fauna_config}")
    logger.info(f"Training model type: {args.model_type}")
    
    # Initialize trainer
    trainer = FaunaDetectionTrainer(fauna_config)
    
    # Prepare data
    trainer.prepare_data(fauna_config['data_dir'])
    
    # Train models
    epochs = fauna_config.get('epochs', 40)
    
    if args.model_type in ['detection', 'both']:
        logger.info("Training detection model...")
        trainer.train_detection(epochs=epochs, save_dir=args.save_dir)
    
    if args.model_type in ['density', 'both']:
        logger.info("Training density estimation model...")
        trainer.train_density(epochs=epochs, save_dir=args.save_dir)
    
    logger.info("Fauna detection training completed successfully")
    

if __name__ == "__main__":
    main()
