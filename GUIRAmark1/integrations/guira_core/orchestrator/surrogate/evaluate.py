"""
Evaluation Script for FireSpreadNet Surrogate Model

Evaluates trained model on test set and generates metrics/visualizations.

Usage:
    python evaluate.py --model-path models/fire_spreadnet.pt --data-dir physx_dataset --split test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from models import FireSpreadNet, FireSpreadNetLite, brier_score
from train import FireSpreadDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurrogateEvaluator:
    """Evaluator for FireSpreadNet surrogate model."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained FireSpreadNet model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        total_mse = 0.0
        total_bce = 0.0
        total_brier = 0.0
        total_iou = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, target_ignition, target_intensity in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                target_ignition = target_ignition.to(self.device)
                target_intensity = target_intensity.to(self.device)
                
                # Forward pass
                pred_ignition, pred_intensity = self.model(inputs)
                
                # Calculate metrics
                mse = torch.nn.functional.mse_loss(pred_intensity, target_intensity)
                bce = torch.nn.functional.binary_cross_entropy(pred_ignition, target_ignition)
                brier = brier_score(pred_ignition, target_ignition)
                
                # Calculate IoU for binary ignition
                pred_binary = (pred_ignition > 0.5).float()
                target_binary = (target_ignition > 0.5).float()
                intersection = (pred_binary * target_binary).sum()
                union = (pred_binary + target_binary).clamp(0, 1).sum()
                iou = intersection / (union + 1e-6)
                
                # Accumulate
                batch_size = inputs.shape[0]
                total_mse += mse.item() * batch_size
                total_bce += bce.item() * batch_size
                total_brier += brier.item() * batch_size
                total_iou += iou.item() * batch_size
                total_samples += batch_size
        
        # Compute averages
        metrics = {
            'mse': total_mse / total_samples,
            'bce': total_bce / total_samples,
            'brier': total_brier / total_samples,
            'iou': total_iou / total_samples
        }
        
        return metrics
    
    def evaluate_sample(self, inputs: torch.Tensor, target_ignition: torch.Tensor, 
                       target_intensity: torch.Tensor) -> Dict:
        """
        Evaluate a single sample and return predictions.
        
        Args:
            inputs: Input tensor (C, H, W)
            target_ignition: Target ignition (1, H, W)
            target_intensity: Target intensity (1, H, W)
        
        Returns:
            Dictionary with predictions and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension
            inputs = inputs.unsqueeze(0).to(self.device)
            target_ignition = target_ignition.unsqueeze(0).to(self.device)
            target_intensity = target_intensity.unsqueeze(0).to(self.device)
            
            # Predict
            pred_ignition, pred_intensity = self.model(inputs)
            
            # Calculate metrics
            mse = torch.nn.functional.mse_loss(pred_intensity, target_intensity)
            bce = torch.nn.functional.binary_cross_entropy(pred_ignition, target_ignition)
            brier = brier_score(pred_ignition, target_ignition)
            
            # Convert to numpy
            pred_ignition_np = pred_ignition.squeeze().cpu().numpy()
            pred_intensity_np = pred_intensity.squeeze().cpu().numpy()
            target_ignition_np = target_ignition.squeeze().cpu().numpy()
            target_intensity_np = target_intensity.squeeze().cpu().numpy()
        
        return {
            'pred_ignition': pred_ignition_np,
            'pred_intensity': pred_intensity_np,
            'target_ignition': target_ignition_np,
            'target_intensity': target_intensity_np,
            'mse': mse.item(),
            'bce': bce.item(),
            'brier': brier.item()
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate FireSpreadNet surrogate model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda, cpu, or auto)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model-type', type=str, default='full',
                       choices=['full', 'lite'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset info
    data_dir = Path(args.data_dir)
    info_file = data_dir / 'dataset_info.json'
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Load model
    in_channels = dataset_info['input_channels']
    if args.model_type == 'full':
        model = FireSpreadNet(in_channels=in_channels)
    else:
        model = FireSpreadNetLite(in_channels=in_channels)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {args.model_path}")
    logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    # Create dataset
    dataset = FireSpreadDataset(args.data_dir, split=args.split)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create evaluator
    evaluator = SurrogateEvaluator(model, device)
    
    # Evaluate
    metrics = evaluator.evaluate(loader)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Split: {args.split}")
    logger.info(f"Samples: {len(dataset)}")
    logger.info(f"MSE (Intensity): {metrics['mse']:.6f}")
    logger.info(f"BCE (Ignition): {metrics['bce']:.6f}")
    logger.info(f"Brier Score: {metrics['brier']:.6f}")
    logger.info(f"IoU (Ignition): {metrics['iou']:.4f}")
    logger.info("="*60)
    
    # Check acceptance criteria
    logger.info("\nACCEPTANCE CRITERIA:")
    mse_pass = metrics['mse'] < 0.10
    bce_pass = metrics['bce'] < 0.5
    brier_pass = metrics['brier'] < 0.25
    
    logger.info(f"MSE < 0.10: {'✓ PASS' if mse_pass else '✗ FAIL'} ({metrics['mse']:.6f})")
    logger.info(f"BCE < 0.50: {'✓ PASS' if bce_pass else '✗ FAIL'} ({metrics['bce']:.6f})")
    logger.info(f"Brier < 0.25: {'✓ PASS' if brier_pass else '✗ FAIL'} ({metrics['brier']:.6f})")
    
    all_pass = mse_pass and bce_pass and brier_pass
    logger.info(f"\nOverall: {'✓ ALL CRITERIA MET' if all_pass else '✗ SOME CRITERIA NOT MET'}")
    
    # Save results
    results = {
        'model_path': str(args.model_path),
        'data_dir': str(args.data_dir),
        'split': args.split,
        'num_samples': len(dataset),
        'metrics': metrics,
        'acceptance_criteria': {
            'mse_pass': mse_pass,
            'bce_pass': bce_pass,
            'brier_pass': brier_pass,
            'all_pass': all_pass
        }
    }
    
    results_file = output_dir / f'evaluation_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")


if __name__ == '__main__':
    main()
