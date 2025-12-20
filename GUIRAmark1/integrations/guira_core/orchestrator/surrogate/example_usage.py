"""
Example Usage of FireSpreadNet Surrogate Model

Demonstrates how to:
1. Generate a synthetic dataset
2. Train the surrogate model
3. Load and use the trained model for prediction
4. Evaluate model performance

This is a complete end-to-end example.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import tempfile
import shutil

# Import surrogate components
from dataset_builder import DatasetBuilder
from models import FireSpreadNet, FireSpreadNetLite
from train import FireSpreadDataset, SurrogateTrainer
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_generate_dataset():
    """Example: Generate a synthetic dataset."""
    logger.info("="*60)
    logger.info("EXAMPLE 1: Generate Synthetic Dataset")
    logger.info("="*60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Creating dataset in: {temp_dir}")
    
    # Initialize dataset builder
    builder = DatasetBuilder(temp_dir, grid_size=(64, 64))
    
    # Generate 20 synthetic runs with varying parameters
    logger.info("Generating 20 synthetic simulation runs...")
    for i in range(20):
        # Vary parameters
        wind_speed = 2.0 + i * 0.5  # 2-12 m/s
        wind_direction = (i * 18) % 360  # Rotate through directions
        fuel_moisture = 0.2 + (i % 5) * 0.1  # 0.2-0.6
        
        builder.add_synthetic_run(
            run_id=f'example_run_{i:02d}',
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            fuel_moisture=fuel_moisture,
            n_timesteps=10
        )
    
    # Finalize dataset
    logger.info("Finalizing dataset...")
    builder.finalize()
    
    logger.info(f"✓ Dataset created with {len(builder.metadata)} samples")
    logger.info(f"  Location: {temp_dir}")
    
    return temp_dir


def example_train_model(data_dir, epochs=5):
    """Example: Train surrogate model on dataset."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Train Surrogate Model")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_dataset = FireSpreadDataset(data_dir, split='train')
    val_dataset = FireSpreadDataset(data_dir, split='val')
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model (use lite for faster training in example)
    model = FireSpreadNetLite(in_channels=6)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create temporary save directory
    save_dir = tempfile.mkdtemp()
    logger.info(f"Model checkpoints will be saved to: {save_dir}")
    
    # Train (just a few epochs for demo)
    logger.info(f"Training for {epochs} epochs...")
    
    # Simple training loop (without MLflow for this example)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, target_ign, target_int in train_loader:
            inputs = inputs.to(device)
            target_ign = target_ign.to(device)
            target_int = target_int.to(device)
            
            optimizer.zero_grad()
            pred_ign, pred_int = model(inputs)
            
            loss = (torch.nn.functional.binary_cross_entropy(pred_ign, target_ign) +
                   torch.nn.functional.mse_loss(pred_int, target_int))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, target_ign, target_int in val_loader:
                inputs = inputs.to(device)
                target_ign = target_ign.to(device)
                target_int = target_int.to(device)
                
                pred_ign, pred_int = model(inputs)
                
                loss = (torch.nn.functional.binary_cross_entropy(pred_ign, target_ign) +
                       torch.nn.functional.mse_loss(pred_int, target_int))
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(save_dir) / 'best_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, model_path)
    
    logger.info(f"✓ Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Model saved to: {model_path}")
    
    return model_path, device


def example_inference(model_path, device):
    """Example: Load model and run inference."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Model Inference")
    logger.info("="*60)
    
    # Load model
    model = FireSpreadNetLite(in_channels=6)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    
    # Create synthetic input
    H, W = 64, 64
    
    # Simulate a small fire in the center
    fire_t0 = np.zeros((H, W), dtype=np.float32)
    fire_t0[30:34, 30:34] = 0.8  # Initial fire patch
    
    # Wind blowing east (positive u component)
    wind_u = np.ones((H, W), dtype=np.float32) * 5.0  # 5 m/s east
    wind_v = np.zeros((H, W), dtype=np.float32)
    
    # Environmental conditions
    humidity = np.ones((H, W), dtype=np.float32) * 0.4  # 40% humidity
    fuel_density = np.ones((H, W), dtype=np.float32) * 0.7  # 70% fuel density
    slope = np.zeros((H, W), dtype=np.float32)  # Flat terrain
    
    # Stack inputs
    input_stack = np.stack([
        fire_t0,
        wind_u,
        wind_v,
        humidity,
        fuel_density,
        slope
    ], axis=0)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(input_stack).unsqueeze(0).to(device)
    
    logger.info("Input conditions:")
    logger.info(f"  Initial fire area: {(fire_t0 > 0).sum()} cells")
    logger.info(f"  Wind: 5 m/s eastward")
    logger.info(f"  Humidity: 40%")
    logger.info(f"  Fuel density: 70%")
    
    # Run inference
    with torch.no_grad():
        pred_ignition, pred_intensity = model(input_tensor)
    
    # Extract predictions
    pred_ignition_map = pred_ignition.squeeze().cpu().numpy()
    pred_intensity_map = pred_intensity.squeeze().cpu().numpy()
    
    # Analyze predictions
    ignition_threshold = 0.5
    predicted_fire = pred_ignition_map > ignition_threshold
    
    logger.info("\nPredictions:")
    logger.info(f"  Predicted fire area: {predicted_fire.sum()} cells")
    logger.info(f"  Growth: {predicted_fire.sum() - (fire_t0 > 0).sum()} cells")
    logger.info(f"  Max ignition probability: {pred_ignition_map.max():.3f}")
    logger.info(f"  Max predicted intensity: {pred_intensity_map.max():.3f}")
    logger.info(f"  Mean predicted intensity: {pred_intensity_map.mean():.3f}")
    
    # Check if fire spread eastward (as expected with eastward wind)
    center_y = W // 2
    left_half = predicted_fire[:, :center_y].sum()
    right_half = predicted_fire[:, center_y:].sum()
    
    if right_half > left_half:
        logger.info("✓ Fire correctly spread eastward with wind")
    else:
        logger.info("⚠ Fire did not spread as expected with wind")
    
    return pred_ignition_map, pred_intensity_map


def main():
    """Run all examples."""
    logger.info("\n" + "="*80)
    logger.info("FireSpreadNet Surrogate Model - Complete Example")
    logger.info("="*80 + "\n")
    
    # Example 1: Generate dataset
    data_dir = example_generate_dataset()
    
    # Example 2: Train model
    model_path, device = example_train_model(data_dir, epochs=10)
    
    # Example 3: Run inference
    pred_ignition, pred_intensity = example_inference(model_path, device)
    
    # Cleanup
    logger.info("\n" + "="*60)
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(data_dir)
    shutil.rmtree(Path(model_path).parent)
    
    logger.info("\n" + "="*80)
    logger.info("✓ All examples completed successfully!")
    logger.info("="*80)
    
    logger.info("\nNext steps:")
    logger.info("1. Generate larger dataset: python generate_ensemble.py --n-runs 1000")
    logger.info("2. Train full model: python train.py --data-dir physx_dataset --epochs 50")
    logger.info("3. Evaluate: python evaluate.py --model-path models/fire_spreadnet.pt")
    logger.info("4. Deploy for fast predictions in production")


if __name__ == '__main__':
    main()
