#!/usr/bin/env python3
"""
Dry Run Training Test - Fire Prevention System
Tests all training modules with minimal epochs to validate functionality
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_fire_training():
    """Test fire detection training"""
    logger.info("Testing Fire Detection Training...")
    
    try:
        from fire.fire_detection import FireDetectionModel
          # Create model
        model = FireDetectionModel()
        
        # Create dummy data (more realistic for YOLO)
        batch_size = 2
        # Create RGB images with values in [0, 255] range
        images = torch.randint(0, 256, (batch_size, 3, 640, 640), dtype=torch.uint8)
        targets = [
            {
                'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.int64)  # fire class
            } for _ in range(batch_size)
        ]
        
        # Test forward pass
        model.train()
        outputs = model(images, targets)
        logger.info(f"Fire training forward pass successful. Loss keys: {list(outputs.keys())}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        logger.info(f"Fire inference successful. Predictions: {len(predictions)} items")
        
        return True
        
    except Exception as e:
        logger.error(f"Fire training test failed: {e}")
        return False

def test_smoke_training():
    """Test smoke detection training"""
    logger.info("Testing Smoke Detection Training...")
    
    try:
        from smoke.smoke_detection import SmokeDetectionModel
        
        # Create model
        model = SmokeDetectionModel()
        
        # Create dummy video data (batch_size, frames, channels, height, width)
        batch_size = 1
        frames = 8
        video_data = torch.randn(batch_size, frames, 3, 224, 224)
        labels = torch.tensor([1], dtype=torch.long)  # smoke present
        
        # Test forward pass
        model.train()
        outputs = model(video_data)
        logger.info(f"Smoke training forward pass successful. Output shape: {outputs.shape}")
        
        # Test loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        logger.info(f"Smoke loss calculation successful. Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Smoke training test failed: {e}")
        return False

def test_fauna_training():
    """Test fauna detection training"""
    logger.info("Testing Fauna Detection Training...")
    
    try:
        from fauna.fauna_detection import FaunaDetectionModel
          # Create model
        model = FaunaDetectionModel()
        
        # Create dummy data (more realistic for YOLO)
        batch_size = 2
        # Create RGB images with values in [0, 255] range
        images = torch.randint(0, 256, (batch_size, 3, 640, 640), dtype=torch.uint8)
        targets = [
            {
                'boxes': torch.tensor([[150, 150, 250, 250]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.int64)  # animal class
            } for _ in range(batch_size)
        ]
        
        # Test forward pass
        model.train()
        outputs = model(images, targets)
        logger.info(f"Fauna training forward pass successful. Loss keys: {list(outputs.keys())}")
        
        # Test health assessment
        health_features = torch.randn(batch_size, 1000)  # Features from backbone
        health_probs = model.assess_health(health_features)
        logger.info(f"Fauna health assessment successful. Health probs shape: {health_probs.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fauna training test failed: {e}")
        return False

def test_vegetation_training():
    """Test vegetation health training"""
    logger.info("Testing Vegetation Health Training...")
    
    try:
        from vegetation.vegetation_health import VegetationHealthModel
        
        # Create model
        model = VegetationHealthModel()
        
        # Create dummy data
        batch_size = 4
        images = torch.randn(batch_size, 3, 256, 256)
        vari_indices = torch.randn(batch_size, 1)
        health_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # health classes
        
        # Test forward pass
        model.train()
        outputs = model(images, vari_indices)
        logger.info(f"Vegetation training forward pass successful. Output shape: {outputs.shape}")
        
        # Test loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, health_labels)
        logger.info(f"Vegetation loss calculation successful. Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Vegetation training test failed: {e}")
        return False

def test_geospatial_projection():
    """Test geospatial projection"""
    logger.info("Testing Geospatial Projection...")
    
    try:
        from geospatial.geospatial_projection import GeospatialProjector
          # Create projector
        projector = GeospatialProjector("data/dem/sample_dem.tif", "config/intrinsics.json")
        
        # Test projection
        detections = [
            {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'class': 'fire', 'confidence': 0.9}
        ]
        
        # The project_detections method expects frame_id, not pose directly
        frame_id = 0
        
        projected = projector.project_detections(detections, frame_id)
        logger.info(f"Geospatial projection successful. Projected: {len(projected)} detections")
        
        return True
        
    except Exception as e:
        logger.error(f"Geospatial projection test failed: {e}")
        return False

def test_fire_spread_simulation():
    """Test fire spread simulation"""
    logger.info("Testing Fire Spread Simulation...")
    
    try:
        from spread.fire_spread_simulation import FireSpreadSimulator
        
        # Create simulator
        simulator = FireSpreadSimulator()
          # Test physics-based simulation
        initial_fire_state = np.zeros((64, 64), dtype=np.float32)
        initial_fire_state[30:34, 30:34] = 1.0  # Fire area
        
        env_state = {
            'wind_x': np.full((64, 64), 0.5, dtype=np.float32),
            'wind_y': np.full((64, 64), 0.3, dtype=np.float32),
            'humidity': np.full((64, 64), 0.4, dtype=np.float32),
            'slope': np.zeros((64, 64), dtype=np.float32),
            'vegetation': np.full((64, 64), 0.8, dtype=np.float32)
        }
        
        spread_prediction = simulator.simulate(
            initial_fire_state, env_state, steps=5, use_neural=False
        )
        logger.info(f"Physics simulation successful. Predicted spread: {len(spread_prediction)} timesteps")
        
        # Test neural simulation
        predicted_state = simulator.simulate(
            initial_fire_state, env_state, steps=3, use_neural=True
        )
        logger.info(f"Neural simulation successful. Output timesteps: {len(predicted_state)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fire spread simulation test failed: {e}")
        return False

def main():
    """Run all dry-run training tests"""
    logger.info("🔥 Fire Prevention System - Dry Run Training Tests")
    logger.info("=" * 60)
    
    start_time = time.time()
    tests = [
        ("Fire Detection", test_fire_training),
        ("Smoke Detection", test_smoke_training), 
        ("Fauna Detection", test_fauna_training),
        ("Vegetation Health", test_vegetation_training),
        ("Geospatial Projection", test_geospatial_projection),
        ("Fire Spread Simulation", test_fire_spread_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Testing {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n🎯 Dry Run Training Test Summary")
    logger.info("=" * 40)
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<25} {status}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        logger.info("🎉 All dry-run training tests PASSED!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
