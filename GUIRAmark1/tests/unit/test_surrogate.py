"""
Unit tests for FireSpreadNet surrogate model.
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

# Add surrogate module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'integrations' / 'guira_core' / 'orchestrator' / 'surrogate'))

from models import FireSpreadNet, FireSpreadNetLite, combined_loss, brier_score
from dataset_builder import DatasetBuilder, load_sample


class TestFireSpreadNet(unittest.TestCase):
    """Test FireSpreadNet model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.in_channels = 6
        self.height = 64
        self.width = 64
        
        # Create sample input
        self.sample_input = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width
        )
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = FireSpreadNet(in_channels=self.in_channels)
        self.assertIsInstance(model, FireSpreadNet)
    
    def test_model_forward(self):
        """Test forward pass produces correct output shapes."""
        model = FireSpreadNet(in_channels=self.in_channels)
        ignition_prob, intensity = model(self.sample_input)
        
        # Check output shapes
        self.assertEqual(ignition_prob.shape, (self.batch_size, 1, self.height, self.width))
        self.assertEqual(intensity.shape, (self.batch_size, 1, self.height, self.width))
        
        # Check output ranges
        self.assertTrue(torch.all(ignition_prob >= 0) and torch.all(ignition_prob <= 1))
        self.assertTrue(torch.all(intensity >= 0))
    
    def test_model_lite_forward(self):
        """Test lightweight model forward pass."""
        model = FireSpreadNetLite(in_channels=self.in_channels)
        ignition_prob, intensity = model(self.sample_input)
        
        # Check output shapes
        self.assertEqual(ignition_prob.shape, (self.batch_size, 1, self.height, self.width))
        self.assertEqual(intensity.shape, (self.batch_size, 1, self.height, self.width))
    
    def test_model_device_transfer(self):
        """Test model can be moved to different devices."""
        model = FireSpreadNet(in_channels=self.in_channels)
        
        # Test CPU
        model_cpu = model.to('cpu')
        self.assertEqual(next(model_cpu.parameters()).device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            self.assertEqual(next(model_cuda.parameters()).device.type, 'cuda')


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 64
        self.width = 64
        
        # Create sample predictions and targets
        self.pred_ignition = torch.rand(self.batch_size, 1, self.height, self.width)
        self.pred_intensity = torch.rand(self.batch_size, 1, self.height, self.width) * 2
        self.target_ignition = torch.randint(0, 2, (self.batch_size, 1, self.height, self.width)).float()
        self.target_intensity = torch.rand(self.batch_size, 1, self.height, self.width) * 2
    
    def test_brier_score(self):
        """Test Brier score calculation."""
        score = brier_score(self.pred_ignition, self.target_ignition)
        
        self.assertIsInstance(score, torch.Tensor)
        self.assertTrue(score >= 0)
        self.assertTrue(score <= 1)
    
    def test_combined_loss(self):
        """Test combined loss calculation."""
        loss, loss_dict = combined_loss(
            self.pred_ignition,
            self.pred_intensity,
            self.target_ignition,
            self.target_intensity
        )
        
        # Check loss is valid
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss > 0)
        
        # Check loss dict
        self.assertIn('bce', loss_dict)
        self.assertIn('mse', loss_dict)
        self.assertIn('brier', loss_dict)
        self.assertIn('total', loss_dict)
    
    def test_loss_backward(self):
        """Test loss can be used for backpropagation."""
        model = FireSpreadNet(in_channels=6)
        input_tensor = torch.randn(2, 6, 64, 64)
        
        # Forward pass
        pred_ignition, pred_intensity = model(input_tensor)
        
        # Calculate loss
        loss, _ = combined_loss(
            pred_ignition,
            pred_intensity,
            self.target_ignition,
            self.target_intensity
        )
        
        # Backward pass should not raise
        loss.backward()


class TestDatasetBuilder(unittest.TestCase):
    """Test dataset builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.grid_size = (32, 32)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_builder_initialization(self):
        """Test builder can be initialized."""
        builder = DatasetBuilder(self.temp_dir, self.grid_size)
        
        # Check directories created
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / 'samples').exists())
        self.assertTrue((Path(self.temp_dir) / 'metadata').exists())
    
    def test_add_synthetic_run(self):
        """Test adding synthetic simulation run."""
        builder = DatasetBuilder(self.temp_dir, self.grid_size)
        
        n_samples = builder.add_synthetic_run(
            run_id='test_run',
            wind_speed=5.0,
            wind_direction=45.0,
            fuel_moisture=0.3,
            n_timesteps=5
        )
        
        # Should create 4 samples (5 timesteps - 1)
        self.assertEqual(n_samples, 4)
        
        # Check samples directory
        samples = list((Path(self.temp_dir) / 'samples').glob('*.npz'))
        self.assertEqual(len(samples), 4)
    
    def test_add_physx_run(self):
        """Test adding PhysX simulation run."""
        builder = DatasetBuilder(self.temp_dir, self.grid_size)
        
        H, W = self.grid_size
        T = 5
        
        # Create synthetic data
        fire_states = np.random.rand(T, H, W, 2).astype(np.float32)
        wind_field = np.random.rand(H, W, 2).astype(np.float32)
        humidity_field = np.random.rand(H, W).astype(np.float32)
        fuel_density = np.random.rand(H, W).astype(np.float32)
        slope = np.random.rand(H, W).astype(np.float32)
        
        metadata = {
            'wind_speed': 5.0,
            'wind_direction': 45.0,
            'fuel_moisture': 0.3,
            'humidity': 0.5
        }
        
        n_samples = builder.add_physx_run(
            'test_run',
            fire_states,
            wind_field,
            humidity_field,
            fuel_density,
            slope,
            metadata
        )
        
        # Should create 4 samples (5 timesteps - 1)
        self.assertEqual(n_samples, 4)
    
    def test_finalize_dataset(self):
        """Test dataset finalization."""
        builder = DatasetBuilder(self.temp_dir, self.grid_size)
        
        # Add some synthetic runs
        for i in range(5):
            builder.add_synthetic_run(
                run_id=f'run_{i}',
                wind_speed=5.0,
                wind_direction=45.0,
                fuel_moisture=0.3,
                n_timesteps=5
            )
        
        # Finalize
        builder.finalize()
        
        # Check metadata files exist
        self.assertTrue((Path(self.temp_dir) / 'metadata' / 'train.json').exists())
        self.assertTrue((Path(self.temp_dir) / 'metadata' / 'val.json').exists())
        self.assertTrue((Path(self.temp_dir) / 'metadata' / 'test.json').exists())
        self.assertTrue((Path(self.temp_dir) / 'metadata' / 'full.json').exists())
        self.assertTrue((Path(self.temp_dir) / 'dataset_info.json').exists())
    
    def test_load_sample(self):
        """Test loading a sample."""
        builder = DatasetBuilder(self.temp_dir, self.grid_size)
        
        # Add a synthetic run
        builder.add_synthetic_run(
            run_id='test',
            wind_speed=5.0,
            wind_direction=45.0,
            fuel_moisture=0.3,
            n_timesteps=3
        )
        
        # Get sample path
        samples = list((Path(self.temp_dir) / 'samples').glob('*.npz'))
        self.assertTrue(len(samples) > 0)
        
        # Load sample
        input_stack, target_ignition, target_intensity = load_sample(str(samples[0]))
        
        # Check shapes
        H, W = self.grid_size
        self.assertEqual(input_stack.shape, (6, H, W))
        self.assertEqual(target_ignition.shape, (H, W))
        self.assertEqual(target_intensity.shape, (H, W))


class TestModelSaving(unittest.TestCase):
    """Test model saving and loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_model(self):
        """Test model can be saved and loaded."""
        model = FireSpreadNet(in_channels=6)
        
        # Save model
        save_path = Path(self.temp_dir) / 'test_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 10,
            'val_loss': 0.5
        }, save_path)
        
        # Load model
        checkpoint = torch.load(save_path)
        model_loaded = FireSpreadNet(in_channels=6)
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        
        # Check parameters match
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()
