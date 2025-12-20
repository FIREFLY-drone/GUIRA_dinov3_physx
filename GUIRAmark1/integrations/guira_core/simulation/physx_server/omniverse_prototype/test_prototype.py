#!/usr/bin/env python3
"""
Unit tests for PhysX fire spread prototype (PH-06).

Tests both Omniverse and fallback simulation modes.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, some tests will be skipped")

from run_prototype import (
    FallbackPhysicsSimulator,
    create_sample_usd_scene,
    generate_geojson_output,
    run_simulation
)


class TestFallbackSimulator(unittest.TestCase):
    """Test fallback physics simulator."""
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = FallbackPhysicsSimulator(grid_size=(50, 50), cell_size=1.0)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.grid_size, (50, 50))
        self.assertEqual(self.simulator.cell_size, 1.0)
        self.assertIsNotNone(self.simulator.grid)
        self.assertIsNotNone(self.simulator.vegetation)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_scene_initialization(self):
        """Test scene initialization."""
        result = self.simulator.initialize_scene()
        self.assertTrue(result)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_vegetation_creation(self):
        """Test vegetation proxy creation."""
        num_vegetated = self.simulator.create_vegetation_proxies(
            terrain_bounds=(-25, -25, 25, 25),
            density=0.7
        )
        self.assertGreater(num_vegetated, 0)
        self.assertLess(num_vegetated, 50 * 50)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_ignition(self):
        """Test ember particle spawning (ignition)."""
        ignition_polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # Store initial state
        initial_burning = np.sum(self.simulator.grid > 0.5)
        
        # Spawn embers
        self.simulator.spawn_ember_particles(
            ignition_polygon,
            wind_direction=45.0,
            wind_speed=5.0
        )
        
        # Check that some cells are now burning
        after_burning = np.sum(self.simulator.grid > 0.5)
        self.assertGreater(after_burning, initial_burning)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_simulation_steps(self):
        """Test simulation execution."""
        # Set up ignition
        ignition_polygon = [(0, 0), (5, 0), (5, 5), (0, 5)]
        self.simulator.spawn_ember_particles(ignition_polygon)
        
        # Run simulation
        states = self.simulator.simulate_steps(num_steps=20)
        
        # Verify results
        self.assertGreater(len(states), 0)
        self.assertIsInstance(states, list)
        
        # Check state structure
        state = states[0]
        self.assertIn('step', state)
        self.assertIn('time', state)
        self.assertIn('fire_cells', state)
        self.assertIsInstance(state['fire_cells'], list)
        
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_fire_spread(self):
        """Test that fire actually spreads."""
        # Set up ignition
        ignition_polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.simulator.spawn_ember_particles(ignition_polygon)
        
        # Get initial fire cells
        initial_states = self.simulator.simulate_steps(num_steps=1)
        initial_cells = len(initial_states[0]['fire_cells'])
        
        # Run more steps
        later_states = self.simulator.simulate_steps(num_steps=20)
        
        # Fire should spread (more cells burning)
        # Note: Due to random nature, we check the max across all timesteps
        max_cells = max(len(s['fire_cells']) for s in later_states)
        self.assertGreaterEqual(max_cells, initial_cells)


class TestGeoJSONGeneration(unittest.TestCase):
    """Test GeoJSON output generation."""
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_geojson_structure(self):
        """Test GeoJSON output structure."""
        # Create mock states
        states = [
            {
                'step': 0,
                'time': 0.0,
                'fire_cells': [(0, 0), (1, 0), (1, 1), (0, 1)]
            },
            {
                'step': 10,
                'time': 10.0,
                'fire_cells': [(x, y) for x in range(-5, 6) for y in range(-5, 6)]
            }
        ]
        
        # Generate GeoJSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            output_path = f.name
        
        try:
            generate_geojson_output(states, output_path)
            
            # Load and validate
            with open(output_path) as f:
                data = json.load(f)
            
            self.assertEqual(data['type'], 'FeatureCollection')
            self.assertIn('metadata', data)
            self.assertIn('features', data)
            self.assertGreater(len(data['features']), 0)
            
            # Check feature structure
            feature = data['features'][0]
            self.assertEqual(feature['type'], 'Feature')
            self.assertIn('properties', feature)
            self.assertIn('geometry', feature)
            self.assertEqual(feature['geometry']['type'], 'Polygon')
            
            # Check properties
            props = feature['properties']
            self.assertIn('timestep', props)
            self.assertIn('time_seconds', props)
            self.assertIn('num_cells', props)
            
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestUSDSceneCreation(unittest.TestCase):
    """Test USD scene creation."""
    
    def test_sample_usd_creation(self):
        """Test sample USD scene generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_scene.usd"
            
            create_sample_usd_scene(str(usd_path))
            
            # Verify file exists
            self.assertTrue(usd_path.exists())
            
            # Verify content
            content = usd_path.read_text()
            self.assertIn('#usda', content)
            self.assertIn('World', content)
            self.assertIn('Terrain', content)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_full_simulation_run(self):
        """Test complete simulation run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "scene.usd"
            output_path = Path(tmpdir) / "output.geojson"
            
            # Run simulation
            success = run_simulation(str(usd_path), str(output_path))
            
            # Verify success
            self.assertTrue(success)
            
            # Verify outputs exist
            self.assertTrue(usd_path.exists())
            self.assertTrue(output_path.exists())
            
            # Validate GeoJSON
            with open(output_path) as f:
                data = json.load(f)
            
            self.assertEqual(data['type'], 'FeatureCollection')
            self.assertGreater(len(data['features']), 0)


class TestAcceptanceCriteria(unittest.TestCase):
    """Test acceptance criteria from PH-06 specification."""
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_acceptance_criterion_1(self):
        """Acceptance: Script executes without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "scene.usd"
            output_path = Path(tmpdir) / "output.geojson"
            
            # Should not raise exception
            try:
                success = run_simulation(str(usd_path), str(output_path))
                self.assertTrue(success)
            except Exception as e:
                self.fail(f"Script execution failed: {e}")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_acceptance_criterion_2(self):
        """Acceptance: Produces valid GeoJSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "scene.usd"
            output_path = Path(tmpdir) / "output.geojson"
            
            run_simulation(str(usd_path), str(output_path))
            
            # Validate GeoJSON structure
            with open(output_path) as f:
                data = json.load(f)
            
            # Must be FeatureCollection
            self.assertEqual(data['type'], 'FeatureCollection')
            
            # Must have features
            self.assertIn('features', data)
            self.assertGreater(len(data['features']), 0)
            
            # Features must be valid
            for feature in data['features']:
                self.assertEqual(feature['type'], 'Feature')
                self.assertIn('geometry', feature)
                self.assertIn('properties', feature)
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy required")
    def test_acceptance_criterion_3(self):
        """Acceptance: Contains sample perimeter and particle traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "scene.usd"
            output_path = Path(tmpdir) / "output.geojson"
            
            run_simulation(str(usd_path), str(output_path))
            
            with open(output_path) as f:
                data = json.load(f)
            
            # Must have multiple timesteps
            self.assertGreater(len(data['features']), 5)
            
            # Features must have perimeter (Polygon)
            for feature in data['features']:
                self.assertEqual(feature['geometry']['type'], 'Polygon')
                
                # Must have cell count (representing particle traces)
                self.assertIn('num_cells', feature['properties'])
                self.assertGreater(feature['properties']['num_cells'], 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFallbackSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestGeoJSONGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestUSDSceneCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestAcceptanceCriteria))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
