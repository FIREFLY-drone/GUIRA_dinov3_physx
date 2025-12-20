"""
Generate Ensemble Dataset from PhysX Server

Runs PhysX fire spread simulations with parameter sweeps (wind, moisture)
and stores outputs for surrogate training.

Usage:
    python generate_ensemble.py --output-dir physx_dataset --n-runs 1000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset_builder import DatasetBuilder, SimulationMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterSampler:
    """Sample simulation parameters for ensemble generation."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize parameter sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
    
    def sample_wind(self) -> Tuple[float, float]:
        """
        Sample wind parameters.
        
        Returns:
            wind_speed: Wind speed in m/s (0-20)
            wind_direction: Wind direction in degrees (0-360)
        """
        # Wind speed distribution (log-normal)
        wind_speed = np.random.lognormal(mean=1.5, sigma=0.5)
        wind_speed = np.clip(wind_speed, 0.0, 20.0)
        
        # Wind direction (uniform)
        wind_direction = np.random.uniform(0, 360)
        
        return wind_speed, wind_direction
    
    def sample_fuel_moisture(self) -> float:
        """
        Sample fuel moisture content.
        
        Returns:
            fuel_moisture: Moisture content (0-1)
        """
        # Beta distribution for moisture
        moisture = np.random.beta(a=2, b=5)
        return moisture
    
    def sample_humidity(self) -> float:
        """
        Sample relative humidity.
        
        Returns:
            humidity: Relative humidity (0-1)
        """
        # Normal distribution centered at 0.4
        humidity = np.random.normal(loc=0.4, scale=0.15)
        humidity = np.clip(humidity, 0.1, 0.9)
        return humidity
    
    def sample_temperature(self) -> float:
        """
        Sample temperature in Celsius.
        
        Returns:
            temperature: Temperature in C (10-45)
        """
        temp = np.random.normal(loc=25, scale=8)
        temp = np.clip(temp, 10, 45)
        return temp


class MockPhysXRunner:
    """
    Mock PhysX runner for testing when PhysX server is not available.
    
    Generates synthetic fire spread data with realistic dynamics.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        """Initialize mock runner."""
        self.grid_size = grid_size
    
    def run_simulation(self,
                      wind_speed: float,
                      wind_direction: float,
                      fuel_moisture: float,
                      humidity: float,
                      temperature: float,
                      n_timesteps: int = 10) -> Tuple[np.ndarray, Dict]:
        """
        Run mock PhysX simulation.
        
        Args:
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees
            fuel_moisture: Fuel moisture (0-1)
            humidity: Relative humidity (0-1)
            temperature: Temperature in C
            n_timesteps: Number of timesteps
        
        Returns:
            fire_states: Fire states over time, shape (T, H, W, 2)
            fields: Dictionary with static fields
        """
        H, W = self.grid_size
        
        # Generate static fields
        fuel_density = self._generate_fuel_field()
        slope = self._generate_slope_field()
        
        # Wind field (constant with some spatial variation)
        wind_rad = np.radians(wind_direction)
        wind_u = wind_speed * np.sin(wind_rad) * (1 + 0.1 * np.random.randn(H, W))
        wind_v = wind_speed * np.cos(wind_rad) * (1 + 0.1 * np.random.randn(H, W))
        wind_field = np.stack([wind_u, wind_v], axis=-1)
        
        # Humidity field (with spatial variation)
        humidity_field = humidity * (1 + 0.1 * np.random.randn(H, W))
        humidity_field = np.clip(humidity_field, 0.1, 0.9)
        
        # Generate fire states
        fire_states = self._simulate_fire_spread(
            fuel_density, slope, wind_field,
            humidity_field, fuel_moisture, n_timesteps
        )
        
        fields = {
            'wind_field': wind_field,
            'humidity_field': humidity_field,
            'fuel_density': fuel_density,
            'slope': slope
        }
        
        return fire_states, fields
    
    def _generate_fuel_field(self) -> np.ndarray:
        """Generate synthetic fuel density field."""
        H, W = self.grid_size
        
        # Create patchy fuel distribution
        x = np.linspace(0, 10, W)
        y = np.linspace(0, 10, H)
        X, Y = np.meshgrid(x, y)
        
        # Multiple scales of variation
        fuel = 0.5 + 0.2 * np.sin(X) * np.cos(Y)
        fuel += 0.1 * np.sin(2 * X + 1) * np.cos(2 * Y + 1)
        fuel += 0.15 * np.random.randn(H, W)
        
        fuel = np.clip(fuel, 0.1, 1.0)
        return fuel
    
    def _generate_slope_field(self) -> np.ndarray:
        """Generate synthetic slope field."""
        H, W = self.grid_size
        
        # Create terrain with some slope
        x = np.linspace(0, 5, W)
        y = np.linspace(0, 5, H)
        X, Y = np.meshgrid(x, y)
        
        slope = 5 + 15 * np.sin(0.5 * X) * np.cos(0.5 * Y)
        slope += 5 * np.random.randn(H, W)
        slope = np.clip(slope, 0, 30)
        
        return slope
    
    def _simulate_fire_spread(self,
                             fuel: np.ndarray,
                             slope: np.ndarray,
                             wind: np.ndarray,
                             humidity: np.ndarray,
                             moisture: float,
                             n_timesteps: int) -> np.ndarray:
        """Simulate fire spread with physics-inspired rules."""
        H, W = self.grid_size
        fire_states = np.zeros((n_timesteps, H, W, 2), dtype=np.float32)
        
        # Initial ignition (random location)
        center_x = np.random.randint(H // 4, 3 * H // 4)
        center_y = np.random.randint(W // 4, 3 * W // 4)
        fire_states[0, center_x-1:center_x+2, center_y-1:center_y+2, 0] = 1.0
        fire_states[0, center_x-1:center_x+2, center_y-1:center_y+2, 1] = 0.8
        
        # Simulate spread
        for t in range(1, n_timesteps):
            prev_ignition = fire_states[t-1, :, :, 0]
            prev_intensity = fire_states[t-1, :, :, 1]
            
            # Calculate spread probability for each cell
            spread_prob = np.zeros((H, W))
            
            # Check neighbors for existing fire
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Shift fire map
                    shifted = np.roll(np.roll(prev_ignition, dx, axis=0), dy, axis=1)
                    
                    # Base spread rate
                    base_rate = 0.3
                    
                    # Wind effect (dot product with direction)
                    wind_u = wind[:, :, 0]
                    wind_v = wind[:, :, 1]
                    wind_effect = 1.0 + 0.1 * (wind_u * dx + wind_v * dy)
                    wind_effect = np.clip(wind_effect, 0.5, 2.0)
                    
                    # Fuel and moisture effect
                    fuel_effect = fuel * (1.0 - 0.5 * moisture)
                    
                    # Humidity effect
                    humidity_effect = 1.0 - 0.3 * humidity
                    
                    # Combine effects
                    cell_prob = base_rate * wind_effect * fuel_effect * humidity_effect
                    spread_prob += shifted * cell_prob
            
            # Normalize and apply randomness
            spread_prob = np.clip(spread_prob, 0, 1)
            new_ignition = (np.random.rand(H, W) < spread_prob).astype(np.float32)
            
            # Keep existing fire
            new_ignition = np.maximum(prev_ignition, new_ignition)
            
            # Intensity decreases slightly over time
            new_intensity = new_ignition * (0.6 + 0.4 * np.random.rand(H, W))
            
            fire_states[t, :, :, 0] = new_ignition
            fire_states[t, :, :, 1] = new_intensity
        
        return fire_states


def generate_ensemble_dataset(
    output_dir: str,
    n_runs: int,
    grid_size: Tuple[int, int] = (64, 64),
    n_timesteps: int = 10,
    use_physx: bool = False,
    seed: int = 42
):
    """
    Generate ensemble dataset from PhysX simulations.
    
    Args:
        output_dir: Output directory for dataset
        n_runs: Number of simulation runs
        grid_size: Grid size (H, W)
        n_timesteps: Timesteps per simulation
        use_physx: Whether to use real PhysX server (not implemented)
        seed: Random seed
    """
    logger.info(f"Generating ensemble dataset: {n_runs} runs")
    logger.info(f"Grid size: {grid_size}, Timesteps: {n_timesteps}")
    
    # Initialize dataset builder
    builder = DatasetBuilder(output_dir, grid_size)
    
    # Initialize parameter sampler
    sampler = ParameterSampler(seed=seed)
    
    # Initialize PhysX runner
    if use_physx:
        logger.error("Real PhysX integration not yet implemented")
        logger.info("Falling back to mock runner")
        use_physx = False
    
    runner = MockPhysXRunner(grid_size)
    
    # Generate runs
    total_samples = 0
    
    for run_idx in tqdm(range(n_runs), desc="Generating runs"):
        # Sample parameters
        wind_speed, wind_direction = sampler.sample_wind()
        fuel_moisture = sampler.sample_fuel_moisture()
        humidity = sampler.sample_humidity()
        temperature = sampler.sample_temperature()
        
        run_id = f"run_{run_idx:04d}"
        
        try:
            # Run simulation
            fire_states, fields = runner.run_simulation(
                wind_speed, wind_direction, fuel_moisture,
                humidity, temperature, n_timesteps
            )
            
            # Add to dataset
            metadata = {
                'wind_speed': float(wind_speed),
                'wind_direction': float(wind_direction),
                'fuel_moisture': float(fuel_moisture),
                'humidity': float(humidity),
                'temperature': float(temperature)
            }
            
            n_samples = builder.add_physx_run(
                run_id,
                fire_states,
                fields['wind_field'],
                fields['humidity_field'],
                fields['fuel_density'],
                fields['slope'],
                metadata
            )
            
            total_samples += n_samples
            
        except Exception as e:
            logger.error(f"Failed to generate run {run_id}: {e}")
            continue
    
    # Finalize dataset
    logger.info(f"Generated {total_samples} samples from {n_runs} runs")
    builder.finalize()
    
    logger.info("✓ Dataset generation complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate ensemble dataset from PhysX simulations'
    )
    parser.add_argument('--output-dir', type=str, default='physx_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--n-runs', type=int, default=1000,
                       help='Number of simulation runs')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[64, 64],
                       help='Grid size (H W)')
    parser.add_argument('--n-timesteps', type=int, default=10,
                       help='Number of timesteps per simulation')
    parser.add_argument('--use-physx', action='store_true',
                       help='Use real PhysX server (not implemented)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    grid_size = tuple(args.grid_size)
    
    generate_ensemble_dataset(
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        grid_size=grid_size,
        n_timesteps=args.n_timesteps,
        use_physx=args.use_physx,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
