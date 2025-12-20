"""
Dataset Builder for FireSpreadNet Surrogate Model

Generates training datasets from PhysX simulation outputs.
Each sample consists of:
- Input: Raster stack at t0 [fire_t0, wind_u, wind_v, humidity, fuel_density, slope]
- Target: Fire state at t1 [ignition_binary, intensity]
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationMetadata:
    """Metadata for a PhysX simulation run."""
    run_id: str
    wind_speed: float
    wind_direction: float
    fuel_moisture: float
    humidity: float
    temperature: float
    resolution_m: float
    timesteps: int
    grid_size: Tuple[int, int]
    timestamp: str


class DatasetBuilder:
    """
    Build surrogate training dataset from PhysX simulation outputs.
    
    Each PhysX run produces a time series of fire states. We extract
    consecutive pairs (t, t+1) to create training samples.
    """
    
    def __init__(self, output_dir: str, grid_size: Tuple[int, int] = (64, 64)):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save dataset
            grid_size: Target grid size (H, W) for resampling
        """
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.samples = []
        self.metadata = []
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
    
    def add_physx_run(self,
                      run_id: str,
                      fire_states: np.ndarray,
                      wind_field: np.ndarray,
                      humidity_field: np.ndarray,
                      fuel_density: np.ndarray,
                      slope: np.ndarray,
                      metadata: Dict) -> int:
        """
        Add samples from a PhysX simulation run.
        
        Args:
            run_id: Unique identifier for this run
            fire_states: Fire states over time, shape (T, H, W, 2) 
                        where [:,:,:,0] is ignition binary, [:,:,:,1] is intensity
            wind_field: Wind field, shape (H, W, 2) for (u, v) components
            humidity_field: Humidity field, shape (H, W)
            fuel_density: Fuel density map, shape (H, W)
            slope: Slope map, shape (H, W)
            metadata: Dictionary with simulation parameters
        
        Returns:
            Number of samples added
        """
        n_samples = 0
        T = fire_states.shape[0]
        
        # Resample all fields to target grid size if needed
        if fire_states.shape[1:3] != self.grid_size:
            fire_states = self._resample_temporal(fire_states, self.grid_size)
            wind_field = self._resample_spatial(wind_field, self.grid_size)
            humidity_field = self._resample_spatial(humidity_field, self.grid_size)
            fuel_density = self._resample_spatial(fuel_density, self.grid_size)
            slope = self._resample_spatial(slope, self.grid_size)
        
        # Extract consecutive pairs (t, t+1)
        for t in range(T - 1):
            sample_id = f"{run_id}_t{t:03d}"
            
            # Input: state at time t + static fields
            fire_t0 = fire_states[t, :, :, 1]  # Use intensity as input
            
            # Stack input channels: [fire_t0, wind_u, wind_v, humidity, fuel, slope]
            input_stack = np.stack([
                fire_t0,
                wind_field[:, :, 0],  # wind_u
                wind_field[:, :, 1],  # wind_v
                humidity_field,
                fuel_density,
                slope
            ], axis=0).astype(np.float32)
            
            # Target: state at time t+1
            target_ignition = fire_states[t + 1, :, :, 0].astype(np.float32)
            target_intensity = fire_states[t + 1, :, :, 1].astype(np.float32)
            
            # Save sample
            sample_path = self.output_dir / 'samples' / f'{sample_id}.npz'
            np.savez_compressed(
                sample_path,
                input=input_stack,
                target_ignition=target_ignition,
                target_intensity=target_intensity
            )
            
            # Store metadata
            sample_metadata = {
                'sample_id': sample_id,
                'run_id': run_id,
                'timestep': t,
                'wind_speed': metadata.get('wind_speed', 0.0),
                'wind_direction': metadata.get('wind_direction', 0.0),
                'fuel_moisture': metadata.get('fuel_moisture', 0.0),
                'humidity': metadata.get('humidity', 0.0)
            }
            self.metadata.append(sample_metadata)
            
            n_samples += 1
        
        logger.info(f"Added {n_samples} samples from run {run_id}")
        return n_samples
    
    def _resample_spatial(self, field: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resample a spatial field to target size using bilinear interpolation.
        
        Args:
            field: Input field, shape (H, W) or (H, W, C)
            target_size: Target (H, W)
        
        Returns:
            Resampled field
        """
        try:
            from scipy.ndimage import zoom
            
            if field.ndim == 2:
                factors = (target_size[0] / field.shape[0], target_size[1] / field.shape[1])
                return zoom(field, factors, order=1)
            elif field.ndim == 3:
                factors = (target_size[0] / field.shape[0], target_size[1] / field.shape[1], 1)
                return zoom(field, factors, order=1)
            else:
                raise ValueError(f"Unsupported field shape: {field.shape}")
        except ImportError:
            logger.warning("scipy not available, using simple reshape")
            # Fallback to simple averaging (not ideal but works)
            if field.ndim == 2:
                return field[:target_size[0], :target_size[1]]
            else:
                return field[:target_size[0], :target_size[1], :]
    
    def _resample_temporal(self, field: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resample a temporal sequence of spatial fields.
        
        Args:
            field: Input field, shape (T, H, W, C)
            target_size: Target (H, W)
        
        Returns:
            Resampled field, shape (T, target_H, target_W, C)
        """
        T, H, W, C = field.shape
        resampled = np.zeros((T, target_size[0], target_size[1], C), dtype=field.dtype)
        
        for t in range(T):
            for c in range(C):
                resampled[t, :, :, c] = self._resample_spatial(field[t, :, :, c], target_size)
        
        return resampled
    
    def add_synthetic_run(self,
                         run_id: str,
                         wind_speed: float,
                         wind_direction: float,
                         fuel_moisture: float,
                         n_timesteps: int = 10) -> int:
        """
        Add a synthetic simulation run for testing.
        
        Creates a simple expanding circular fire pattern with wind bias.
        
        Args:
            run_id: Unique run identifier
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees (0=North, 90=East)
            fuel_moisture: Fuel moisture content (0-1)
            n_timesteps: Number of timesteps to generate
        
        Returns:
            Number of samples added
        """
        H, W = self.grid_size
        
        # Generate static fields
        fuel_density = 0.5 + 0.3 * np.random.rand(H, W)
        slope = 5.0 + 10.0 * np.random.rand(H, W)
        humidity_field = 0.3 + 0.2 * np.random.rand(H, W)
        
        # Wind field (constant)
        wind_rad = np.radians(wind_direction)
        wind_u = wind_speed * np.sin(wind_rad) * np.ones((H, W))
        wind_v = wind_speed * np.cos(wind_rad) * np.ones((H, W))
        wind_field = np.stack([wind_u, wind_v], axis=-1)
        
        # Generate fire states
        fire_states = np.zeros((n_timesteps, H, W, 2), dtype=np.float32)
        
        # Initial ignition at center
        center_x, center_y = H // 2, W // 2
        fire_states[0, center_x-2:center_x+2, center_y-2:center_y+2, 0] = 1.0
        fire_states[0, center_x-2:center_x+2, center_y-2:center_y+2, 1] = 1.0
        
        # Simulate spread
        for t in range(1, n_timesteps):
            prev_ignition = fire_states[t-1, :, :, 0]
            
            # Simple spreading: dilate and add wind bias
            new_ignition = prev_ignition.copy()
            
            # Expand fire (simple dilation)
            from scipy.ndimage import binary_dilation
            structure = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])
            new_ignition = binary_dilation(new_ignition > 0.5, structure=structure).astype(np.float32)
            
            # Add wind bias
            wind_bias_x = int(np.round(wind_u[0, 0] * 0.1))
            wind_bias_y = int(np.round(wind_v[0, 0] * 0.1))
            
            if wind_bias_x != 0 or wind_bias_y != 0:
                new_ignition = np.roll(new_ignition, (wind_bias_x, wind_bias_y), axis=(0, 1))
            
            # Apply fuel and moisture effects
            spread_prob = fuel_density * (1.0 - fuel_moisture * 0.5)
            new_ignition = new_ignition * (np.random.rand(H, W) < spread_prob)
            
            fire_states[t, :, :, 0] = new_ignition
            fire_states[t, :, :, 1] = new_ignition * (0.5 + 0.5 * np.random.rand(H, W))
        
        # Add to dataset
        metadata = {
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'fuel_moisture': fuel_moisture,
            'humidity': humidity_field.mean()
        }
        
        return self.add_physx_run(
            run_id, fire_states, wind_field, 
            humidity_field, fuel_density, slope, metadata
        )
    
    def finalize(self, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Finalize dataset: save metadata and create train/val/test splits.
        
        Args:
            split_ratios: (train, val, test) split ratios
        """
        total_samples = len(self.metadata)
        logger.info(f"Finalizing dataset with {total_samples} samples")
        
        # Shuffle
        indices = np.random.permutation(total_samples)
        
        # Split
        train_split = int(total_samples * split_ratios[0])
        val_split = train_split + int(total_samples * split_ratios[1])
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        
        # Create split files
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        for split_name, split_indices in splits.items():
            split_metadata = [self.metadata[i] for i in split_indices]
            split_file = self.output_dir / 'metadata' / f'{split_name}.json'
            
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
            
            logger.info(f"  {split_name}: {len(split_indices)} samples")
        
        # Save full metadata
        metadata_file = self.output_dir / 'metadata' / 'full.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save dataset info
        info = {
            'total_samples': total_samples,
            'grid_size': self.grid_size,
            'input_channels': 6,
            'splits': {
                'train': len(train_indices),
                'val': len(val_indices),
                'test': len(test_indices)
            },
            'channel_names': ['fire_t0', 'wind_u', 'wind_v', 'humidity', 'fuel_density', 'slope']
        }
        
        info_file = self.output_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset saved to {self.output_dir}")


def load_sample(sample_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single sample from disk.
    
    Args:
        sample_path: Path to .npz file
    
    Returns:
        input_stack: Input raster stack, shape (C, H, W)
        target_ignition: Target ignition binary mask, shape (H, W)
        target_intensity: Target intensity map, shape (H, W)
    """
    data = np.load(sample_path)
    return data['input'], data['target_ignition'], data['target_intensity']
