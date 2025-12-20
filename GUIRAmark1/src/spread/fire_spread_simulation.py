"""
Fire Spread Simulation using physics-based and learned models.
Simulates fire propagation over terrain using environmental factors.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point, Polygon
import geojson
import yaml
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter, binary_dilation
import cv2


class FireSpreadDataset(Dataset):
    """Dataset for fire spread simulation training."""
    
    def __init__(self, data_dir: str, split: str = 'train', sequence_length: int = 5):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        
        # Load historical fire data
        self.satellite_dir = self.data_dir / 'satellite' / split
        self.annotations_dir = self.data_dir / 'annotations' / split
        
        self.sequences = []
        
        # Find time-series sequences
        if self.satellite_dir.exists() and self.annotations_dir.exists():
            # Group files by fire event
            fire_events = {}
            
            for sat_file in self.satellite_dir.glob('*.tif'):
                # Extract fire event and timestamp from filename
                # Expected format: fire_event_YYYYMMDD_HHMMSS.tif
                parts = sat_file.stem.split('_')
                if len(parts) >= 3:
                    event_id = '_'.join(parts[:-2])
                    timestamp = '_'.join(parts[-2:])
                    
                    if event_id not in fire_events:
                        fire_events[event_id] = []
                    
                    ann_file = self.annotations_dir / f"{sat_file.stem}.geojson"
                    if ann_file.exists():
                        fire_events[event_id].append({
                            'satellite': str(sat_file),
                            'annotation': str(ann_file),
                            'timestamp': timestamp
                        })
            
            # Create sequences
            for event_id, files in fire_events.items():
                if len(files) >= sequence_length:
                    # Sort by timestamp
                    files.sort(key=lambda x: x['timestamp'])
                    
                    # Create overlapping sequences
                    for i in range(len(files) - sequence_length + 1):
                        sequence = files[i:i + sequence_length]
                        self.sequences.append({
                            'event_id': event_id,
                            'sequence': sequence
                        })
        
        logger.info(f"Loaded {len(self.sequences)} fire spread sequences for {split} split")
    
    def __len__(self):
        return len(self.sequences)
    
    def load_satellite_image(self, sat_path: str) -> np.ndarray:
        """Load and preprocess satellite image."""
        with rasterio.open(sat_path) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)
            
            # Transpose to (height, width, bands)
            image = np.transpose(image, (1, 2, 0))
            
            # Normalize to [0, 1]
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            return image
    
    def load_fire_mask(self, ann_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """Load fire perimeter annotation and convert to mask."""
        try:
            gdf = gpd.read_file(ann_path)
            
            # Create binary mask
            mask = np.zeros(image_shape, dtype=np.float32)
            
            # This is a simplified approach - in practice, you'd need proper
            # coordinate system transformations
            for _, row in gdf.iterrows():
                geometry = row.geometry
                if isinstance(geometry, Polygon):
                    # Convert polygon to mask (simplified)
                    # In practice, use proper rasterization with rasterio.features.rasterize
                    bounds = geometry.bounds
                    minx, miny, maxx, maxy = bounds
                    
                    # Simple rectangular approximation
                    h, w = image_shape
                    x1 = int(minx * w)
                    y1 = int(miny * h)
                    x2 = int(maxx * w)
                    y2 = int(maxy * h)
                    
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    
                    mask[y1:y2, x1:x2] = 1.0
            
            return mask
            
        except Exception as e:
            logger.warning(f"Failed to load fire mask from {ann_path}: {e}")
            return np.zeros(image_shape, dtype=np.float32)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        sequence = sequence_data['sequence']
        
        # Load sequence of satellite images and fire masks
        images = []
        fire_masks = []
        
        for item in sequence:
            # Load satellite image
            sat_image = self.load_satellite_image(item['satellite'])
            images.append(sat_image)
            
            # Load fire mask
            fire_mask = self.load_fire_mask(item['annotation'], sat_image.shape[:2])
            fire_masks.append(fire_mask)
        
        # Convert to tensors
        images = np.stack(images)  # (T, H, W, C)
        fire_masks = np.stack(fire_masks)  # (T, H, W)
        
        # Transpose to (T, C, H, W) for PyTorch
        images = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
        fire_masks = torch.tensor(fire_masks, dtype=torch.float32)
        
        return {
            'images': images,
            'fire_masks': fire_masks,
            'event_id': sequence_data['event_id']
        }


class FireSpreadNet(nn.Module):
    """Neural network for learning fire spread dynamics."""
    
    def __init__(self, input_channels: int = 6, hidden_dim: int = 64, grid_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.input_channels = input_channels  # RGB + thermal + wind + vegetation
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        
        # Encoder for environmental state
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fire state processing
        self.fire_processor = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Combined processing
        self.combiner = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.ignition_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.intensity_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, env_state: torch.Tensor, fire_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            env_state: Environmental state (B, C, H, W)
            fire_state: Current fire state (B, 1, H, W)
        
        Returns:
            Ignition probability and fire intensity
        """
        # Encode environmental state
        env_features = self.encoder(env_state)
        
        # Process fire state
        fire_features = self.fire_processor(fire_state)
        
        # Combine features
        combined = torch.cat([env_features, fire_features], dim=1)
        combined_features = self.combiner(combined)
        
        # Predict outputs
        ignition_prob = self.ignition_head(combined_features)
        fire_intensity = self.intensity_head(combined_features)
        
        return ignition_prob, fire_intensity


class PhysicsBasedFireModel:
    """Physics-based fire spread model."""
    
    def __init__(self, grid_size: Tuple[int, int] = (256, 256), cell_size: float = 30.0):
        """
        Initialize physics model.
        
        Args:
            grid_size: Grid dimensions (height, width)
            cell_size: Size of each cell in meters
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Model parameters
        self.wind_factor = 0.3
        self.slope_factor = 0.2
        self.humidity_factor = -0.4
        self.vegetation_factor = 0.5
        
        logger.info(f"Initialized physics fire model with {grid_size} grid, {cell_size}m cells")
    
    def compute_spread_probability(self, wind_x: np.ndarray, wind_y: np.ndarray,
                                 humidity: np.ndarray, slope: np.ndarray,
                                 vegetation_density: np.ndarray) -> np.ndarray:
        """
        Compute fire spread probability based on environmental factors.
        
        Args:
            wind_x, wind_y: Wind components (m/s)
            humidity: Relative humidity (0-1)
            slope: Terrain slope (radians)
            vegetation_density: Vegetation density (0-1)
        
        Returns:
            Spread probability for each cell (0-1)
        """
        # Wind magnitude
        wind_magnitude = np.sqrt(wind_x**2 + wind_y**2)
        
        # Combine factors
        spread_factor = (
            self.wind_factor * wind_magnitude +
            self.slope_factor * np.abs(slope) +
            self.humidity_factor * humidity +
            self.vegetation_factor * vegetation_density
        )
        
        # Convert to probability using sigmoid
        spread_prob = 1 / (1 + np.exp(-spread_factor))
        
        return spread_prob
    
    def simulate_step(self, fire_state: np.ndarray, env_state: Dict[str, np.ndarray],
                     dt: float = 300.0) -> np.ndarray:
        """
        Simulate one timestep of fire spread.
        
        Args:
            fire_state: Current fire intensity map
            env_state: Environmental state variables
            dt: Time step in seconds
        
        Returns:
            Updated fire state
        """
        # Extract environmental variables
        wind_x = env_state.get('wind_x', np.zeros(self.grid_size))
        wind_y = env_state.get('wind_y', np.zeros(self.grid_size))
        humidity = env_state.get('humidity', np.ones(self.grid_size) * 0.5)
        slope = env_state.get('slope', np.zeros(self.grid_size))
        vegetation = env_state.get('vegetation', np.ones(self.grid_size) * 0.5)
        
        # Compute spread probability
        spread_prob = self.compute_spread_probability(
            wind_x, wind_y, humidity, slope, vegetation
        )
        
        # Fire spread kernel (neighboring cells)
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        # Convolve fire state with spread kernel
        from scipy.ndimage import convolve
        spread_influence = convolve(fire_state, kernel, mode='constant', cval=0.0)
        
        # New ignitions
        new_ignitions = spread_influence * spread_prob * dt / 3600.0  # Convert to hours
        
        # Fire decay (burnout)
        decay_rate = 0.1  # fires decay over time
        fire_decay = fire_state * decay_rate * dt / 3600.0
        
        # Update fire state
        new_fire_state = fire_state + new_ignitions - fire_decay
        new_fire_state = np.clip(new_fire_state, 0, 1)
        
        return new_fire_state


class FireSpreadTrainer:
    """Trainer for fire spread simulation model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize models
        self.neural_model = FireSpreadNet(
            input_channels=6,  # RGB + thermal + wind_x + wind_y
            hidden_dim=64,
            grid_size=tuple(config.get('grid_size', [256, 256]))
        ).to(self.device)
        
        self.physics_model = PhysicsBasedFireModel(
            grid_size=tuple(config.get('grid_size', [256, 256])),
            cell_size=config.get('cell_size', 30.0)
        )
        
        # Loss functions
        self.ignition_criterion = nn.BCELoss()
        self.intensity_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.neural_model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        logger.info(f"Initialized FireSpreadTrainer on {self.device}")
    
    def prepare_data(self, data_dir: str):
        """Prepare training and validation datasets."""
        self.train_dataset = FireSpreadDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=self.config.get('sequence_length', 5)
        )
        
        self.val_dataset = FireSpreadDataset(
            data_dir=data_dir,
            split='val',
            sequence_length=self.config.get('sequence_length', 5)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=True,
            num_workers=self.config.get('workers', 2),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            num_workers=self.config.get('workers', 2),
            pin_memory=True
        )
        
        logger.info(f"Prepared datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def create_environmental_state(self, images: torch.Tensor) -> torch.Tensor:
        """Create environmental state from satellite images."""
        # This is a simplified approach
        # In practice, you'd extract specific bands and compute derived features
        
        # Assume first 3 channels are RGB, add synthetic environmental layers
        B, T, C, H, W = images.shape
        
        # Use last image in sequence
        current_image = images[:, -1]  # (B, C, H, W)
        
        # Create synthetic environmental layers
        # Wind (simplified)
        wind_x = torch.randn(B, 1, H, W, device=images.device) * 0.1
        wind_y = torch.randn(B, 1, H, W, device=images.device) * 0.1
        
        # Humidity (from blue channel as proxy)
        if C >= 3:
            humidity = current_image[:, 2:3]  # Blue channel
        else:
            humidity = torch.ones(B, 1, H, W, device=images.device) * 0.5
        
        # Combine into environmental state
        env_state = torch.cat([current_image, wind_x, wind_y, humidity], dim=1)
        
        return env_state
    
    def train_epoch(self):
        """Train for one epoch."""
        self.neural_model.train()
        total_loss = 0
        total_ignition_loss = 0
        total_intensity_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)  # (B, T, C, H, W)
            fire_masks = batch['fire_masks'].to(self.device)  # (B, T, H, W)
            
            # Use sequence for training
            B, T, C, H, W = images.shape
            
            sequence_loss = 0
            
            for t in range(1, T):  # Predict t from t-1
                # Environmental state at time t-1
                env_state = self.create_environmental_state(images[:, :t])
                
                # Fire state at time t-1
                fire_state = fire_masks[:, t-1:t]  # (B, 1, H, W)
                
                # Target at time t
                target_fire = fire_masks[:, t]  # (B, H, W)
                
                # Predict
                ignition_prob, fire_intensity = self.neural_model(env_state, fire_state)
                
                # Losses
                # Ignition: binary classification
                target_ignition = (target_fire > 0).float().unsqueeze(1)
                ignition_loss = self.ignition_criterion(ignition_prob, target_ignition)
                
                # Intensity: regression on fire areas
                fire_mask = target_ignition.squeeze(1)
                if fire_mask.sum() > 0:
                    intensity_loss = self.intensity_criterion(
                        fire_intensity.squeeze(1)[fire_mask > 0],
                        target_fire[fire_mask > 0]
                    )
                else:
                    intensity_loss = torch.tensor(0.0, device=self.device)
                
                step_loss = ignition_loss + intensity_loss
                sequence_loss += step_loss
            
            sequence_loss = sequence_loss / (T - 1)
            
            self.optimizer.zero_grad()
            sequence_loss.backward()
            self.optimizer.step()
            
            total_loss += sequence_loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {sequence_loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.neural_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                fire_masks = batch['fire_masks'].to(self.device)
                
                B, T, C, H, W = images.shape
                sequence_loss = 0
                
                for t in range(1, T):
                    env_state = self.create_environmental_state(images[:, :t])
                    fire_state = fire_masks[:, t-1:t]
                    target_fire = fire_masks[:, t]
                    
                    ignition_prob, fire_intensity = self.neural_model(env_state, fire_state)
                    
                    target_ignition = (target_fire > 0).float().unsqueeze(1)
                    ignition_loss = self.ignition_criterion(ignition_prob, target_ignition)
                    
                    fire_mask = target_ignition.squeeze(1)
                    if fire_mask.sum() > 0:
                        intensity_loss = self.intensity_criterion(
                            fire_intensity.squeeze(1)[fire_mask > 0],
                            target_fire[fire_mask > 0]
                        )
                    else:
                        intensity_loss = torch.tensor(0.0, device=self.device)
                    
                    step_loss = ignition_loss + intensity_loss
                    sequence_loss += step_loss
                
                sequence_loss = sequence_loss / (T - 1)
                total_loss += sequence_loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs: int = 50, save_dir: str = 'models'):
        """Train the model."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                model_path = Path(save_dir) / 'fire_spread_model_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.neural_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss,
                    'config': self.config
                }, model_path)
                logger.info(f'New best model saved: {val_loss:.4f}')
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = Path(save_dir) / f'fire_spread_model_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.neural_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config
                }, checkpoint_path)
        
        logger.info(f'Training completed. Best validation loss: {best_loss:.4f}')


class FireSpreadSimulator:
    """Fire spread simulator combining neural and physics models."""
    
    def __init__(self, neural_model_path: str = None, config: Dict = None):
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.grid_size = tuple(self.config.get('grid_size', [256, 256]))
        self.cell_size = self.config.get('cell_size', 30.0)
        self.timestep = self.config.get('timestep', 300)  # seconds
        
        # Load neural model if path provided
        if neural_model_path and os.path.exists(neural_model_path):
            self.neural_model = FireSpreadNet(
                input_channels=6,
                hidden_dim=64,
                grid_size=self.grid_size
            ).to(self.device)
            
            checkpoint = torch.load(neural_model_path, map_location=self.device)
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded neural fire spread model from {neural_model_path}")
        else:
            # Create a simple neural model for testing
            self.neural_model = FireSpreadNet(
                input_channels=6,
                hidden_dim=64,
                grid_size=self.grid_size
            ).to(self.device)
            if neural_model_path:
                logger.warning(f"Neural model not found at {neural_model_path}, using default")
        
        self.neural_model.eval()
        
        # Initialize physics model
        self.physics_model = PhysicsBasedFireModel(
            grid_size=self.grid_size,
            cell_size=self.cell_size
        )
        
        logger.info("Initialized FireSpreadSimulator")
    
    def create_initial_fire_state(self, ignition_points: List[Tuple[int, int]]) -> np.ndarray:
        """Create initial fire state from ignition points."""
        fire_state = np.zeros(self.grid_size, dtype=np.float32)
        
        for y, x in ignition_points:
            if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                fire_state[y, x] = 1.0
        
        return fire_state
    
    def create_environmental_state(self, wind_speed: float = 5.0, wind_direction: float = 0.0,
                                 humidity: float = 0.4, vegetation_density: float = 0.7) -> Dict[str, np.ndarray]:
        """Create synthetic environmental state."""
        # Convert wind direction to components
        wind_x = wind_speed * np.cos(np.radians(wind_direction))
        wind_y = wind_speed * np.sin(np.radians(wind_direction))
        
        env_state = {
            'wind_x': np.full(self.grid_size, wind_x, dtype=np.float32),
            'wind_y': np.full(self.grid_size, wind_y, dtype=np.float32),
            'humidity': np.full(self.grid_size, humidity, dtype=np.float32),
            'slope': np.random.normal(0, 0.1, self.grid_size).astype(np.float32),
            'vegetation': np.full(self.grid_size, vegetation_density, dtype=np.float32)
        }
        
        return env_state
    
    def simulate(self, initial_fire_state: np.ndarray, env_state: Dict[str, np.ndarray],
                steps: int = 100, use_neural: bool = True) -> List[np.ndarray]:
        """
        Run fire spread simulation.
        
        Args:
            initial_fire_state: Initial fire intensity map
            env_state: Environmental state variables
            steps: Number of simulation steps
            use_neural: Whether to use neural model (otherwise physics-only)
        
        Returns:
            List of fire states for each timestep
        """
        fire_states = [initial_fire_state.copy()]
        current_fire_state = initial_fire_state.copy()
        
        for step in range(steps):
            if use_neural:
                # Use neural model
                fire_state_tensor = torch.tensor(current_fire_state).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Create environmental tensor
                env_tensor = torch.stack([
                    torch.tensor(env_state['wind_x']),
                    torch.tensor(env_state['wind_y']),
                    torch.tensor(env_state['humidity']),
                    torch.tensor(env_state['vegetation']),
                    torch.zeros_like(torch.tensor(env_state['humidity'])),  # Placeholder
                    torch.zeros_like(torch.tensor(env_state['humidity']))   # Placeholder
                ]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    ignition_prob, fire_intensity = self.neural_model(env_tensor, fire_state_tensor)
                    
                    # Convert to numpy
                    ignition_prob = ignition_prob.squeeze().cpu().numpy()
                    fire_intensity = fire_intensity.squeeze().cpu().numpy()
                
                # Update fire state
                # Simple update rule combining ignition probability and current state
                new_ignitions = ignition_prob * (1 - current_fire_state) * 0.1
                intensity_update = fire_intensity * current_fire_state * 0.9
                
                current_fire_state = np.clip(current_fire_state + new_ignitions, 0, 1)
                
            else:
                # Use physics model
                current_fire_state = self.physics_model.simulate_step(
                    current_fire_state, env_state, dt=self.timestep
                )
            
            fire_states.append(current_fire_state.copy())
            
            if step % 10 == 0:
                fire_area = np.sum(current_fire_state > 0.1)
                logger.info(f"Step {step}: Fire area = {fire_area} cells")
        
        return fire_states
    
    def save_simulation_results(self, fire_states: List[np.ndarray], output_dir: str):
        """Save simulation results as GeoJSON files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for t, fire_state in enumerate(fire_states):
            # Convert fire state to polygon features
            features = []
            
            # Find fire cells
            fire_cells = np.where(fire_state > 0.1)
            
            for i, (y, x) in enumerate(zip(fire_cells[0], fire_cells[1])):
                # Convert grid coordinates to geographic coordinates (simplified)
                # In practice, use proper coordinate transformation
                lon = x * self.cell_size / 111000  # Rough conversion to degrees
                lat = y * self.cell_size / 111000
                
                # Create cell polygon
                cell_size_deg = self.cell_size / 111000
                polygon_coords = [
                    [lon, lat],
                    [lon + cell_size_deg, lat],
                    [lon + cell_size_deg, lat + cell_size_deg],
                    [lon, lat + cell_size_deg],
                    [lon, lat]
                ]
                
                feature = geojson.Feature(
                    geometry=geojson.Polygon([polygon_coords]),
                    properties={
                        'fire_intensity': float(fire_state[y, x]),
                        'timestep': t,
                        'time_hours': t * self.timestep / 3600
                    }
                )
                
                features.append(feature)
            
            # Save timestep
            if features:
                feature_collection = geojson.FeatureCollection(features)
                output_path = Path(output_dir) / f'timestep_{t:03d}.geojson'
                
                with open(output_path, 'w') as f:
                    geojson.dump(feature_collection, f, indent=2)
        
        logger.info(f"Saved simulation results to {output_dir}")
    
    def visualize_simulation(self, fire_states: List[np.ndarray], save_path: str = 'fire_simulation.gif'):
        """Create animated visualization of fire spread."""
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            fire_state = fire_states[frame]
            
            im = ax.imshow(fire_state, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Fire Spread - Step {frame} (Time: {frame * self.timestep / 3600:.1f} hours)')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            
            return [im]
        
        ani = animation.FuncAnimation(
            fig, animate, frames=len(fire_states),
            interval=200, blit=False, repeat=True
        )
        
        ani.save(save_path, writer='pillow', fps=5)
        plt.close()
        
        logger.info(f"Saved simulation animation to {save_path}")


def create_sample_fire_data():
    """Create sample fire spread data for testing."""
    sample_data = {
        'ignition_points': [(50, 50), (100, 100)],
        'wind_conditions': {
            'speed': 8.0,  # m/s
            'direction': 45.0  # degrees
        },
        'environmental_conditions': {
            'humidity': 0.3,
            'vegetation_density': 0.8,
            'temperature': 35.0  # Celsius
        }
    }
    return sample_data


if __name__ == "__main__":
    # Test fire spread simulation
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['train', 'simulate'], default='simulate', help='Mode to run')
    parser.add_argument('--steps', type=int, default=50, help='Simulation steps')
    parser.add_argument('--output', default='outputs/spread_predictions', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['spread']
    
    if args.mode == 'train':
        trainer = FireSpreadTrainer(config)
        trainer.prepare_data(config['data_dir'])
        trainer.train(epochs=config.get('epochs', 50))
    
    elif args.mode == 'simulate':
        model_path = config.get('model_path', 'models/fire_spread_model_best.pt')
        simulator = FireSpreadSimulator(model_path, config)
        
        # Create sample scenario
        ignition_points = [(64, 64), (128, 128)]
        initial_fire_state = simulator.create_initial_fire_state(ignition_points)
        
        env_state = simulator.create_environmental_state(
            wind_speed=8.0,
            wind_direction=45.0,
            humidity=0.3,
            vegetation_density=0.8
        )
        
        # Run simulation
        logger.info(f"Running fire spread simulation for {args.steps} steps...")
        fire_states = simulator.simulate(
            initial_fire_state,
            env_state,
            steps=args.steps,
            use_neural=False  # Use physics model for demo
        )
        
        # Save results
        simulator.save_simulation_results(fire_states, args.output)
        simulator.visualize_simulation(fire_states, f"{args.output}/fire_spread.gif")
        
        logger.info(f"Simulation completed. Results saved to {args.output}")
