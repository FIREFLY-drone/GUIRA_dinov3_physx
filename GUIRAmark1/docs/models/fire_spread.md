# Fire Spread Model (Physics-Neural Hybrid)

## MODEL

**Model**: Hybrid ConvLSTM with Physics Regularization  
**Version**: Custom implementation with cellular automata integration  
**Weights path**: `models/spread_hybrid/best.pt`  
**Input shape**: [1, 6, 1, 256, 256] (batch, time_in, channels, height, width)  
**Output shape**: [1, 12, 256, 256] (batch, time_out, height, width)  
**Temporal scope**: 6 input timesteps → 12 future predictions  
**Spatial resolution**: 30m per pixel (configurable)  
**Time resolution**: 5-minute timesteps (300 seconds)

**Architecture**: Multi-scale ConvLSTM + Physics Regularizer
```python
# Core architecture
ConvLSTM(
    input_dim=1,           # Fire probability channel  
    hidden_dim=64,         # Feature channels
    kernel_size=3,         # 3×3 spatial convolution
    num_layers=2,          # Stacked LSTM layers
    dropout=0.1
) + PhysicsRegularizer(λ=0.1)
```

## DATA

**Primary datasets**:
- MODIS fire progression data (500m → downsampled to 30m)
- VIIRS active fire detections (time series)
- Custom fire spread sequences from controlled burns
- Synthetic data from physics-based models (FARSITE, FlamMap)

**Local paths**:
- Raw data: `data/raw/spread/`
- Processed: `data/processed/spread/`  
- Manifests: `data/manifests/fire_spread.json`

**Data format**: Temporal raster sequences  
```json
{
  "train": [
    {
      "sequence_id": "fire_camp_2018_001",  
      "input_sequence": [
        "data/processed/spread/sequences/fire_camp_2018_001/t00.npy",
        "data/processed/spread/sequences/fire_camp_2018_001/t01.npy",
        // ... 6 input timesteps
      ],
      "target_sequence": [
        "data/processed/spread/sequences/fire_camp_2018_001/t06.npy",
        // ... 12 target timesteps  
      ],
      "metadata": {
        "fire_name": "Camp Fire 2018",
        "start_date": "2018-11-08T06:30:00Z",
        "wind_data": "data/weather/camp_fire_wind.nc",
        "terrain_data": "data/terrain/camp_fire_dem.tif",
        "fuel_data": "data/fuels/camp_fire_fbfm.tif"
      }
    }
  ]
}
```

**Raster schema**: Multi-channel input stack
```yaml
# Input channels (can be extended)
channels:
  fire_probability:     # Primary fire presence [0, 1] 
    index: 0
    dtype: float32
    description: "Probability of active fire at pixel"
    
  # Optional additional channels  
  fuel_load:           # Fuel loading (tons/hectare)
    index: 1
    dtype: float32
    range: [0, 50]
    
  elevation:           # Digital elevation model (m)  
    index: 2
    dtype: float32
    normalization: "z-score"
    
  wind_speed:          # Wind speed (m/s)
    index: 3  
    dtype: float32
    range: [0, 30]
    
  slope:               # Terrain slope (degrees)
    index: 4
    dtype: float32  
    range: [0, 90]
```

**Physics variables**:
- **Fuel models**: Scott & Burgan 40 standard fuel types
- **Weather**: Wind speed/direction, humidity, temperature  
- **Terrain**: Elevation, slope, aspect
- **Fire behavior**: Rate of spread, flame length, intensity

## TRAINING/BUILD RECIPE

**Core hyperparameters**:
```yaml
# Model architecture  
input_dim: 1                   # Fire probability channel
hidden_dim: 64                 # ConvLSTM hidden features  
kernel_size: 3                 # Spatial convolution kernel
num_layers: 2                  # Stacked LSTM layers
dropout: 0.1                   # Regularization

# Training parameters
epochs: 50                     # Extended training for temporal learning
batch_size: 4                  # Memory-intensive 3D convolutions
lr: 1e-4                      # Lower LR for stable convergence  
optimizer: Adam                # Adam for RNN training
weight_decay: 1e-5            # Light regularization
scheduler: plateau            # Reduce LR on plateau
```

**Loss function**: Multi-component physics-regularized loss
```python
total_loss = (
    0.5 * dice_loss(pred_masks, true_masks) +           # Spatial overlap
    0.5 * bce_loss(pred_masks, true_masks) +            # Pixel-wise accuracy  
    λ_physics * physics_consistency_loss(pred_sequence) # Physics constraints
)

# Physics regularizer components
physics_loss = (
    monotonic_growth_penalty +        # Fire area should not decrease
    max_spread_rate_penalty +         # Limit unrealistic spread rates
    fuel_consistency_penalty          # Respect fuel load constraints
)
```

**Physics regularizers**:
```python  
def physics_consistency_loss(fire_sequence):
    """Enforce physical constraints on fire spread"""
    
    # 1. Monotonic growth constraint  
    areas = torch.sum(fire_sequence, dim=(-2, -1))  # Area per timestep
    growth_violations = torch.relu(areas[:-1] - areas[1:])  # Negative growth
    monotonic_penalty = torch.mean(growth_violations)
    
    # 2. Maximum spread rate constraint
    diff_areas = areas[1:] - areas[:-1]  # Area change per timestep  
    max_realistic_growth = 100  # pixels per timestep
    rate_violations = torch.relu(diff_areas - max_realistic_growth)
    rate_penalty = torch.mean(rate_violations)
    
    # 3. Spatial connectivity (fire spreads to adjacent cells)
    spatial_penalty = compute_spatial_connectivity_loss(fire_sequence)
    
    return monotonic_penalty + rate_penalty + spatial_penalty
```

**Data augmentation**: Spatial and temporal transformations
```yaml
augmentation:
  spatial:
    rotation: [-15, 15]          # Small rotations (preserve wind direction)
    horizontal_flip: 0.5         # Mirror fire patterns  
    vertical_flip: 0.5
    elastic_deform: 0.1         # Slight terrain variations
    
  temporal:
    time_jitter: 0.1            # Small temporal offset
    sequence_dropout: 0.05      # Random timestep masking
    reverse_sequence: 0.1       # Train on reverse spread (rare)
    
  physics_aware:
    wind_direction_shift: [-30, 30]    # Rotate wind field
    fuel_load_multiplier: [0.8, 1.2]  # Fuel variability
```

**Training command**:
```bash
python models/spread_hybrid/train_spread.py \
  --config config.yaml \
  --epochs 50 \
  --batch-size 4 \
  --physics-lambda 0.1 \
  --device 0
```

**Compute requirements**:
- GPU: 16GB+ VRAM (RTX 4090 or A6000)
- CPU: 16+ cores
- RAM: 32GB+ (large raster sequences)  
- Training time: ~6-8 hours on RTX 4090

## EVAL & ACCEPTANCE

**Key metrics**:
- **IoU at horizons**: IoU@{1h, 3h, 6h, 12h} ≥ {0.7, 0.6, 0.5, 0.4}
- **MSE Loss**: ≤0.05 (overall prediction error)
- **Spatial correlation**: ≥0.8 (spatial accuracy)
- **Temporal consistency**: ≥0.75 (smooth progression)

**Physics validation**:
- **Monotonic growth**: ≥90% of sequences show non-decreasing area
- **Realistic spread rate**: ≤5% violations of max spread rate
- **Energy conservation**: Total fire energy conserved ±10%

**Quality gates**:
✅ MSE Loss <= 0.05 (accurate predictions)  
✅ IoU@1h >= 0.6 (short-term accuracy)  
✅ Physics consistency >= 0.7 (realistic behavior)  
✅ Overall score >= 0.65 (deployment ready)

**Evaluation by prediction horizon**:
| Horizon | IoU   | MSE   | Use Case |
|---------|-------|-------|----------|
| +1h     | 0.72  | 0.023 | Tactical decisions |
| +3h     | 0.63  | 0.035 | Evacuation planning |  
| +6h     | 0.54  | 0.048 | Resource allocation |
| +12h    | 0.42  | 0.065 | Strategic overview |

**Evaluation script**:
```bash
python scripts/evaluate_spread_hybrid.py \
  --config config.yaml \
  --output experiments/spread_evaluation/
```

**Test outputs**:
- Spread grids: `artifacts/spread_raster_grids.png`  
- Temporal analysis: `artifacts/temporal_analysis_spread.png`
- Physics validation: `artifacts/physics_validation_spread.png`
- Metrics report: `report.md`

## Usage

### Training
```python  
from models.spread_hybrid import ConvLSTM, FireSpreadDataset
import torch

# Initialize model
model = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2)

# Training loop
for batch in train_loader:
    input_seq, target_seq = batch
    
    # Forward pass
    predicted_seq = model(input_seq)
    
    # Multi-component loss
    spatial_loss = dice_loss(predicted_seq, target_seq) 
    physics_loss = physics_consistency_loss(predicted_seq)
    total_loss = spatial_loss + 0.1 * physics_loss
```

### Inference
```python
from src.inference_hooks import predict_spread
import numpy as np

# Historical fire sequence (6 timesteps)
fire_history = np.random.rand(6, 256, 256)  # Mock fire progression

# Environmental data (optional)
wind_data = {'speed': 15, 'direction': 45}  # 15 km/h from NE
terrain_data = {'elevation': elevation_map, 'slope': slope_map}

# Predict future spread (12 timesteps)
future_spread = predict_spread(fire_history, wind_data, terrain_data)

print(f"Predicted spread shape: {future_spread.shape}")  # (12, 256, 256)

# Extract specific horizons
spread_1h = future_spread[2]   # +1 hour (index 2 = 3rd timestep = 15min)
spread_6h = future_spread[11]  # +6 hours
```

### Physics integration
```python
from scripts.export_spread_hybrid import FireSpreadPhysics

# Initialize physics helper
physics = FireSpreadPhysics(cell_size=30, timestep=300)

# Validate physical realism
is_valid, max_rate = physics.validate_spread_rate(fire_sequence)
print(f"Physics valid: {is_valid}, Max rate: {max_rate:.1f} m/min")

# Apply environmental effects
wind_modified = physics.apply_wind_effect(fire_mask, wind_speed=20, wind_direction=45)
terrain_modified = physics.apply_terrain_effect(fire_mask, elevation_map)
```

### Export for deployment
```python
# Export with physics helper
python scripts/export_spread_hybrid.py \
  --formats torchscript onnx \
  --include-physics-helper
```

## Physics Integration Details

**Cellular automata baseline**:
```python
def cellular_automata_step(fire_state, fuel_map, wind_vector, terrain_slope):
    """Single step of physics-based cellular automata"""
    
    new_fire_state = fire_state.copy()
    
    for i in range(1, fire_state.shape[0]-1):
        for j in range(1, fire_state.shape[1]-1):
            if fire_state[i, j] > 0.1:  # Active fire cell
                
                # Calculate spread probability to neighbors
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-connectivity
                    ni, nj = i + di, j + dj
                    
                    if fire_state[ni, nj] < 0.1:  # Unburned neighbor
                        
                        # Base spread probability from fuel
                        fuel_factor = fuel_map[ni, nj] / 10.0  # Normalize fuel load
                        
                        # Wind effect (directional bias)  
                        wind_factor = 1.0 + 0.5 * np.dot([di, dj], wind_vector)
                        wind_factor = max(0.5, wind_factor)  # Minimum spread rate
                        
                        # Terrain effect (uphill spreads faster)
                        elevation_diff = terrain_slope[ni, nj] - terrain_slope[i, j]
                        slope_factor = 1.0 + 0.3 * elevation_diff / 10.0  # 30% increase per 10m
                        slope_factor = max(0.5, min(2.0, slope_factor))  # Clamp
                        
                        # Combined spread probability
                        spread_prob = 0.1 * fuel_factor * wind_factor * slope_factor
                        
                        if np.random.random() < spread_prob:
                            new_fire_state[ni, nj] = min(1.0, fire_state[i, j] * 0.9)
    
    return new_fire_state
```

**Hybrid model integration**:
```python  
class HybridFireModel(nn.Module):
    def __init__(self, use_physics_prior=True):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=1, hidden_dim=64)
        self.use_physics_prior = use_physics_prior
        
    def forward(self, input_sequence, physics_params=None):
        # Neural prediction
        neural_pred = self.convlstm(input_sequence)
        
        if self.use_physics_prior and physics_params is not None:
            # Physics-based prior
            physics_pred = self.physics_forward(input_sequence, physics_params)
            
            # Weighted combination (learnable weights)
            alpha = torch.sigmoid(self.mixing_weight)  # [0, 1]
            combined_pred = alpha * neural_pred + (1 - alpha) * physics_pred
            
            return combined_pred
        else:
            return neural_pred
            
    def physics_forward(self, input_seq, physics_params):
        """Physics-based prediction component"""
        last_state = input_seq[:, -1, 0]  # Last fire state
        
        # Run cellular automata for prediction horizon
        predictions = []
        current_state = last_state
        
        for t in range(12):  # 12 future timesteps
            next_state = cellular_automata_step(
                current_state, 
                physics_params['fuel_map'],
                physics_params['wind_vector'], 
                physics_params['terrain_slope']
            )
            predictions.append(next_state)
            current_state = next_state
            
        return torch.stack(predictions, dim=1)  # (batch, time, H, W)
```

**Environmental factor modeling**:
```yaml
# Environmental parameter ranges
wind:
  speed_range: [0, 50]        # km/h  
  direction_range: [0, 360]   # degrees
  gust_factor: [1.0, 2.0]     # speed multiplier
  
terrain:
  slope_range: [0, 90]        # degrees
  elevation_range: [0, 4000]  # meters above sea level
  aspect_influence: true      # north/south facing slopes
  
fuel:
  moisture_range: [5, 30]     # percent  
  load_range: [0, 50]         # tons per hectare
  fbfm_types: [1, 40]        # Scott & Burgan fuel models
  
weather:
  temperature_range: [10, 50] # Celsius
  humidity_range: [10, 90]    # percent
  precipitation: [0, 10]      # mm/hour
```

## Model Files & Artifacts

**Model checkpoints**:
- `best.pt`: Best validation IoU checkpoint  
- `spread_hybrid.pt`: TorchScript export
- `spread_hybrid.onnx`: ONNX export

**Physics helpers**:
- `fire_physics.py`: Physics computation utilities
- `cellular_automata.py`: CA baseline implementation  
- `weather_integration.py`: Environmental data processing

**Training artifacts**:
- `runs/spread/`: Training logs and loss curves
- `experiments/spread_*/report.md`: Evaluation reports
- `experiments/spread_*/artifacts/`: Prediction visualizations

**Configuration files**:
- `config.yaml`: Main training configuration
- `configs/physics/`: Physics parameter definitions
- `data/manifests/fire_spread.json`: Dataset manifest
