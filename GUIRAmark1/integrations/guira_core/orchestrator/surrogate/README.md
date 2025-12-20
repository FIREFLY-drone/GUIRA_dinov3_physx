# FireSpreadNet - Surrogate Model for PhysX Fire Spread Simulation

## MODEL

**Model**: FireSpreadNet (Encoder-decoder CNN with U-Net architecture)
**Version**: 1.0
**Weight Path**: `integrations/guira_core/orchestrator/surrogate/models/fire_spreadnet.pt`

**Architecture**:
- Encoder-decoder CNN with skip connections (U-Net style)
- 4 encoder levels with max pooling
- Bottleneck layer
- 4 decoder levels with transposed convolutions
- Dual output heads: ignition probability and fire intensity

**Input Shape**: (B, 6, H, W) where channels are:
1. fire_t0: Fire state at time t (intensity)
2. wind_u: Wind u-component (m/s)
3. wind_v: Wind v-component (m/s)
4. humidity: Relative humidity (0-1)
5. fuel_density: Fuel load density (0-1)
6. slope: Terrain slope (degrees)

**Output Shape**: 
- Ignition probability: (B, 1, H, W) - Binary probability [0-1]
- Fire intensity: (B, 1, H, W) - Fire intensity [0+]

**Model Variants**:
- `FireSpreadNet`: Full model (~1.2M parameters) for best accuracy
- `FireSpreadNetLite`: Lightweight model (~300K parameters) for fast inference

## DATA

**Datasets**: PhysX fire spread simulation ensemble with parameter sweeps

**Data Source**: Generated from PhysX fire spread simulations or synthetic data

**Local Path**: `integrations/guira_core/orchestrator/physx_dataset/`

**Dataset Format**:
- Input/target pairs stored as compressed `.npz` files
- Metadata in JSON format per split (train/val/test)
- Dataset info in `dataset_info.json`

**Sample Structure**:
```
physx_dataset/
├── samples/              # .npz files with input/target pairs
│   ├── run_0000_t000.npz
│   ├── run_0000_t001.npz
│   └── ...
├── metadata/            # Split metadata
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── full.json
└── dataset_info.json    # Dataset configuration
```

**Dataset Generation**:
```bash
python generate_ensemble.py --output-dir physx_dataset --n-runs 1000
```

**Minimum Dataset Requirements**:
- **1000 PhysX runs** for initial surrogate training
- Each run produces ~10 timesteps → ~9000 training samples
- Grid resolution: 64x64 (coarse grid for fast training)
- Train/Val/Test split: 70/15/15

**Parameter Sweep**:
- Wind speed: 0-20 m/s (log-normal distribution)
- Wind direction: 0-360 degrees (uniform)
- Fuel moisture: 0-1 (beta distribution)
- Humidity: 0.1-0.9 (normal, centered at 0.4)
- Temperature: 10-45°C (normal, centered at 25°C)

**Augmentation**:
- Scene rotation (90°, 180°, 270°)
- Wind direction variations
- Fuel density perturbations

## TRAINING/BUILD RECIPE

**Architecture**: Encoder-decoder CNN (U-Net) with dual output heads

**Loss Function**: Combined loss with three components:
- Binary Cross-Entropy (BCE) for ignition probability
- Mean Squared Error (MSE) for fire intensity
- Brier score for probabilistic calibration

**Loss Weights**:
- BCE weight: 1.0
- MSE weight: 1.0
- Brier weight: 0.5

**Hyperparameters**:
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-3
- Optimizer: Adam with weight decay 1e-5
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Device: GPU (CUDA) if available, else CPU

**Training Command**:
```bash
cd integrations/guira_core/orchestrator/surrogate
python train.py \
    --data-dir physx_dataset \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --exp-name physx-surrogate \
    --model-type full \
    --save-dir models
```

**Training with Lightweight Model**:
```bash
python train.py \
    --data-dir physx_dataset \
    --epochs 50 \
    --batch-size 16 \
    --model-type lite \
    --exp-name physx-surrogate-lite
```

**MLflow Tracking**:
- Experiment name: `physx-surrogate`
- Logged metrics: train/val loss, BCE, MSE, Brier score, learning rate
- Logged parameters: model config, training hyperparameters, dataset info
- Logged artifacts: best model, checkpoints

**View MLflow UI**:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

**Compute Requirements**:
- GPU Memory: ~4GB for batch size 8 (full model)
- GPU Memory: ~2GB for batch size 16 (lite model)
- Training Time: ~2-4 hours on single GPU (V100/A100)
- CPU Training: Possible but slower (~8-12 hours)

## EVALUATION/ACCEPTANCE

**Metrics**:

1. **MSE (Intensity)**: Mean squared error for fire intensity prediction
   - Target: MSE < 0.05 on validation set
   - Acceptable: MSE < 0.10

2. **BCE (Ignition)**: Binary cross-entropy for ignition probability
   - Target: BCE < 0.3 on validation set
   - Acceptable: BCE < 0.5

3. **Brier Score**: Probabilistic calibration metric
   - Target: Brier < 0.15
   - Acceptable: Brier < 0.25
   - Measures calibration of probability predictions

4. **Total Loss**: Combined weighted loss
   - Target: Total < 0.5
   - Logs best validation loss in MLflow

**Physics Validation**:
- Fire spreads preferentially with wind direction
- Higher fuel density increases spread rate
- Moisture and humidity inhibit spread
- Slope effects (uphill spread faster)

**Test Script**:
```bash
python -m pytest tests/unit/test_surrogate.py -v
```

**Evaluation Script**:
```bash
python evaluate.py \
    --model-path models/fire_spreadnet.pt \
    --data-dir physx_dataset \
    --split test \
    --output-dir evaluation_results
```

**Comparison with PhysX**:
- Run side-by-side comparison on validation scenarios
- Compare fire perimeters at t+1, t+5, t+10
- Calculate IoU and Hausdorff distance
- Speedup: Surrogate should be 100-1000x faster than PhysX

**Acceptance Criteria**:
✓ Surrogate achieves MSE < 0.10 vs PhysX on validation set
✓ BCE < 0.5 for ignition predictions
✓ Brier score < 0.25 for probabilistic calibration
✓ Model saved to `models/fire_spreadnet.pt`
✓ MLflow experiment logged with all metrics
✓ Inference speed: <50ms per prediction on GPU

## USAGE

**Training**:
```python
from surrogate.train import main
# Run with command line args or import and use programmatically
```

**Inference**:
```python
import torch
import numpy as np
from surrogate.models import FireSpreadNet

# Load model
model = FireSpreadNet(in_channels=6)
checkpoint = torch.load('models/fire_spreadnet.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
input_stack = np.stack([
    fire_t0,        # (H, W)
    wind_u,         # (H, W)
    wind_v,         # (H, W)
    humidity,       # (H, W)
    fuel_density,   # (H, W)
    slope          # (H, W)
], axis=0)

input_tensor = torch.from_numpy(input_stack).unsqueeze(0).float()

# Predict
with torch.no_grad():
    ignition_prob, intensity = model(input_tensor)

# Extract predictions
ignition_map = ignition_prob.squeeze().numpy()
intensity_map = intensity.squeeze().numpy()
```

**Integration with Orchestrator**:
```python
from integrations.guira_core.orchestrator.surrogate import PhysXSurrogate

surrogate = PhysXSurrogate(model_path='models/fire_spreadnet.pt')
result = surrogate.predict_fire_spread(
    ignition_point=(50, 50),
    wind_vector=(5.0, 45.0),  # speed, direction
    fuel_density=0.7
)
```

## DATA COLLECTION & LABELING CHECKLIST

- [x] Minimum dataset: 1000 PhysX runs
- [x] Each run produces ~10 timesteps → ~9000 samples
- [x] Parameter sweep: wind (speed, direction), fuel moisture
- [x] Save metadata per run: wind profile, fuel moisture, humidity, temperature
- [x] Grid resolution: 64x64 (coarse grid for efficiency)
- [x] Data format: .npz with input/target pairs
- [x] Standardization: normalize each channel appropriately
- [x] Augmentation: rotation, wind direction variations
- [x] Train/val/test split: 70/15/15
- [x] Dataset info JSON with channel names and statistics

## SECURITY

**Dataset Storage**:
- Store dataset in access-controlled directory
- Use environment variables for storage paths
- Do not commit raw datasets to git
- Use `.gitignore` to exclude `physx_dataset/`

**Model Checkpoints**:
- Encrypt model checkpoints if sensitive
- Use Azure KeyVault or similar for production
- Version control only model configs, not weights

**Data Privacy**:
- Synthetic PhysX data contains no PII
- Terrain data may be sensitive (military/critical infrastructure)
- Apply appropriate access controls per deployment

## FILES

**Core Implementation**:
- `models.py`: FireSpreadNet and FireSpreadNetLite architectures
- `dataset_builder.py`: Dataset generation from PhysX outputs
- `train.py`: Training loop with MLflow tracking
- `generate_ensemble.py`: Parameter sweep and ensemble generation
- `README.md`: This documentation

**Testing**:
- `tests/unit/test_surrogate.py`: Unit tests for surrogate components
- `tests/data/surrogate/`: Minimal sample data

**Scripts**:
- `evaluate.py`: Evaluation script (to be created)
- `export_onnx.py`: ONNX export for deployment (to be created)

## FUTURE ENHANCEMENTS

1. **Real PhysX Integration**: Connect to actual PhysX gRPC server
2. **Multi-timestep Prediction**: Predict multiple future states
3. **Uncertainty Quantification**: Ensemble or Bayesian approaches
4. **Higher Resolution**: Train on 128x128 or 256x256 grids
5. **Temporal Modeling**: Add LSTM or attention for temporal dynamics
6. **Transfer Learning**: Pre-train on large synthetic dataset
7. **Active Learning**: Select informative PhysX runs to improve model
8. **ONNX/TensorRT Export**: Optimize for production deployment

## REFERENCES

- PhysX Documentation: `integrations/guira_core/simulation/physx_server/README.md`
- Fire Spread Model: `models/spread/README.md`
- GUIRA Copilot Instructions: `COPILOT_INSTRUCTIONS.md`
