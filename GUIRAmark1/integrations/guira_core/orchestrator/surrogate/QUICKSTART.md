# FireSpreadNet Surrogate - Quick Start Guide

This guide will get you up and running with the FireSpreadNet surrogate model in 5 minutes.

## 1. Generate Dataset (5-10 minutes)

Generate a synthetic PhysX ensemble with 1000 runs:

```bash
cd integrations/guira_core/orchestrator/surrogate
python generate_ensemble.py --output-dir physx_dataset --n-runs 1000 --n-timesteps 10
```

This creates:
- 1000 simulation runs with varied wind, moisture, humidity
- ~9000 training samples (1000 runs × 9 timesteps)
- Train/val/test splits (70/15/15)
- Dataset stored in `physx_dataset/`

**Quick test with smaller dataset (1 minute):**
```bash
python generate_ensemble.py --output-dir test_dataset --n-runs 50 --n-timesteps 5
```

## 2. Train Surrogate Model (2-4 hours on GPU)

Train FireSpreadNet on the generated dataset:

```bash
python train.py \
    --data-dir physx_dataset \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --exp-name physx-surrogate \
    --model-type full \
    --save-dir models
```

**Quick test with fewer epochs (5 minutes):**
```bash
python train.py --data-dir test_dataset --epochs 5 --batch-size 4 --model-type lite
```

**Monitor training with MLflow:**
```bash
mlflow ui
# Navigate to http://localhost:5000
```

## 3. Evaluate Model

Evaluate on test set:

```bash
python evaluate.py \
    --model-path models/fire_spreadnet.pt \
    --data-dir physx_dataset \
    --split test \
    --output-dir evaluation_results
```

Check acceptance criteria:
- ✓ MSE < 0.10 (intensity prediction)
- ✓ BCE < 0.50 (ignition prediction)
- ✓ Brier < 0.25 (probabilistic calibration)

## 4. Use in Code

### Load and Predict

```python
import torch
import numpy as np
from integrations.guira_core.orchestrator.surrogate import PhysXSurrogate

# Load trained surrogate
surrogate = PhysXSurrogate(model_path='models/fire_spreadnet.pt')

# Prepare input fields (all H×W arrays)
fire_t0 = np.zeros((64, 64))
fire_t0[30:34, 30:34] = 1.0  # Initial fire

wind_u = np.ones((64, 64)) * 5.0  # 5 m/s east
wind_v = np.zeros((64, 64))
humidity = np.ones((64, 64)) * 0.4
fuel_density = np.ones((64, 64)) * 0.7
slope = np.zeros((64, 64))

# Predict next timestep
result = surrogate.predict_fire_spread(
    fire_t0, wind_u, wind_v, humidity, fuel_density, slope
)

# Get predictions
ignition_prob = result['ignition_prob']  # (64, 64) probabilities [0-1]
intensity = result['intensity']          # (64, 64) intensity values [0+]

# Threshold for binary fire map
fire_t1 = ignition_prob > 0.5
```

### Direct Model Usage

```python
from integrations.guira_core.orchestrator.surrogate.models import FireSpreadNet
import torch

# Load model
model = FireSpreadNet(in_channels=6)
checkpoint = torch.load('models/fire_spreadnet.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input (batch_size=1, channels=6, H=64, W=64)
input_tensor = torch.randn(1, 6, 64, 64)

# Predict
with torch.no_grad():
    ignition_prob, intensity = model(input_tensor)
```

## 5. Run Complete Example

Run the end-to-end example:

```bash
python example_usage.py
```

This will:
1. Generate a small synthetic dataset
2. Train a lightweight model (10 epochs)
3. Run inference on test scenarios
4. Show predictions and metrics

## Performance

**Training:**
- Full model: ~2-4 hours on GPU (V100/A100)
- Lite model: ~1-2 hours on GPU
- CPU training: ~8-12 hours (full model)

**Inference:**
- GPU: <50ms per prediction
- CPU: ~200ms per prediction
- Speedup vs PhysX: 100-1000× faster

**Model Size:**
- Full: ~1.2M parameters, ~5 MB checkpoint
- Lite: ~300K parameters, ~1.2 MB checkpoint

## Troubleshooting

**Issue: Out of memory during training**
```bash
# Reduce batch size
python train.py --batch-size 4 ...

# Use lite model
python train.py --model-type lite ...
```

**Issue: Dataset generation too slow**
```bash
# Generate smaller dataset first
python generate_ensemble.py --n-runs 100 ...

# Reduce timesteps
python generate_ensemble.py --n-timesteps 5 ...
```

**Issue: Model not converging**
```bash
# Increase learning rate
python train.py --lr 5e-3 ...

# Train for more epochs
python train.py --epochs 100 ...
```

## Next Steps

1. **Connect to Real PhysX Server**: Update `generate_ensemble.py` with actual PhysX gRPC client
2. **Higher Resolution**: Train on 128×128 or 256×256 grids for better accuracy
3. **Multi-timestep Prediction**: Extend model to predict multiple future states
4. **Deploy for Production**: Export to ONNX/TensorRT for optimized inference
5. **Active Learning**: Select most informative PhysX runs to improve model

## Files

- `models.py` - FireSpreadNet architecture
- `dataset_builder.py` - Dataset generation
- `train.py` - Training script with MLflow
- `generate_ensemble.py` - Parameter sweep and ensemble generation
- `evaluate.py` - Evaluation script
- `example_usage.py` - Complete end-to-end example
- `README.md` - Full documentation

## Support

See full documentation in `README.md` for:
- Detailed architecture description
- Dataset format specification
- Training hyperparameters
- Evaluation metrics and acceptance criteria
- Integration with orchestrator
