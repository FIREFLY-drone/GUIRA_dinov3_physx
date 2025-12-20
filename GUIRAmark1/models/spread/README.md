# Fire Spread Model (Hybrid Physics+NN)

## MODEL  
- **Model**: UNet/ConvLSTM hybrid with physics regularization
- **Version**: Custom architecture with spatio-temporal processing
- **Weight Path**: `models/spread/runs/<timestamp>/best.pt`
- **Input**: Raster stacks (fuel, DEM, weather, prior burns)
- **Output**: Future burn probability maps

## DATA
- **Datasets**: Raster stacks per timestep (fuel type, DEM, slope/aspect, wind u/v, temperature, humidity, precipitation, prior burn mask)
- **Local Path**: `data/processed/fire_spread/`
- **Format**: 256x256 windows, sequence T_in=6, horizon T_out=12
- **Grid Resolution**: 10-30m standardized

## TRAINING/BUILD RECIPE
- **Architecture**: UNet encoder-decoder with ConvLSTM for temporal modeling
- **Loss**: 0.5*Dice + 0.5*BCE + λ_phys*L_phys (physics regularizer)
- **Hyperparameters**:
  - Input Sequence: 6 timesteps
  - Output Horizon: 12 timesteps  
  - Epochs: 100
  - Batch Size: 8
  - Learning Rate: 1e-4
  - Physics Loss Weight: 0.1
- **Training Command**: `python train_spread.py --config config.yaml`

## EVALUATION/ACCEPTANCE
- **Metrics**: IoU@horizon>=0.65, Hausdorff distance<50m, CRPS (probabilistic)
- **Physics Validation**: Spread respects wind direction, slope effects
- **Test Script**: `python test_spread.py --run-eval`