# Vegetation Health Model (ResNet50 + VARI)

## MODEL
- **Model**: ResNet50 + VARI feature fusion
- **Version**: Pre-trained ImageNet ResNet50
- **Weight Path**: `models/vegetation/runs/<timestamp>/best.pt`
- **Classes**: healthy, stressed, burned (3-way classification)
- **Input**: Crown patches + VARI index

## DATA
- **Datasets**: DeepForest NEON canopy detections, iSAID aerial tree classes
- **Local Path**: `data/processed/vegetation_health/`
- **Format**: Crown patches with VARI computed from (G-R)/(G+R-B)
- **Sample Data**: `tests/data/vegetation/`

## TRAINING/BUILD RECIPE
- **Hyperparameters**:
  - Architecture: ResNet50 + MLP head
  - Input Size: 224x224 crown patches
  - Epochs: 35
  - Batch Size: 32
  - Learning Rate: 1e-3
  - Optimizer: Adam with weight decay
- **VARI Integration**: Concatenate pooled features + VARI → MLP(128) → 3-way softmax
- **Training Command**: `python train_vegetation.py --config config.yaml`

## EVALUATION/ACCEPTANCE
- **Metrics**: Macro-F1 >= 0.70, Per-class F1 >= 0.60
- **Test Script**: `python test_vegetation.py --run-eval`