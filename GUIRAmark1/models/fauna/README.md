# Fauna Detection Model (YOLOv8 + CSRNet)

## MODEL
- **Detection Model**: YOLOv8 for species bounding box detection
- **Density Model**: CSRNet for robust counting
- **Weight Path**: `models/fauna/runs/<timestamp>/yolo_best.pt`, `models/fauna/runs/<timestamp>/csr_best.pt`
- **Classes**: Species from unified taxonomy (see fauna_labelmap.yaml)

## DATA
- **Datasets**: waid_fauna (primary), kaggle_fauna, awir_fauna
- **Local Path**: `data/processed/fauna_yolo/` (detection), `data/processed/fauna_csrnet/` (density)
- **Format**: YOLO format + density maps from dot annotations
- **Input Size**: Detection=960x960, Density=512x512
- **Sample Data**: `tests/data/fauna/`

## TRAINING/BUILD RECIPE
- **YOLOv8 Hyperparameters**:
  - Image Size: 960
  - Epochs: 200
  - Batch Size: 16
  - Learning Rate: 0.01
  - Box Gain: 0.04, Cls Gain: 0.7 (for small objects)
- **CSRNet Hyperparameters**:
  - Input Size: 512 crops
  - Optimizer: Adam 1e-5
  - Loss: MSE to density map
  - Curriculum: Fixed σ → adaptive
- **Training Commands**: 
  - `python train_fauna.py --config config.yaml --mode detection`
  - `python train_fauna.py --config config.yaml --mode density`

## EVALUATION/ACCEPTANCE
- **Detection Metrics**: Species mAP@50 >= 0.55 (wildlife detection challenging)
- **Density Metrics**: Count MAE <= 15%, MAPE <= 20%
- **Joint Eval**: Combined detection + counting accuracy
- **Test Script**: `python test_fauna.py --run-eval`