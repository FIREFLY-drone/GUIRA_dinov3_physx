# Fire Detection Model (YOLOv8)

## MODEL
- **Model**: YOLOv8 (nano/small selectable)
- **Version**: v8.0+
- **Base**: Pre-trained YOLOv8 on COCO dataset
- **Weight Path**: `models/fire/runs/<timestamp>/best.pt`
- **Classes**: fire (0), smoke (1)

## DATA
- **Datasets**: flame_rgb, flame_rgb_simplified, flame2_rgb_ir, sfgdn_fire, flame3_thermal, wit_uas_thermal
- **Local Path**: `data/processed/fire_yolo/`
- **Format**: YOLO format (images + .txt labels)
- **Input Size**: 640x640 pixels
- **Sample Data**: `tests/data/fire/sample_images/`

## TRAINING/BUILD RECIPE
- **Hyperparameters**:
  - Image Size: 640
  - Epochs: 150
  - Batch Size: 16
  - Learning Rate: 0.01
  - Final LR: 0.1 * lr0
  - Optimizer: SGD
  - Cosine Annealing: True
  - Warmup Epochs: 3
- **Augmentations**: 
  - Mosaic: 1.0 until epoch 130, then off
  - Mixup: False (for smoke stability)
  - HSV: 0.015/0.7/0.4
  - Copy-paste: 0.3 on positives
- **Training Command**: `python train_fire.py --config config.yaml --epochs 150 --batch 16`
- **GPU Memory**: ~4-6GB for batch_size=16

## EVALUATION/ACCEPTANCE
- **Metrics**:
  - mAP@50: >= 0.6 (site-dependent)
  - mAP@50-95: >= 0.4
  - Small-object AP: >= 0.3
  - Per-class mAP: fire >= 0.65, smoke >= 0.55
- **Thresholds**:
  - Confidence: 0.25
  - IoU (NMS): 0.45
- **Test Script**: `python test_fire.py --run-eval`
- **Evaluation Data**: Held-out test set with geo/site-wise splits

## FUSION (Late-Fusion Mode)
- **RGB Model**: Standard YOLOv8 trained on visible spectrum
- **Thermal Model**: YOLOv8 trained on thermal/pseudo-RGB
- **Fusion Method**: Weighted-NMS with class-aware weights
- **Weights**: w_fire=1.0, w_smoke=0.7