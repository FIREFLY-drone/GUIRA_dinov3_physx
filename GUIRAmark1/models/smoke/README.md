# Smoke Detection Model (TimeSFormer)

## MODEL
- **Model**: TimeSFormer_base_patch16_224 
- **Version**: Pre-trained on Kinetics-400
- **Base**: Vision Transformer for video understanding
- **Weight Path**: `models/smoke/runs/<timestamp>/best.pt`
- **Input**: 16 frames @ 224x224 pixels
- **Classes**: smoke (1), no_smoke (0)

## DATA
- **Datasets**: Sequences from flame_rgb, flame2_rgb_ir, wit_uas_thermal
- **Local Path**: `data/processed/smoke_timesformer/`
- **Format**: Clips of 16 frames at 8 fps, stride=2
- **Clip Length**: 16 frames
- **Sample Rate**: 8 fps
- **Sample Data**: `tests/data/smoke/sample_clips/`

## TRAINING/BUILD RECIPE
- **Hyperparameters**:
  - Clip Length: 16 frames
  - Image Size: 224x224
  - Epochs: 30
  - Batch Size: 8
  - Learning Rate: 5e-4
  - Weight Decay: 0.05
  - Optimizer: AdamW
  - Scheduler: Cosine decay with warmup (2 epochs)
- **Loss**: Focal Loss (γ=2) with class weights
- **Training Command**: `python train_smoke.py --config config.yaml --epochs 30 --batch 8`
- **GPU Memory**: ~8-12GB for batch_size=8

## EVALUATION/ACCEPTANCE
- **Metrics**:
  - AUC: >= 0.85
  - F1@0.5: >= 0.75
  - Precision: >= 0.70
  - Recall: >= 0.80
- **Aggregation**: Frame→clip aggregation with majority voting
- **Test Script**: `python test_smoke.py --run-eval`
- **Evaluation Data**: Held-out clips with temporal consistency

## TEMPORAL MODELING
- **Attention**: Self-attention over spatial-temporal patches
- **Positional Encoding**: Learned 3D positional embeddings
- **Clip Processing**: Non-overlapping 16-frame windows
- **Inference**: Sliding window with stride=8 frames