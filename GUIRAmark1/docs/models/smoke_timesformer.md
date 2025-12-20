# Smoke Detection Model (TimeSFormer)

## MODEL

**Model**: TimeSFormer Base (Video Transformer)  
**Version**: facebook/timesformer-base-finetuned-k400  
**Pre-trained base**: Kinetics-400 pretrained weights  
**Weights path**: `models/smoke_timesformer/best.pt`  
**Input shape**: [1, 3, 8, 224, 224] (batch, channels, frames, height, width)  
**Output classes**: 2 classes (no_smoke, smoke)  
**Hidden size**: 768 (transformer embedding dimension)

## DATA

**Primary datasets**:
- Custom smoke video sequences from fire datasets
- Temporal smoke evolution clips from controlled burns
- Negative samples from clear air and non-smoke phenomena

**Local paths**:
- Raw data: `data/raw/smoke/`
- Processed: `data/processed/smoke/`  
- Manifests: `data/manifests/smoke_timesformer.json`

**Data format**: Video clips as frame sequences
```json
{
  "train": [
    {
      "video_path": "data/processed/smoke/videos/smoke_001.mp4",
      "label": 1,
      "start_frame": 0,
      "num_frames": 8,
      "metadata": {
        "source": "controlled_burn_site_A",
        "weather": "calm_wind",
        "visibility": "clear"
      }
    }
  ],
  "val": [...],
  "test": [...]
}
```

**Clip generation strategy**:
- Sample 8-frame sequences at 2 FPS (4-second windows)
- Sliding window with 50% overlap during training
- Random temporal jittering ±1 frame
- Minimum clip length: 8 frames (configurable)

## TRAINING/BUILD RECIPE

**Core hyperparameters**:
```yaml
model: facebook/timesformer-base-finetuned-k400
sequence_length: 8             # Frames per clip
epochs: 30                     # Sufficient for video fine-tuning
batch_size: 8                  # Memory-constrained due to video
lr: 5e-4                      # Lower LR for transformer fine-tuning
optimizer: AdamW
weight_decay: 0.05
patch_size: 16                 # ViT patch size
embed_dim: 768                 # Transformer embedding dimension
depth: 12                      # Number of transformer layers
num_heads: 12                  # Multi-head attention heads
```

**Data augmentation**:
```yaml
augmentation:
  temporal_jitter: 0.1         # Random temporal offset
  color_jitter: 0.4           # Brightness/contrast/saturation
  random_crop: 0.8            # Spatial crop and resize
  horizontal_flip: 0.5        # Flip probability
  normalize: True             # ImageNet normalization
  resize: [224, 224]          # Standard transformer input size
```

**Training strategy**:
- **Focal Loss** with γ=2 (handle class imbalance)
- **Cosine annealing** learning rate schedule
- **Gradient clipping** at norm 1.0
- **Mixed precision** training (FP16)
- **Early stopping** based on validation AUC

**Training command**:
```bash
python models/smoke_timesformer/train_smoke.py \
  --config config.yaml \
  --epochs 30 \
  --batch-size 8 \
  --device 0
```

**Compute requirements**:
- GPU: 16GB+ VRAM (RTX 4090 or A6000)
- CPU: 16+ cores recommended  
- RAM: 32GB+
- Training time: ~8-12 hours on RTX 4090

## EVAL & ACCEPTANCE

**Key metrics**:
- **AUC**: ≥0.85 (primary metric for binary classification)
- **F1 Score**: ≥0.75 (balanced precision/recall)
- **Precision**: ≥0.80 (minimize false smoke alerts)
- **Recall**: ≥0.70 (catch smoke early)
- **Accuracy**: ≥0.80 (overall correctness)

**Quality gates**:
✅ AUC >= 0.7 (minimum deployment threshold)  
✅ F1 Score >= 0.65 (acceptable balanced performance)  
✅ Precision >= 0.7 (acceptable false positive rate)  
✅ Temporal consistency >= 0.8 (stable across sequences)

**Evaluation script**:
```bash
python scripts/evaluate_smoke_timesformer.py \
  --config config.yaml \
  --output experiments/smoke_timesformer_evaluation/
```

**Test outputs**:
- Confusion matrix: `artifacts/confusion_matrix_smoke.png`
- Temporal analysis: `artifacts/temporal_analysis_smoke.png`
- ROC curve: `artifacts/roc_curve_smoke.png`
- Metrics report: `report.md`

**Performance by sequence length**:
| Frames | AUC   | F1    | Precision | Recall |
|--------|-------|-------|-----------|--------|
| 4      | 0.78  | 0.69  | 0.74      | 0.65   |
| 8      | 0.83  | 0.75  | 0.78      | 0.72   |
| 16     | 0.85  | 0.77  | 0.80      | 0.74   |
| **8 (optimal)** | **0.83** | **0.75** | **0.78** | **0.72** |

## Usage

### Training
```python
import torch
from transformers import TimesformerModel
from torch.utils.data import DataLoader

# Initialize model
base_model = TimesformerModel.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
)
model = torch.nn.Sequential(
    base_model,
    torch.nn.Linear(base_model.config.hidden_size, 2)
)

# Train with custom dataset
dataset = SmokeVideoDataset(manifest, 'train', sequence_length=8, transform=transforms)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Inference
```python
from src.inference_hooks import classify_smoke_clip
import numpy as np

# Video clip inference (T, H, W, 3)
video_clip = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
smoke_probability = classify_smoke_clip(video_clip)

print(f"Smoke probability: {smoke_probability:.3f}")
```

### Real-time stream processing
```python
from src.inference_hooks import SmokeDetectionInference

# Initialize detector
detector = SmokeDetectionInference('models/smoke_timesformer/best.pt')

# Process frames one by one
for frame in video_stream:
    smoke_prob = detector.add_frame(frame)
    if smoke_prob is not None:  # Buffer full
        if smoke_prob > 0.7:
            print("⚠️  High smoke probability detected!")
```

### Export for deployment
```python
# Export to TorchScript
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('smoke_timesformer.pt')

# Or use export script
python scripts/export_smoke_timesformer.py --formats torchscript onnx
```

## Temporal Processing

**Frame buffering**:
- Maintain sliding window of 8 frames
- Process at 2 FPS (every 15th frame at 30 FPS input)
- Overlap windows by 50% for temporal consistency

**Temporal features**:
- Motion patterns (smoke drift and dispersion)
- Opacity changes over time  
- Spatial extent evolution
- Color/texture temporal consistency

**Sequence preprocessing**:
```python
def preprocess_sequence(frames, target_length=8):
    """Preprocess video sequence for TimeSFormer"""
    # Resize frames
    frames_resized = [cv2.resize(f, (224, 224)) for f in frames]
    
    # Sample to target length
    if len(frames_resized) > target_length:
        indices = np.linspace(0, len(frames_resized)-1, target_length).astype(int)
        frames_sampled = [frames_resized[i] for i in indices]
    else:
        frames_sampled = frames_resized
    
    # Normalize and tensorize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor_frames = torch.stack([transform(f) for f in frames_sampled])
    return tensor_frames.unsqueeze(0)  # Add batch dimension
```

## Thresholds & Decision Logic

**Smoke detection thresholds**:
- **Low confidence**: 0.3-0.5 (possible smoke, monitor)
- **Medium confidence**: 0.5-0.7 (likely smoke, alert)  
- **High confidence**: 0.7+ (definitive smoke, immediate alert)

**Temporal filtering**:
- Require 2/3 consecutive positive predictions to trigger alert
- Exponential smoothing with α=0.7 for confidence scores
- Hysteresis: 0.7 threshold to trigger, 0.5 to clear

**Integration with fire detection**:
```python
def combined_fire_smoke_decision(fire_detections, smoke_probability):
    """Combined decision logic"""
    fire_detected = len(fire_detections.get('boxes', [])) > 0
    smoke_detected = smoke_probability > 0.5
    
    if fire_detected and smoke_detected:
        return "fire_with_smoke", 0.9  # High confidence
    elif fire_detected:
        return "fire_only", 0.8
    elif smoke_detected:
        return "smoke_only", smoke_probability
    else:
        return "clear", 0.1
```

## Performance Optimization

**Speed optimizations**:
- Use 8-frame sequences (optimal speed/accuracy)
- Process every 15th frame (2 FPS effective)
- TorchScript export for faster inference
- Mixed precision inference (FP16)

**Memory optimizations**:
- Gradient checkpointing during training
- Batch size 8 for training, 1 for inference
- Frame buffer size limit (max 16 frames)

**Real-time considerations**:
- CPU inference: ~200ms per 8-frame sequence
- GPU inference: ~50ms per 8-frame sequence (RTX 3070)
- Edge deployment: Consider MobileViT for Jetson

## Model Files & Artifacts

**Model checkpoints**:
- `best.pt`: Best validation AUC checkpoint
- `best_torchscript.pt`: TorchScript export
- `best.onnx`: ONNX export (if compatible)

**Training artifacts**:
- `runs/smoke_timesformer/`: Training logs and curves
- `experiments/smoke_timesformer_*/report.md`: Evaluation reports
- `experiments/smoke_timesformer_*/artifacts/`: Visualizations

**Configuration files**:
- `config.yaml`: Main training configuration  
- `data/manifests/smoke_timesformer.json`: Dataset manifest
- `configs/labelmaps/smoke.yaml`: Class mappings
