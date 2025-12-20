# Fauna Detection and Counting Models (YOLOv8 + CSRNet)

## MODEL

**Architecture**: Dual-component system  
**Component 1**: YOLOv8n for individual animal detection  
**Component 2**: CSRNet for density estimation and crowd counting  
**Weights paths**:
- YOLO: `models/fauna_yolov8_csrnet/yolo_best.pt`
- CSRNet: `models/fauna_yolov8_csrnet/csrnet_best.pth`

**Input shapes**:
- YOLO: [1, 3, 960, 960] (higher resolution for small animals)
- CSRNet: [1, 3, 512, 512] (density estimation)

**Output classes (YOLO)**: 5 species classes + health status
- Species: deer, elk, bear, bird, other
- Health: healthy, distressed

**Output (CSRNet)**: Density map [1, 1, 64, 64] (downsampled by 8×)

## DATA

**Primary datasets**:
- `waid_fauna`: WAID (~14k drone images, 6 species)  
- `kaggle_fauna`: Kaggle Wildlife Aerial Imagery
- `awir_fauna`: AWIR community-curated repository

**Local paths**:
- Raw data: `data/raw/fauna/`
- Processed: `data/processed/fauna/`
- Manifests: 
  - `data/manifests/fauna_yolov8.yaml` (YOLO format)
  - `data/manifests/fauna_csrnet.json` (density format)

**YOLO data format**: Extended YOLO with health status
```yaml
# fauna_yolov8.yaml
path: /workspaces/FIREPREVENTION/data/processed/fauna
train: images/train
val: images/val  
test: images/test
names:
  0: deer
  1: elk
  2: bear
  3: bird
  4: other
nc: 5

# Health status in separate annotations
health_names:
  0: healthy
  1: distressed
```

**CSRNet data format**: Point annotations → density maps
```json
{
  "train": [
    {
      "image_path": "data/processed/fauna/images/train/deer_herd_001.jpg",
      "points": [[245, 167], [389, 203], [156, 298]],  // Animal centers
      "species_counts": {"deer": 12, "elk": 0, "bear": 0},
      "total_count": 12,
      "density_map_path": "data/processed/fauna/density_maps/train/deer_herd_001.npy"
    }
  ]
}
```

**Taxonomy mapping**:
```yaml
# configs/labelmaps/fauna.yaml
species_hierarchy:
  ungulates:
    - deer
    - elk
  carnivores:
    - bear
  avian:
    - bird
  other:
    - other

health_indicators:
  healthy:
    - normal_posture
    - active_movement  
    - group_behavior
  distressed:
    - isolation
    - abnormal_posture
    - injury_visible
```

## TRAINING/BUILD RECIPE

### YOLOv8 Component

**Hyperparameters**:
```yaml
# YOLO training config
yolo_model: yolov8n.pt
epochs: 200                    # Extended for wildlife detection
batch_size: 12                 # Smaller due to higher resolution
img_size: 960                  # Higher for small animal detection
lr: 1e-3
optimizer: SGD
mosaic: 1.0                   # Strong augmentation
hsv_h: 0.015
hsv_s: 0.7  
hsv_v: 0.4
mixup: 0.1                    # Animal mixing augmentation
copy_paste: 0.1               # Copy-paste augmentation
```

**Multi-task training**: Joint detection + health classification
```python
# Custom loss function
total_loss = (
    yolo_detection_loss + 
    0.5 * health_classification_loss +
    0.1 * species_consistency_loss
)
```

### CSRNet Component

**Architecture**: VGG16 frontend + dilated convolution backend
```python
class CSRNet(nn.Module):
    def __init__(self):
        # Frontend: VGG16 layers 1-10
        self.frontend = make_layers([64,64,'M',128,128,'M',256,256,256,'M',512,512,512])
        
        # Backend: dilated convolutions  
        self.backend = make_layers([512,512,512,256,128,64], dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
```

**Training strategy**:
```yaml
csrnet_config:
  epochs: 100
  batch_size: 8               # Memory intensive  
  lr: 1e-5                   # Lower LR for density estimation
  optimizer: Adam
  loss: MSELoss              # Regression loss for density
  gaussian_sigma: 15         # For density map generation
  downsample_factor: 8       # Output downsampling
```

**Joint training commands**:
```bash
# Stage 1: Train YOLO detector
python models/fauna_yolov8_csrnet/train_yolo.py \
  --config config.yaml \
  --epochs 200 \
  --img-size 960

# Stage 2: Train CSRNet density estimator  
python models/fauna_yolov8_csrnet/train_csrnet.py \
  --config config.yaml \
  --epochs 100

# Stage 3: Joint fine-tuning (optional)
python models/fauna_yolov8_csrnet/train_joint.py \
  --config config.yaml \
  --epochs 50
```

**Compute requirements**:
- GPU: 24GB+ VRAM (RTX 4090 or A6000) for joint training
- CPU: 16+ cores
- RAM: 64GB+ (large image processing)
- Training time: ~12-16 hours total

## EVAL & ACCEPTANCE

### YOLOv8 Detection Metrics

**Key metrics**:
- **mAP@0.5**: ≥0.70 (wildlife detection is challenging)
- **mAP@0.5:0.95**: ≥0.60  
- **Per-species mAP**: All species ≥0.60
- **Health classification accuracy**: ≥0.75

**Quality gates**:
✅ YOLOv8 mAP@0.5 >= 0.65 (minimum deployment)  
✅ All species recall >= 0.60 (no species left behind)  
✅ Health classification F1 >= 0.70 (reliable health assessment)

### CSRNet Density Metrics

**Key metrics**:
- **MAE**: <10% of average ground truth count
- **RMSE**: <15% of average ground truth count  
- **Accuracy within 10%**: ≥0.70 (percentage of predictions within 10%)

**Quality gates**:
✅ CSRNet MAE < 10% of avg count (accurate counting)  
✅ RMSE < 15% of avg count (consistent performance)  
✅ Correlation coefficient >= 0.85 (strong linear relationship)

### Combined Performance

**Overall score**: Weighted combination
```python
combined_score = (
    0.6 * yolo_map50 + 
    0.4 * (1 - csrnet_mae_normalized)
)
```

**Quality gates**:
✅ Combined score >= 0.7 (deployment ready)  
✅ Cross-validation consistency >= 0.8 (robust across sites)

**Evaluation script**:
```bash
python scripts/evaluate_fauna_yolo_csrnet.py \
  --config config.yaml \
  --output experiments/fauna_evaluation/
```

**Test outputs**:
- Species PR curves: `artifacts/pr_curves_fauna.png`
- Density analysis: `artifacts/density_analysis_fauna.png`  
- Joint evaluation: `artifacts/joint_evaluation_fauna.png`
- Combined report: `report.md`

## Usage

### Detection + Counting Inference
```python
from src.inference_hooks import detect_fauna
import numpy as np

# Single image inference
image = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)
detections, density_map = detect_fauna(image)

print(f"Individual detections: {len(detections['boxes'])}")
print(f"Total density count: {density_map.sum():.1f}")

# Per-species breakdown
for i, (box, species, health) in enumerate(zip(
    detections['boxes'], 
    detections['species'], 
    detections['health_status']
)):
    x1, y1, x2, y2 = box
    print(f"Animal {i+1}: {species} ({health}) at [{x1:.0f}, {y1:.0f}]")
```

### Training Custom Species
```python
# Extend taxonomy for new species
new_species_config = {
    'names': {
        0: 'deer', 1: 'elk', 2: 'bear', 3: 'bird', 4: 'moose', 5: 'other'
    },
    'nc': 6
}

# Retrain with new data
model = YOLO('fauna_yolo_base.pt')
model.train(data='fauna_extended.yaml', epochs=100)
```

### Export for Production
```python
# Export both components
python scripts/export_fauna_yolo_csrnet.py \
  --formats onnx torchscript \
  --output production_models/
```

## Joint Evaluation & Fusion

**Decision fusion strategies**:
1. **Consensus**: Both models agree (high confidence)
2. **Detection priority**: YOLO detections + CSRNet for groups
3. **Density priority**: CSRNet count + YOLO for species ID

**Spatial fusion**:
```python
def spatial_fusion(yolo_boxes, density_map, threshold=0.5):
    """Combine detection boxes with density hotspots"""
    # Find density peaks
    peaks = find_peaks_2d(density_map, threshold=threshold)
    
    # Match boxes to peaks
    matched_detections = []
    for box in yolo_boxes:
        box_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        closest_peak = find_closest_peak(box_center, peaks)
        
        matched_detections.append({
            'box': box,
            'density_support': density_map[closest_peak],
            'confidence_boost': 0.1 if density_map[closest_peak] > 0.3 else 0
        })
    
    return matched_detections
```

**Count reconciliation**:
```python
def reconcile_counts(yolo_count, csrnet_count, confidence_weights):
    """Reconcile individual detection count with density estimate"""
    if abs(yolo_count - csrnet_count) <= 2:
        # Close agreement - use weighted average
        return (
            confidence_weights['yolo'] * yolo_count + 
            confidence_weights['csrnet'] * csrnet_count
        ) / sum(confidence_weights.values())
    elif yolo_count > csrnet_count:
        # Individual detection found more - trust YOLO
        return yolo_count
    else:
        # Density estimation higher - possible crowding/occlusion
        return csrnet_count
```

## Performance Optimization

**Multi-scale detection**:
- Use different input sizes for different scenarios
- Small herds: 640×640 (faster)
- Large groups: 960×960 (more accurate)
- Aerial surveys: 1280×1280 (maximum detail)

**Speed optimizations**:
- YOLO TensorRT: ~30ms per image (RTX 3070)  
- CSRNet optimized: ~50ms per image
- Combined pipeline: ~80ms per image

**Memory optimizations**:
- Sliding window for large images
- Batch processing for video streams
- Gradient checkpointing during training

## Model Files & Artifacts

**Model checkpoints**:
- `yolo_best.pt`: Best YOLO detection model
- `csrnet_best.pth`: Best CSRNet density model
- `yolo_fauna.onnx` / `csrnet_fauna.onnx`: Production exports

**Training artifacts**:
- `runs/fauna_yolo/`: YOLO training logs
- `runs/fauna_csrnet/`: CSRNet training logs  
- `experiments/fauna_*/report.md`: Joint evaluation reports

**Configuration files**:
- `config.yaml`: Main training configuration
- `configs/labelmaps/fauna.yaml`: Species taxonomy
- `data/manifests/fauna_*.yaml`: Dataset manifests
