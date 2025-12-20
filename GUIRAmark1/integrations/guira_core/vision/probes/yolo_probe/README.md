# YOLO Detection Probe

YOLOv8-based fire and smoke detection probe for GUIRA Core Integration.

## MODEL

**Model**: YOLOv8n (nano) for fire/smoke detection
- Architecture: YOLOv8 nano variant optimized for speed
- Pretrained base: COCO dataset
- Fine-tuned on: Fire and smoke datasets
- Classes: `fire` (0), `smoke` (1)
- Weights: `models/yolo_fire.pt` (place trained model here, or use default yolov8n.pt)
- Input size: 640×640 (auto-scaled)
- Output: Bounding boxes in xyxy format with confidence scores

**Alternative models**:
- `yolov8s.pt`: Small model for better accuracy
- `yolov8m.pt`: Medium model for balanced performance
- `yolov8l.pt`: Large model for highest accuracy

## DATA

**Training datasets**:
- `flame_rgb`: RGB fire detection dataset
- `flame2_rgb_ir`: RGB+IR fire dataset
- `sfgdn_fire`: SFGDN fire detection dataset
- `wit_uas_thermal`: Thermal UAV imagery

**Annotation format**:
- YOLO format: `<class> <x_center> <y_center> <width> <height>` (normalized 0-1)
- Each image has a corresponding `.txt` file with annotations
- Dataset structure:
  ```
  data/fire/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── data.yaml
  ```

**Minimal test data**: 5k annotated UAV images with bounding boxes

## TRAINING/BUILD RECIPE

**Training command**:
```bash
python train_fire.py \
  --data data/fire/data.yaml \
  --model yolov8n.pt \
  --img 640 \
  --epochs 150 \
  --batch 16 \
  --device 0
```

**Hyperparameters**:
- Image size: 640×640
- Batch size: 16
- Learning rate: 0.01 (auto-adjusted)
- Epochs: 150
- Optimizer: AdamW
- Data augmentation: mosaic, brightness, synthetic smoke overlay

**Hardware requirements**:
- GPU: 8GB VRAM minimum (NVIDIA RTX 2060 or better)
- Training time: ~4-6 hours on RTX 3080

**Augmentations**:
- Mosaic augmentation (4-image grid)
- Random brightness/contrast
- Horizontal flip
- Random scaling and translation
- Synthetic smoke overlay (optional)

## EVAL & ACCEPTANCE

**Metrics**:
- mAP@0.5: >= 0.6 (site-dependent, may vary by environment)
- mAP@0.5:0.95: >= 0.4
- Precision: >= 0.7
- Recall: >= 0.6
- Inference latency: <100ms per image on GPU, <500ms on CPU

**Test script**:
```bash
python scripts/evaluate_fire_yolov8.py \
  --model models/yolo_fire.pt \
  --data data/fire/data.yaml
```

**Acceptance criteria**:
- Detection returns COCO-like JSON format
- Response includes: xyxy coordinates, confidence, class, class_name
- Fusion head can integrate DINOv3 embeddings (optional)
- Health check endpoint returns service status
- Batch processing supported

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service status and configuration.

### Single Image Detection
```bash
POST /detect
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- embedding_uri: Optional URI to DINOv3 embeddings
```

**Example with curl**:
```bash
curl -F "file=@sample.jpg" http://localhost:8100/detect
```

**Example with embeddings**:
```bash
curl -F "file=@sample.jpg" \
     "http://localhost:8100/detect?embedding_uri=file:///tmp/embed.npz"
```

**Response format**:
```json
{
  "detections": [
    {
      "xyxy": [100.5, 200.3, 350.7, 450.2],
      "conf": 0.85,
      "cls": 0,
      "class_name": "fire"
    }
  ],
  "metadata": {
    "filename": "sample.jpg",
    "image_size": [640, 480],
    "num_detections": 1,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "embedding_fusion_used": false
  }
}
```

### Batch Detection
```bash
POST /detect_batch
Content-Type: multipart/form-data

Parameters:
- files: Multiple image files
- embedding_uris: Optional comma-separated URIs
```

## Installation & Usage

### Local Development

1. Install dependencies:
```bash
cd integrations/guira_core/vision/probes/yolo_probe
pip install -r requirements.txt
```

2. Place model weights:
```bash
# Option 1: Use fine-tuned fire detection model
cp /path/to/yolo_fire.pt models/yolo_fire.pt

# Option 2: Use default YOLOv8n (will be auto-downloaded)
# No action needed
```

3. Run the service:
```bash
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8100
```

4. Test the service:
```bash
curl -F "file=@../../samples/sample_data/sample.jpg" \
     http://localhost:8100/detect
```

### Docker Deployment

1. Build the image:
```bash
docker build -t guira-yolo-probe .
```

2. Run the container:
```bash
docker run -p 8100:8100 \
  -v $(pwd)/models:/app/models \
  -e YOLO_MODEL_PATH=models/yolo_fire.pt \
  -e YOLO_CONF_THRESHOLD=0.25 \
  guira-yolo-probe
```

## Environment Variables

- `YOLO_MODEL_PATH`: Path to YOLO weights (default: `models/yolo_fire.pt`)
- `YOLO_CONF_THRESHOLD`: Confidence threshold for detections (default: `0.25`)
- `YOLO_IOU_THRESHOLD`: IoU threshold for NMS (default: `0.45`)
- `USE_EMBEDDING_FUSION`: Enable DINOv3 fusion (default: `false`)
- `EMBEDDING_STORAGE`: Storage backend for embeddings (default: `file`)
- `PORT`: Service port (default: `8100`)

## Feature Fusion

The probe supports two modes:

### Fast Path (Default)
Direct YOLO inference on raw images. Fast and efficient.

### Feature-Fusion Path
When `embedding_uri` is provided:
1. Load pre-computed DINOv3 embeddings
2. Pass through fusion head MLP
3. Enhance detections with vegetation health labels
4. Add fire intensity scores

**Enable fusion**:
```bash
export USE_EMBEDDING_FUSION=true
python app.py
```

**Fusion head architecture**:
- Input: DINOv3 embeddings (768-dim)
- Hidden layers: 256-dim with ReLU and Dropout
- Outputs: Health classification (healthy/dry/burned) + fire intensity (0-1)

## Security & Privacy

- **Rate limiting**: Implement rate limiting in production (use nginx or FastAPI middleware)
- **API key**: Require API key header for production deployment
- **Temp files**: Images are processed in-memory and not stored to disk
- **Logging**: Structured logging with request IDs for audit trails

## Retraining Data Checklist

For fine-tuning the YOLO probe:

- [ ] 5k+ annotated UAV images with fire/smoke bounding boxes
- [ ] Annotations in YOLO format (class x_center y_center width height)
- [ ] Train/val/test split (70/20/10)
- [ ] Dataset diversity: different times of day, weather conditions, fire sizes
- [ ] Negative samples: images without fire/smoke
- [ ] Data augmentation pipeline configured
- [ ] Validation metrics tracked (mAP, precision, recall)
- [ ] Model checkpoints saved regularly
- [ ] Final model exported and tested

## Troubleshooting

**Model not found**:
- Ensure `models/yolo_fire.pt` exists or set `YOLO_MODEL_PATH` env var
- Default yolov8n.pt will be downloaded automatically

**Low mAP scores**:
- Check dataset quality and annotations
- Increase training epochs
- Adjust confidence threshold
- Try data augmentation strategies

**Slow inference**:
- Use GPU if available (set CUDA_VISIBLE_DEVICES)
- Reduce image size
- Use yolov8n (nano) instead of larger models
- Enable half-precision inference (FP16)

**Fusion head errors**:
- Ensure `fusion_head.py` is in the same directory
- Check embedding dimensions match (768 for dinov2-base)
- Verify embedding file format (.npz with 'embeddings' key)
