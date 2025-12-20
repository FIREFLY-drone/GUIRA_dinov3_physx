# Vision Probes Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUIRA Vision Probes                      │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────┐
                          │   Client     │
                          │ Application  │
                          └──────┬───────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                │                │                │
        ┌───────▼────────┐ ┌────▼─────────┐ ┌───▼──────────┐
        │  DINOv3 Embed  │ │  YOLO Probe  │ │  TimeSFormer │
        │    Service     │ │   :8100      │ │    Probe     │
        │     :8000      │ │              │ │    :8101     │
        └────────┬───────┘ └──────┬───────┘ └──────┬───────┘
                 │                │                 │
                 │     ┌──────────┴──────────┐      │
                 │     │                     │      │
                 ▼     ▼                     ▼      ▼
        ┌────────────────┐           ┌──────────────────┐
        │   Embedding    │           │   Detection      │
        │    Storage     │           │    Results       │
        │  (MinIO/File)  │           │   (Session DB)   │
        └────────────────┘           └──────────────────┘
```

## YOLO Probe Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      YOLO Detection Probe                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐                                              │
│  │   POST     │    Upload Image                              │
│  │  /detect   │────────────┐                                 │
│  └────────────┘            │                                 │
│                            ▼                                 │
│                   ┌────────────────┐                         │
│                   │ Image Decoder  │                         │
│                   │  (PIL/Pillow)  │                         │
│                   └────────┬───────┘                         │
│                            │                                 │
│          ┌─────────────────┴─────────────────┐              │
│          │                                   │              │
│    ┌─────▼──────┐                    ┌──────▼─────┐        │
│    │ Fast Path  │                    │Feature-    │        │
│    │ YOLO Model │                    │Fusion Path │        │
│    │ (Direct)   │                    └──────┬─────┘        │
│    └─────┬──────┘                           │              │
│          │                          ┌───────▼────────┐     │
│          │                          │Load Embeddings │     │
│          │                          │from URI        │     │
│          │                          └───────┬────────┘     │
│          │                                  │              │
│          │                          ┌───────▼────────┐     │
│          │                          │ Fusion Head    │     │
│          │                          │ MLP (768→256)  │     │
│          │                          └───────┬────────┘     │
│          │                                  │              │
│          └──────────┬───────────────────────┘              │
│                     │                                       │
│                ┌────▼──────────────┐                        │
│                │ Detection Results │                        │
│                │ (COCO-like JSON)  │                        │
│                └───────────────────┘                        │
│                     │                                       │
│                     ▼                                       │
│            ┌─────────────────────┐                          │
│            │ {detections: [...], │                          │
│            │  metadata: {...}}   │                          │
│            └─────────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

### YOLO Detection Flow

1. **Input**: RGB image (any size)
2. **Preprocessing**: Convert to numpy array, resize if needed
3. **Detection**: YOLOv8 inference with confidence filtering
4. **Post-processing**: Extract bounding boxes, classes, confidences
5. **Fusion** (optional): Enhance with DINOv3 embeddings via fusion head
6. **Output**: COCO-like JSON with xyxy coordinates

## TimeSFormer Probe Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   TimeSFormer Smoke Probe                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐    ┌──────────────────┐                     │
│  │   POST     │    │  Upload Video    │                     │
│  │  /analyze  │────┤  or Image Seq    │                     │
│  └────────────┘    └────────┬─────────┘                     │
│                              │                               │
│                  ┌───────────▼───────────┐                   │
│                  │ Video/Image Loader    │                   │
│                  │ - Extract N frames    │                   │
│                  │ - Resize to 224×224   │                   │
│                  │ - Uniform sampling    │                   │
│                  └───────────┬───────────┘                   │
│                              │                               │
│          ┌───────────────────┴───────────────────┐          │
│          │                                       │          │
│    ┌─────▼──────┐                        ┌──────▼─────┐    │
│    │ Fast Path  │                        │Feature-    │    │
│    │TimeSFormer │                        │Fusion Path │    │
│    │ Model      │                        └──────┬─────┘    │
│    └─────┬──────┘                               │          │
│          │                          ┌───────────▼────────┐ │
│          │                          │Load Frame          │ │
│          │                          │Embeddings          │ │
│          │                          │(per frame URI)     │ │
│          │                          └───────────┬────────┘ │
│          │                                      │          │
│          │                          ┌───────────▼────────┐ │
│          │                          │Temporal            │ │
│          │                          │Aggregation         │ │
│          │                          │(Average Pool)      │ │
│          │                          └───────────┬────────┘ │
│          │                                      │          │
│          └──────────┬───────────────────────────┘          │
│                     │                                       │
│              ┌──────▼────────────┐                          │
│              │ Temporal Analysis │                          │
│              │ - Smoke prob      │                          │
│              │ - Frame scores    │                          │
│              │ - Consistency     │                          │
│              └──────┬────────────┘                          │
│                     │                                       │
│                     ▼                                       │
│            ┌────────────────────┐                           │
│            │{smoke_detected:    │                           │
│            │ confidence: 0.87,  │                           │
│            │ frame_scores: [...],                           │
│            │ temporal_features} │                           │
│            └────────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

### TimeSFormer Analysis Flow

1. **Input**: Video file or image sequence
2. **Frame extraction**: Extract N frames uniformly sampled
3. **Preprocessing**: Resize to 224×224, convert to RGB
4. **Temporal analysis**: TimeSFormer processes frame sequence
5. **Scoring**: Generate per-frame smoke probabilities
6. **Aggregation** (optional): Combine with temporal embeddings
7. **Post-processing**: Calculate temporal features (consistency, peak)
8. **Output**: Smoke detection result with frame-level details

## Fusion Head Architecture (YOLO Probe)

```
┌──────────────────────────────────────────────────┐
│              Fusion Head MLP                      │
├──────────────────────────────────────────────────┤
│                                                   │
│  Input: DINOv3 Embeddings                        │
│  Shape: (num_patches, 768)                       │
│         │                                         │
│         ▼                                         │
│  ┌──────────────────┐                            │
│  │ Adaptive Avg     │                            │
│  │ Pooling          │                            │
│  └────────┬─────────┘                            │
│           │ (768,)                                │
│           ▼                                       │
│  ┌──────────────────┐                            │
│  │ Linear + ReLU    │                            │
│  │ 768 → 256        │                            │
│  │ Dropout (0.3)    │                            │
│  └────────┬─────────┘                            │
│           │                                       │
│           ▼                                       │
│  ┌──────────────────┐                            │
│  │ Linear + ReLU    │                            │
│  │ 256 → 256        │                            │
│  │ Dropout (0.3)    │                            │
│  └────────┬─────────┘                            │
│           │                                       │
│     ┌─────┴─────┐                                │
│     │           │                                 │
│     ▼           ▼                                 │
│  ┌─────┐    ┌──────┐                             │
│  │Health│    │Fire  │                             │
│  │Cls  │    │Intens│                             │
│  │3cls │    │0-1   │                             │
│  └─────┘    └──────┘                             │
│                                                   │
│  Outputs:                                         │
│  - health_label: healthy/dry/burned               │
│  - health_confidence: 0.0-1.0                     │
│  - fire_intensity: 0.0-1.0                        │
└──────────────────────────────────────────────────┘
```

## Data Flow: Complete Pipeline

```
┌────────────┐
│ Raw Image  │
│   or       │
│   Video    │
└──────┬─────┘
       │
       ▼
┌──────────────────┐
│  DINOv3 Embed    │
│  Service         │
└──────┬───────────┘
       │ embedding_uri
       │
   ┌───┴────────────────────────────┐
   │                                │
   ▼                                ▼
┌──────────────────┐    ┌───────────────────┐
│  YOLO Probe      │    │ TimeSFormer Probe │
│  (spatial)       │    │  (temporal)       │
└──────┬───────────┘    └────────┬──────────┘
       │                         │
       │ detections              │ smoke_analysis
       │                         │
       └────────┬────────────────┘
                │
                ▼
        ┌───────────────┐
        │  Session      │
        │  Indexer      │
        │  (Store)      │
        └───────────────┘
```

## Technology Stack

### YOLO Probe
- **Framework**: FastAPI 0.100+
- **ML**: PyTorch 2.0+, Ultralytics 8.3+
- **Image**: Pillow 10.0+
- **Container**: Python 3.10-slim

### TimeSFormer Probe
- **Framework**: FastAPI 0.100+
- **ML**: PyTorch 2.0+
- **Video**: OpenCV 4.8+
- **Image**: Pillow 10.0+
- **Container**: Python 3.10-slim with ffmpeg

### Fusion Head
- **Framework**: PyTorch nn.Module
- **Architecture**: 2-layer MLP
- **Activation**: ReLU
- **Regularization**: Dropout (0.3)

## Deployment Options

### Option 1: Standalone Docker Containers
```bash
docker-compose up -d yolo-probe timesformer-probe
```

### Option 2: Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolo-probe
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: yolo-probe
        image: guira-yolo-probe:latest
        ports:
        - containerPort: 8100
```

### Option 3: Serverless (with limitations)
- Suitable for low-volume inference
- Cold start latency considerations
- Model size and memory constraints

## Performance Characteristics

### YOLO Probe
| Metric | GPU (RTX 3080) | CPU (i9) |
|--------|----------------|----------|
| Latency (single image) | ~50ms | ~300ms |
| Throughput | ~200 img/s | ~30 img/s |
| Memory | 4GB VRAM | 2GB RAM |

### TimeSFormer Probe
| Metric | GPU (RTX 3090) | CPU (i9) |
|--------|----------------|----------|
| Latency (16 frames) | ~1.5s | ~8s |
| Throughput | ~40 seq/min | ~7 seq/min |
| Memory | 8GB VRAM | 4GB RAM |

## Scaling Strategies

1. **Horizontal Scaling**: Multiple probe replicas behind load balancer
2. **GPU Sharing**: NVIDIA MPS for multi-process GPU utilization
3. **Batch Processing**: Group requests for better GPU efficiency
4. **Model Optimization**: ONNX/TensorRT for faster inference
5. **Caching**: Cache embeddings and intermediate results

## Security Considerations

- **Authentication**: API key required for production
- **Rate Limiting**: Per-client request throttling
- **Input Validation**: File type and size checks
- **Temp File Cleanup**: Automatic cleanup of processing files
- **Logging**: Audit trail without sensitive data
- **Network**: TLS/SSL for encrypted communication
