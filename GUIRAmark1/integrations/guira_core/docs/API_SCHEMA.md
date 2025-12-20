# GUIRA Core API Schema

## Vision Processing APIs

### DINOv3 Embedding Service

**POST /embeddings/extract**
```json
{
  "image_data": "base64_encoded_image",
  "format": "jpg|png",
  "return_format": "numpy|tensor|list"
}
```

**Response:**
```json
{
  "embeddings": [...],
  "processing_time": 0.234,
  "embedding_dim": 1024,
  "status": "success"
}
```

**POST /embeddings/batch**
```json
{
  "images": [
    {
      "id": "img_001",
      "data": "base64_encoded_image"
    }
  ]
}
```

### Enhanced Detection Probes

**POST /probes/yolo/detect**
```json
{
  "image_data": "base64_encoded_image",
  "embeddings": [...],
  "confidence_threshold": 0.5,
  "classes": ["fire", "smoke"]
}
```

**Response:**
```json
{
  "detections": [
    {
      "class": "fire",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2],
      "enhanced_features": [...],
      "spatial_context": {...}
    }
  ]
}
```

**POST /probes/timesformer/analyze**
```json
{
  "frame_sequence": [
    {
      "frame_data": "base64_encoded_frame",
      "timestamp": "2024-01-01T12:00:00Z",
      "embeddings": [...]
    }
  ],
  "sequence_length": 16
}
```

## Simulation APIs

### PhysX Fire Simulator

**POST /simulation/initialize**
```json
{
  "scene_data": {
    "terrain_mesh": "path/to/terrain.obj",
    "vegetation_mesh": "path/to/vegetation.obj"
  },
  "environmental_params": {
    "wind_speed": 15.0,
    "wind_direction": 270.0,
    "temperature": 25.0,
    "humidity": 0.3
  }
}
```

**POST /simulation/run**
```json
{
  "simulation_id": "sim_12345",
  "ignition_points": [
    {"x": 100.0, "y": 200.0, "intensity": 1000.0}
  ],
  "duration": 3600.0,
  "time_step": 0.1
}
```

**Response:**
```json
{
  "simulation_id": "sim_12345",
  "status": "running",
  "estimated_completion": "2024-01-01T13:30:00Z",
  "progress": 0.15
}
```

**GET /simulation/{simulation_id}/results**
```json
{
  "simulation_id": "sim_12345",
  "status": "completed",
  "fire_perimeters": [
    {
      "timestamp": 0.0,
      "perimeter": [[x1, y1], [x2, y2], ...]
    }
  ],
  "burn_intensity": {
    "max": 5000.0,
    "average": 2500.0,
    "distribution": [...]
  },
  "export_formats": ["json", "geojson", "shapefile"]
}
```

### Mesh Preparation

**POST /meshprep/terrain**
```json
{
  "dem_data": "base64_encoded_raster",
  "resolution": 1.0,
  "coordinate_system": "EPSG:4326"
}
```

**POST /meshprep/vegetation**
```json
{
  "vegetation_map": "base64_encoded_raster",
  "fuel_load_map": "base64_encoded_raster",
  "fuel_properties": {
    "moisture_content": 0.15,
    "density": 0.8
  }
}
```

## Orchestration APIs

### Task Scheduler

**POST /orchestrator/submit**
```json
{
  "task_type": "vision_analysis|simulation|hybrid",
  "priority": "critical|high|normal|low",
  "payload": {...},
  "estimated_duration": 120.0
}
```

**Response:**
```json
{
  "task_id": "task_67890",
  "status": "queued",
  "position": 3,
  "estimated_start": "2024-01-01T12:05:00Z"
}
```

**GET /orchestrator/status/{task_id}**
```json
{
  "task_id": "task_67890",
  "status": "running|completed|failed",
  "progress": 0.67,
  "results": {...}
}
```

### Surrogate Models

**POST /surrogate/predict**
```json
{
  "model_type": "physx_fast|dinov3_light",
  "inputs": {...},
  "accuracy_mode": "fast|balanced|accurate"
}
```

## Error Responses

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Image format not supported",
    "details": {...}
  },
  "request_id": "req_12345"
}
```

## Common Headers

- **Content-Type**: application/json
- **Authorization**: Bearer {token}
- **X-Request-ID**: Unique request identifier
- **X-Processing-Node**: Processing node identifier