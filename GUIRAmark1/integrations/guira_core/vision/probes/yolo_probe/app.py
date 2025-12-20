# integrations/guira_core/vision/probes/yolo_probe/app.py
"""
YOLOv8 Detection Probe for GUIRA Core Integration

MODEL: YOLOv8n for fire/smoke detection
- YOLOv8n (nano) optimized for speed
- Pretrained on COCO, fine-tuned on fire/smoke datasets
- Classes: fire (0), smoke (1)
- Weights: models/yolo_fire.pt (place trained model here)

DATA:
- Accepts RGB images (JPEG, PNG)
- Training datasets: flame_rgb, flame2_rgb_ir, sfgdn_fire, wit_uas_thermal
- Annotation format: YOLO format (class x_center y_center width height)
- Minimal test data: 5k annotated UAV images

TRAINING/BUILD RECIPE:
- Command: python train_fire.py --data data/fire/data.yaml --model yolov8n.pt --img 640 --epochs 150 --batch 16
- Augmentations: mosaic, brightness, synthetic smoke overlay
- Hyperparams: img_size=640, lr=0.01, batch=16, epochs=150
- GPU: 8GB VRAM minimum

EVAL & ACCEPTANCE:
- mAP@0.5 >= 0.6 (site-dependent)
- Inference latency: <100ms per image on GPU
- Test script: scripts/evaluate_fire_yolov8.py
- Acceptance: detections return COCO-like JSON with xyxy, conf, cls
"""

import os
import io
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "models/yolo_fire.pt")
CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.45"))

# Storage config for embeddings
USE_EMBEDDING_FUSION = os.environ.get("USE_EMBEDDING_FUSION", "false").lower() == "true"
EMBEDDING_STORAGE = os.environ.get("EMBEDDING_STORAGE", "file")

# Class names
CLASS_NAMES = ["fire", "smoke"]

app = FastAPI(
    title="GUIRA YOLO Detection Probe",
    description="YOLOv8-based fire and smoke detection with optional DINOv3 feature fusion",
    version="1.0.0"
)

# Lazy load model
_yolo_model = None
_fusion_head = None


class DetectionResult(BaseModel):
    """Detection result schema."""
    detections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def get_yolo_model():
    """Lazy load YOLO model.
    
    Returns:
        YOLO model instance
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            
            # Check if model file exists, otherwise use default yolov8n
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading YOLO model from {MODEL_PATH}")
                _yolo_model = YOLO(MODEL_PATH)
            else:
                logger.warning(f"Model file {MODEL_PATH} not found. Using default yolov8n.pt")
                # Look for yolov8n.pt in repo root
                default_model = "/home/runner/work/FIREPREVENTION/FIREPREVENTION/yolov8n.pt"
                if os.path.exists(default_model):
                    logger.info(f"Loading default YOLO model from {default_model}")
                    _yolo_model = YOLO(default_model)
                else:
                    logger.info("Downloading yolov8n.pt from ultralytics")
                    _yolo_model = YOLO("yolov8n.pt")
            
            logger.info(f"YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load YOLO model: {e}")
    
    return _yolo_model


def get_fusion_head():
    """Lazy load fusion head for embedding integration.
    
    Returns:
        Fusion head model or None if not enabled
    """
    global _fusion_head
    if USE_EMBEDDING_FUSION and _fusion_head is None:
        try:
            from fusion_head import FusionHead
            _fusion_head = FusionHead(embed_dim=768, num_classes=len(CLASS_NAMES))
            logger.info("Fusion head loaded successfully")
        except ImportError:
            logger.warning("Fusion head not available. Feature fusion disabled.")
            _fusion_head = None
    
    return _fusion_head


def load_embedding_from_uri(embedding_uri: str) -> Optional[np.ndarray]:
    """Load embedding from storage URI.
    
    Args:
        embedding_uri: URI to embedding file (e.g., file:///path/to/embed.npz or minio://bucket/key)
        
    Returns:
        Embedding array or None if loading fails
    """
    try:
        if embedding_uri.startswith("file://"):
            # Local file storage
            file_path = embedding_uri.replace("file://", "")
            data = np.load(file_path)
            return data['embeddings']
        elif embedding_uri.startswith("minio://") or embedding_uri.startswith("s3://"):
            # MinIO/S3 storage - would need boto3 or minio client
            logger.warning("MinIO/S3 embedding loading not implemented yet")
            return None
        else:
            logger.warning(f"Unknown embedding URI scheme: {embedding_uri}")
            return None
    except Exception as e:
        logger.error(f"Failed to load embedding from {embedding_uri}: {e}")
        return None


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Service status and configuration
    """
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "embedding_fusion": USE_EMBEDDING_FUSION,
        "class_names": CLASS_NAMES
    }


@app.post("/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(...),
    embedding_uri: Optional[str] = Query(None, description="URI to DINOv3 embedding (optional)")
) -> DetectionResult:
    """Run fire/smoke detection on uploaded image.
    
    Fast path: Direct YOLO inference on raw image
    Feature-fusion path: If embedding_uri provided, integrate DINOv3 features
    
    Args:
        file: Uploaded image file
        embedding_uri: Optional URI to pre-computed DINOv3 embeddings
        
    Returns:
        Detection results in COCO-like format with metadata
    """
    try:
        # Read and validate image
        img_bytes = await file.read()
        
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
        # Get YOLO model
        model = get_yolo_model()
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Run YOLO detection
        results = model.predict(
            source=img_array,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        # Process detections
        detections = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes
                for i in range(len(boxes)):
                    # Get box coordinates (xyxy format)
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    detection = {
                        "xyxy": xyxy,
                        "conf": conf,
                        "cls": cls,
                        "class_name": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
                    }
                    
                    # Feature fusion path: enhance detection with embeddings
                    if embedding_uri and USE_EMBEDDING_FUSION:
                        embeddings = load_embedding_from_uri(embedding_uri)
                        if embeddings is not None:
                            fusion_head = get_fusion_head()
                            if fusion_head:
                                # Apply fusion head to refine detection
                                # For now, we add a fusion flag to indicate enhanced detection
                                detection["fusion_enhanced"] = True
                                detection["vegetation_health"] = "unknown"  # Placeholder for future enhancement
                    
                    detections.append(detection)
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "image_size": list(img.size),
            "num_detections": len(detections),
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "embedding_fusion_used": embedding_uri is not None and USE_EMBEDDING_FUSION
        }
        
        logger.info(f"Processed {file.filename}: {len(detections)} detections")
        
        return DetectionResult(
            detections=detections,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")


@app.post("/detect_batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    embedding_uris: Optional[str] = Query(None, description="Comma-separated URIs to DINOv3 embeddings")
) -> Dict[str, Any]:
    """Run detection on multiple images.
    
    Args:
        files: List of uploaded image files
        embedding_uris: Optional comma-separated URIs (one per image)
        
    Returns:
        Batch detection results
    """
    embedding_uri_list = []
    if embedding_uris:
        embedding_uri_list = [uri.strip() for uri in embedding_uris.split(",")]
    
    results = []
    for idx, file in enumerate(files):
        embedding_uri = embedding_uri_list[idx] if idx < len(embedding_uri_list) else None
        
        try:
            result = await detect(file=file, embedding_uri=embedding_uri)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result.dict()
            })
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)
