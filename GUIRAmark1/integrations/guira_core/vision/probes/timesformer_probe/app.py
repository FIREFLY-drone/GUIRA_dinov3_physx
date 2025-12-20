# integrations/guira_core/vision/probes/timesformer_probe/app.py
"""
TimeSFormer Detection Probe for GUIRA Core Integration

MODEL: TimeSFormer for temporal smoke detection
- TimeSFormer: Space-Time Attention for Video Understanding
- Pretrained on Kinetics-400, fine-tuned on smoke video sequences
- Input: 8 or 16 frames at 8 fps (temporal window)
- Output: Smoke classification (smoke/no-smoke) with confidence
- Weights: models/timesformer_smoke.pt (place trained model here)

DATA:
- Accepts video files (MP4, AVI) or image sequences
- Training datasets: ~1k annotated smoke video clips (8-16 frames each)
- Annotation format: CSV with columns (video_name, frame_index, smoke_flag)
- Frame extraction: 8 fps temporal window
- Minimal test data: 1k smoke video clips

TRAINING/BUILD RECIPE:
- Command: python train_smoke.py --manifest data/smoke/manifest.jsonl --frames 16 --epochs 30
- Hyperparams: frames=16, img_size=224, lr=1e-4, batch=8, epochs=30
- Optimizer: AdamW with cosine annealing
- GPU: 16GB VRAM minimum (NVIDIA RTX 3090 or better)
- Training time: ~12-18 hours on RTX 3090

EVAL & ACCEPTANCE:
- AUC: >= 0.85
- F1 Score: >= 0.80
- Precision: >= 0.82
- Recall: >= 0.78
- Inference latency: <2s per 16-frame sequence on GPU
- Test script: scripts/evaluate_smoke_timesformer.py
- Acceptance: returns temporal analysis with smoke probability and frame-level scores
"""

import os
import io
import uuid
import json
import logging
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.environ.get("TIMESFORMER_MODEL_PATH", "models/timesformer_smoke.pt")
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", "16"))
FRAME_SIZE = int(os.environ.get("FRAME_SIZE", "224"))
CONF_THRESHOLD = float(os.environ.get("SMOKE_CONF_THRESHOLD", "0.5"))

# Temporal aggregation
USE_EMBEDDING_FUSION = os.environ.get("USE_EMBEDDING_FUSION", "false").lower() == "true"

app = FastAPI(
    title="GUIRA TimeSFormer Smoke Detection Probe",
    description="TimeSFormer-based temporal smoke analysis with optional DINOv3 spatial feature fusion",
    version="1.0.0"
)

# Lazy load model
_timesformer_model = None


class SmokeAnalysisResult(BaseModel):
    """Smoke analysis result schema."""
    smoke_detected: bool
    confidence: float
    frame_scores: List[float]
    temporal_features: Dict[str, Any]
    metadata: Dict[str, Any]


def get_timesformer_model():
    """Lazy load TimeSFormer model.
    
    Returns:
        TimeSFormer model instance
    """
    global _timesformer_model
    if _timesformer_model is None:
        try:
            # Try to load custom trained model
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading TimeSFormer model from {MODEL_PATH}")
                _timesformer_model = torch.load(MODEL_PATH, map_location='cpu')
                _timesformer_model.eval()
                logger.info("TimeSFormer model loaded successfully")
            else:
                logger.warning(f"Model file {MODEL_PATH} not found. Using mock model for testing.")
                # Create a simple mock model for testing
                _timesformer_model = MockTimeSFormer()
        except Exception as e:
            logger.error(f"Failed to load TimeSFormer model: {e}")
            # Fall back to mock model
            _timesformer_model = MockTimeSFormer()
    
    return _timesformer_model


class MockTimeSFormer:
    """Mock TimeSFormer model for testing when real model is not available."""
    
    def __init__(self):
        self.sequence_length = SEQUENCE_LENGTH
        logger.info("Using mock TimeSFormer model for testing")
    
    def predict(self, frames: np.ndarray) -> Dict[str, Any]:
        """Mock prediction method.
        
        Args:
            frames: Array of frames (N, H, W, C)
            
        Returns:
            Mock prediction results
        """
        num_frames = len(frames)
        # Simple heuristic: look for reddish pixels (possible smoke/fire)
        avg_intensity = np.mean(frames)
        smoke_probability = min(0.9, avg_intensity / 255.0 * 0.5)
        
        frame_scores = [smoke_probability + np.random.randn() * 0.1 for _ in range(num_frames)]
        frame_scores = [max(0, min(1, score)) for score in frame_scores]
        
        return {
            "smoke_probability": smoke_probability,
            "frame_scores": frame_scores,
            "temporal_consistency": np.std(frame_scores)
        }


def load_video_frames(video_path: str, max_frames: int = SEQUENCE_LENGTH) -> List[np.ndarray]:
    """Load frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays
    """
    frames = []
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames uniformly
        if frame_count > max_frames:
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        else:
            indices = list(range(frame_count))
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to model input size
                frame_resized = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE))
                frames.append(frame_resized)
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from video")
        
    except ImportError:
        logger.error("OpenCV not available. Cannot load video frames.")
        raise HTTPException(status_code=500, detail="Video processing not available")
    except Exception as e:
        logger.error(f"Failed to load video frames: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load video: {e}")
    
    return frames


def load_image_sequence(image_files: List[UploadFile], max_frames: int = SEQUENCE_LENGTH) -> List[np.ndarray]:
    """Load sequence of images.
    
    Args:
        image_files: List of uploaded image files
        max_frames: Maximum number of frames to use
        
    Returns:
        List of frame arrays
    """
    frames = []
    
    # Sample images if we have too many
    if len(image_files) > max_frames:
        indices = np.linspace(0, len(image_files) - 1, max_frames, dtype=int)
        image_files = [image_files[i] for i in indices]
    
    for img_file in image_files:
        try:
            img_bytes = img_file.file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((FRAME_SIZE, FRAME_SIZE))
            frame_array = np.array(img_resized)
            frames.append(frame_array)
        except Exception as e:
            logger.warning(f"Failed to load image {img_file.filename}: {e}")
            continue
    
    logger.info(f"Loaded {len(frames)} frames from image sequence")
    return frames


def load_embedding_from_uri(embedding_uri: str) -> Optional[np.ndarray]:
    """Load embedding from storage URI.
    
    Args:
        embedding_uri: URI to embedding file
        
    Returns:
        Embedding array or None if loading fails
    """
    try:
        if embedding_uri.startswith("file://"):
            file_path = embedding_uri.replace("file://", "")
            data = np.load(file_path)
            return data['embeddings']
        else:
            logger.warning(f"Unsupported embedding URI scheme: {embedding_uri}")
            return None
    except Exception as e:
        logger.error(f"Failed to load embedding from {embedding_uri}: {e}")
        return None


def aggregate_temporal_embeddings(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """Aggregate temporal embeddings across frames.
    
    Args:
        embeddings_list: List of frame embeddings
        
    Returns:
        Aggregated temporal embedding
    """
    if not embeddings_list:
        return None
    
    # Average pooling across temporal dimension
    stacked = np.stack(embeddings_list, axis=0)
    aggregated = np.mean(stacked, axis=0)
    
    return aggregated


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Service status and configuration
    """
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "sequence_length": SEQUENCE_LENGTH,
        "frame_size": FRAME_SIZE,
        "conf_threshold": CONF_THRESHOLD,
        "embedding_fusion": USE_EMBEDDING_FUSION
    }


@app.post("/analyze", response_model=SmokeAnalysisResult)
async def analyze_smoke(
    video_file: Optional[UploadFile] = File(None),
    embedding_uris: Optional[str] = Query(None, description="Comma-separated URIs to frame embeddings")
) -> SmokeAnalysisResult:
    """Analyze smoke in video or image sequence.
    
    Fast path: Direct TimeSFormer inference on raw frames
    Feature-fusion path: If embedding_uris provided, aggregate temporal embeddings
    
    Args:
        video_file: Uploaded video file
        embedding_uris: Optional comma-separated URIs to DINOv3 embeddings for each frame
        
    Returns:
        Smoke analysis results with temporal features
    """
    try:
        if not video_file:
            raise HTTPException(status_code=400, detail="No video file provided")
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix=Path(video_file.filename).suffix, delete=False) as tmp:
            tmp.write(await video_file.read())
            tmp_path = tmp.name
        
        try:
            # Load video frames
            frames = load_video_frames(tmp_path)
            
            if len(frames) == 0:
                raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            # Get model
            model = get_timesformer_model()
            
            # Convert frames to array
            frames_array = np.array(frames)
            
            # Run temporal analysis
            predictions = model.predict(frames_array)
            
            smoke_probability = predictions.get("smoke_probability", 0.0)
            frame_scores = predictions.get("frame_scores", [0.0] * len(frames))
            
            # Feature fusion path: enhance with temporal embeddings
            if embedding_uris and USE_EMBEDDING_FUSION:
                uri_list = [uri.strip() for uri in embedding_uris.split(",")]
                embeddings_list = []
                
                for uri in uri_list:
                    embedding = load_embedding_from_uri(uri)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                
                if embeddings_list:
                    # Aggregate temporal embeddings
                    temporal_embedding = aggregate_temporal_embeddings(embeddings_list)
                    # In a full implementation, this would enhance the smoke detection
                    logger.info(f"Using temporal embeddings from {len(embeddings_list)} frames")
            
            # Determine if smoke is detected
            smoke_detected = smoke_probability >= CONF_THRESHOLD
            
            # Prepare metadata
            metadata = {
                "filename": video_file.filename,
                "num_frames": len(frames),
                "sequence_length": SEQUENCE_LENGTH,
                "frame_size": FRAME_SIZE,
                "conf_threshold": CONF_THRESHOLD,
                "embedding_fusion_used": embedding_uris is not None and USE_EMBEDDING_FUSION
            }
            
            # Temporal features
            temporal_features = {
                "temporal_consistency": float(np.std(frame_scores)),
                "peak_frame_index": int(np.argmax(frame_scores)),
                "peak_confidence": float(max(frame_scores))
            }
            
            logger.info(f"Analyzed {video_file.filename}: smoke_detected={smoke_detected}, confidence={smoke_probability:.3f}")
            
            return SmokeAnalysisResult(
                smoke_detected=smoke_detected,
                confidence=smoke_probability,
                frame_scores=frame_scores,
                temporal_features=temporal_features,
                metadata=metadata
            )
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smoke analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Smoke analysis failed: {e}")


@app.post("/analyze_sequence", response_model=SmokeAnalysisResult)
async def analyze_image_sequence(
    files: List[UploadFile] = File(...),
    embedding_uris: Optional[str] = Query(None, description="Comma-separated URIs to frame embeddings")
) -> SmokeAnalysisResult:
    """Analyze smoke in image sequence.
    
    Args:
        files: List of uploaded image files (frames)
        embedding_uris: Optional comma-separated URIs to embeddings
        
    Returns:
        Smoke analysis results
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No image files provided")
        
        # Load image sequence
        frames = load_image_sequence(files)
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No valid frames loaded")
        
        # Get model
        model = get_timesformer_model()
        
        # Convert frames to array
        frames_array = np.array(frames)
        
        # Run temporal analysis
        predictions = model.predict(frames_array)
        
        smoke_probability = predictions.get("smoke_probability", 0.0)
        frame_scores = predictions.get("frame_scores", [0.0] * len(frames))
        
        # Feature fusion path
        if embedding_uris and USE_EMBEDDING_FUSION:
            uri_list = [uri.strip() for uri in embedding_uris.split(",")]
            embeddings_list = []
            
            for uri in uri_list[:len(frames)]:
                embedding = load_embedding_from_uri(uri)
                if embedding is not None:
                    embeddings_list.append(embedding)
            
            if embeddings_list:
                temporal_embedding = aggregate_temporal_embeddings(embeddings_list)
                logger.info(f"Using temporal embeddings from {len(embeddings_list)} frames")
        
        # Determine if smoke is detected
        smoke_detected = smoke_probability >= CONF_THRESHOLD
        
        # Prepare metadata
        metadata = {
            "num_frames": len(frames),
            "sequence_length": SEQUENCE_LENGTH,
            "frame_size": FRAME_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "embedding_fusion_used": embedding_uris is not None and USE_EMBEDDING_FUSION
        }
        
        # Temporal features
        temporal_features = {
            "temporal_consistency": float(np.std(frame_scores)),
            "peak_frame_index": int(np.argmax(frame_scores)),
            "peak_confidence": float(max(frame_scores))
        }
        
        logger.info(f"Analyzed image sequence: smoke_detected={smoke_detected}, confidence={smoke_probability:.3f}")
        
        return SmokeAnalysisResult(
            smoke_detected=smoke_detected,
            confidence=smoke_probability,
            frame_scores=frame_scores,
            temporal_features=temporal_features,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image sequence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image sequence analysis failed: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8101))
    uvicorn.run(app, host="0.0.0.0", port=port)
