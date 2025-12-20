# integrations/guira_core/vision/embed_service/app.py
"""
DINOv3 Embedding Service for GUIRA Core Integration

MODEL: facebook/dinov2-base (or facebook/dinov2-large for higher accuracy)
- Pretrained DINOv2 ViT-Base/14 distilled from ImageNet-1k
- 768-dim embeddings (base) or 1024-dim (large)
- Weights: Hugging Face transformers auto-download

DATA:
- Accepts RGB images of any size (will be tiled if > 1024x1024)
- Tiles: 518x518 with 50% overlap for large images
- Minimal dataset for probe fine-tuning: 2k-10k labeled tiles

TRAINING/BUILD RECIPE:
- No retraining required for base embeddings (frozen backbone)
- For probe fine-tuning: linear classifier on top of frozen embeddings
- Hyperparams: lr=1e-3, batch=32, epochs=10

EVAL & ACCEPTANCE:
- Embedding extraction latency: <500ms per image on GPU
- Embedding shape: (num_patches, 768) for base model
- Unit tests: test_embed.py validates response format & MinIO upload
"""

import os
import io
import uuid
import json
import logging
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
DINO_MODEL_ID = os.environ.get("DINO_MODEL_ID", "facebook/dinov2-base")

# Storage config
USE_MINIO = os.environ.get("USE_MINIO", "true").lower() == "true"
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
EMBED_BUCKET = os.environ.get("EMBED_BUCKET", "embeds")

# Tiling configuration
TILE_SIZE = 518  # DINOv2 optimal input size
MAX_IMAGE_SIZE = 1024  # Tile images larger than this
OVERLAP = 0.5  # 50% overlap for tiles

app = FastAPI(
    title="GUIRA DINOv3 Embed Service",
    description="DINOv3-based feature extraction for visual data",
    version="1.0.0"
)

# Lazy load model
_model = None
_device = None


def get_device() -> torch.device:
    """Get the compute device (GPU if available, else CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")
    return _device


def get_model():
    """Lazy load DINOv2 model and processor.
    
    Returns:
        Tuple of (processor, model)
    """
    global _model
    if _model is None:
        try:
            from transformers import AutoImageProcessor, AutoModel
            logger.info(f"Loading model: {DINO_MODEL_ID}")
            processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
            model = AutoModel.from_pretrained(DINO_MODEL_ID)
            device = get_device()
            model = model.to(device).eval()
            _model = (processor, model)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    return _model


def tile_image(img: Image.Image) -> list:
    """Tile large images into smaller patches with overlap.
    
    Args:
        img: PIL Image
        
    Returns:
        List of PIL Image tiles
    """
    width, height = img.size
    
    # If image is small enough, return as-is
    if width <= MAX_IMAGE_SIZE and height <= MAX_IMAGE_SIZE:
        return [img]
    
    tiles = []
    stride = int(TILE_SIZE * (1 - OVERLAP))
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate tile boundaries
            x_end = min(x + TILE_SIZE, width)
            y_end = min(y + TILE_SIZE, height)
            
            # Adjust start if we're at the edge
            x_start = x if x_end - x == TILE_SIZE else max(0, x_end - TILE_SIZE)
            y_start = y if y_end - y == TILE_SIZE else max(0, y_end - TILE_SIZE)
            
            # Crop tile
            tile = img.crop((x_start, y_start, x_end, y_end))
            tiles.append(tile)
            
            # If we've reached the edge, stop this row/column
            if x_end >= width:
                break
        if y_end >= height:
            break
    
    logger.info(f"Tiled {width}x{height} image into {len(tiles)} tiles")
    return tiles


def extract_embeddings(tiles: list) -> np.ndarray:
    """Extract DINOv2 embeddings from image tiles.
    
    Args:
        tiles: List of PIL Image tiles
        
    Returns:
        Numpy array of embeddings (num_tiles, num_patches, embed_dim)
    """
    processor, model = get_model()
    device = get_device()
    
    all_embeddings = []
    
    # Process tiles in batches for efficiency
    batch_size = 4
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        
        # Preprocess images
        inputs = processor(images=batch_tiles, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
        
        # Get patch embeddings (excluding CLS token)
        # last_hidden_state shape: (batch, num_patches + 1, embed_dim)
        embeddings = outputs.last_hidden_state[:, 1:, :].cpu().numpy()
        all_embeddings.append(embeddings)
    
    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Extracted embeddings with shape: {all_embeddings.shape}")
    
    return all_embeddings


def save_embedding_blob(emb_np: np.ndarray, metadata: dict = None) -> str:
    """Save embeddings to blob storage.
    
    Args:
        emb_np: Numpy array of embeddings
        metadata: Optional metadata dict to save alongside embeddings
        
    Returns:
        URI of saved embedding file
    """
    fname = f"embed_{uuid.uuid4().hex}.npz"
    path = f"/tmp/{fname}"
    
    # Save embeddings with metadata
    if metadata:
        np.savez_compressed(path, embeddings=emb_np, metadata=np.array([metadata], dtype=object))
    else:
        np.savez_compressed(path, embeddings=emb_np)
    
    if USE_MINIO:
        try:
            from minio import Minio
            # Remove http:// or https:// prefix for Minio client
            endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
            secure = MINIO_ENDPOINT.startswith("https://")
            
            client = Minio(
                endpoint,
                access_key=MINIO_ACCESS,
                secret_key=MINIO_SECRET,
                secure=secure
            )
            
            # Ensure bucket exists
            if not client.bucket_exists(EMBED_BUCKET):
                client.make_bucket(EMBED_BUCKET)
                logger.info(f"Created bucket: {EMBED_BUCKET}")
            
            # Upload file
            client.fput_object(EMBED_BUCKET, fname, path)
            logger.info(f"Uploaded embedding to MinIO: {fname}")
            
            # Clean up temp file
            os.remove(path)
            
            return f"minio://{EMBED_BUCKET}/{fname}"
        except Exception as e:
            logger.error(f"MinIO upload failed: {e}")
            # Return local path as fallback
            return f"file://{path}"
    else:
        # Azure Blob Storage implementation
        try:
            from azure.storage.blob import BlobServiceClient
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                logger.warning("Azure Storage connection string not configured")
                return f"file://{path}"
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(EMBED_BUCKET)
            
            # Create container if it doesn't exist
            try:
                container_client.create_container()
            except Exception:
                pass  # Container already exists
            
            blob_client = container_client.get_blob_client(fname)
            with open(path, "rb") as data:
                blob_client.upload_blob(data)
            
            logger.info(f"Uploaded embedding to Azure Blob: {fname}")
            os.remove(path)
            
            return f"azure://{EMBED_BUCKET}/{fname}"
        except Exception as e:
            logger.error(f"Azure Blob upload failed: {e}")
            return f"file://{path}"


class EmbedResponse(BaseModel):
    """Response model for embed endpoint."""
    embedding_uri: str
    shape: Tuple[int, int, int]
    num_tiles: int
    metadata: dict


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": DINO_MODEL_ID,
        "device": str(get_device()),
        "storage": "minio" if USE_MINIO else "azure"
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile = File(...)):
    """Extract DINOv3 embeddings from uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        EmbedResponse with embedding URI, shape, and metadata
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Read and validate image
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        original_size = img.size
        logger.info(f"Processing image: {file.filename}, size: {original_size}")
        
        # Tile if necessary
        tiles = tile_image(img)
        
        # Extract embeddings
        embeddings = extract_embeddings(tiles)
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "original_size": original_size,
            "num_tiles": len(tiles),
            "model": DINO_MODEL_ID,
            "embedding_shape": embeddings.shape
        }
        
        # Save to blob storage
        uri = save_embedding_blob(embeddings, metadata)
        
        return EmbedResponse(
            embedding_uri=uri,
            shape=embeddings.shape,
            num_tiles=len(tiles),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)