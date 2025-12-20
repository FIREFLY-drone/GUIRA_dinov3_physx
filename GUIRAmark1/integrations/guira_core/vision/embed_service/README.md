# DINOv3 Embedding Service

Production-ready DINOv3 embedding microservice for GUIRA fire detection system.

## MODEL

**Model**: `facebook/dinov2-base` (default) or `facebook/dinov2-large`
- DINOv2 Vision Transformer Base/14
- Pretrained on ImageNet-1k with self-supervised learning
- Embedding dimension: 768 (base) or 1024 (large)
- Weights: Auto-downloaded from Hugging Face Hub

**Alternative models**:
- `facebook/dinov2-small`: Faster, 384-dim embeddings
- `facebook/dinov2-large`: Higher quality, 1024-dim embeddings
- `facebook/dinov2-giant`: Best quality, 1536-dim embeddings

## DATA

**Input requirements**:
- RGB images (JPEG, PNG)
- Any resolution (automatically tiled if > 1024x1024)
- Tiling strategy: 518×518 patches with 50% overlap

**For probe fine-tuning** (optional):
- Minimal dataset: 2k-10k labeled image tiles
- Format: Images + labels in COCO/YOLO format
- Store embeddings as `.npz` for fast loading

## TRAINING/BUILD RECIPE

**Base embeddings** (no training required):
- Frozen DINOv2 backbone
- Direct inference for feature extraction

**Linear probe fine-tuning** (optional):
```python
# Example probe training
from torch import nn
import torch.optim as optim

# Linear classifier on top of frozen embeddings
probe = nn.Linear(768, num_classes)
optimizer = optim.Adam(probe.parameters(), lr=1e-3)

# Training loop with frozen backbone
for epoch in range(10):
    for images, labels in dataloader:
        embeddings = extract_embeddings(images)  # Frozen
        logits = probe(embeddings.mean(dim=1))  # Pool patches
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

**Hyperparameters**:
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 10-20
- Optimizer: Adam

## EVAL & ACCEPTANCE

**Performance metrics**:
- Embedding extraction latency: <500ms per image (GPU)
- Embedding shape: `(num_tiles, num_patches, 768)` for base model
- Throughput: ~50 images/sec on A100 GPU

**Acceptance criteria**:
- `/embed` endpoint returns `embedding_uri` with 200 status
- Embeddings saved to blob storage and retrievable
- Unit tests pass with >90% coverage
- Health check endpoint responds within 100ms

## Usage

### Development Mode (PyTorch + FastAPI)

#### 1. Install dependencies

```bash
cd integrations/guira_core/vision/embed_service
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Configure environment

```bash
export DINO_MODEL_ID="facebook/dinov2-base"
export USE_MINIO="true"
export MINIO_ENDPOINT="http://localhost:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export EMBED_BUCKET="embeds"
```

Or create a `.env` file:
```env
DINO_MODEL_ID=facebook/dinov2-base
USE_MINIO=true
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
EMBED_BUCKET=embeds
```

#### 3. Start MinIO (optional for local dev)

```bash
# Using Docker
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

#### 4. Run the service

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 5. Test the service

```bash
# Health check
curl http://localhost:8000/health

# Extract embeddings
curl -F "file=@../../samples/sample_data/sample.jpg" \
  http://localhost:8000/embed

# Expected response:
# {
#   "embedding_uri": "minio://embeds/embed_abc123.npz",
#   "shape": [1, 256, 768],
#   "num_tiles": 1,
#   "metadata": {
#     "filename": "sample.jpg",
#     "original_size": [640, 480],
#     "num_tiles": 1,
#     "model": "facebook/dinov2-base"
#   }
# }
```

### Docker Deployment

#### Build image

```bash
docker build -t guira-embed-service:latest .
```

#### Run container (with GPU)

```bash
docker run --gpus all \
  -p 8000:8000 \
  -e DINO_MODEL_ID=facebook/dinov2-base \
  -e USE_MINIO=true \
  -e MINIO_ENDPOINT=http://host.docker.internal:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  guira-embed-service:latest
```

### Edge Deployment (ONNX)

#### 1. Export model to ONNX

```python
import torch
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 518, 518)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "dinov2_base.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['pixel_values'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'}
    }
)
```

#### 2. Run with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("dinov2_base.onnx")

# Run inference
inputs = {"pixel_values": image_array}
outputs = session.run(None, inputs)
embeddings = outputs[0]
```

### Production Deployment (Triton + TensorRT)

#### 1. Export to TensorRT

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("facebook/dinov2-base")
model = model.cuda().eval()

# Create TorchScript
dummy_input = torch.randn(1, 3, 518, 518).cuda()
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("dinov2_base.pt")

# Convert to TensorRT (requires torch2trt or similar)
from torch2trt import torch2trt
trt_model = torch2trt(model, [dummy_input], fp16_mode=True)
torch.save(trt_model.state_dict(), "dinov2_base_trt.pth")
```

#### 2. Create Triton model config

Create `models/dinov2/config.pbtxt`:

```protobuf
name: "dinov2_base"
platform: "tensorrt_plan"
max_batch_size: 8

input [
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [3, 518, 518]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [256, 768]
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ { name : "tensorrt" } ]
  }
}
```

#### 3. Deploy with Triton

```bash
# Start Triton server
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:23.11-py3 \
  tritonserver --model-repository=/models
```

## API Reference

### `GET /health`

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "model": "facebook/dinov2-base",
  "device": "cuda:0",
  "storage": "minio"
}
```

### `POST /embed`

Extract embeddings from uploaded image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**:
```json
{
  "embedding_uri": "minio://embeds/embed_abc123.npz",
  "shape": [4, 256, 768],
  "num_tiles": 4,
  "metadata": {
    "filename": "large_image.jpg",
    "original_size": [2048, 1536],
    "num_tiles": 4,
    "model": "facebook/dinov2-base",
    "embedding_shape": [4, 256, 768]
  }
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DINO_MODEL_ID` | `facebook/dinov2-base` | Hugging Face model ID |
| `USE_MINIO` | `true` | Use MinIO for storage (false = Azure Blob) |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `EMBED_BUCKET` | `embeds` | Storage bucket name |
| `AZURE_STORAGE_CONNECTION_STRING` | - | Azure Blob connection string (if not using MinIO) |

## Security & Privacy

**Logging**:
- Only metadata is logged (filename, dimensions, shape)
- Full image buffers are never logged
- Enable structured logging for production

**Storage**:
- Embeddings are treated as PII-equivalent
- Store credentials in Azure Key Vault or equivalent
- Enable encryption at rest for blob storage
- Set retention policy (default: 90 days)

**Consent & compliance**:
- Document user consent for drone image processing
- Implement deletion API for GDPR compliance
- Audit all embedding extractions

Example Key Vault integration:

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Fetch secrets from Key Vault
credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://myvault.vault.azure.net/", credential=credential)
MINIO_ACCESS = client.get_secret("minio-access-key").value
MINIO_SECRET = client.get_secret("minio-secret-key").value
```

## Testing

Run unit tests:

```bash
pytest tests/test_embed.py -v
```

Run with coverage:

```bash
pytest tests/test_embed.py --cov=app --cov-report=html
```

## Performance Benchmarks

| Configuration | Latency (ms) | Throughput (img/s) |
|--------------|--------------|-------------------|
| CPU (i7-10700K) | ~2000 | ~0.5 |
| GPU (RTX 3070) | ~150 | ~15 |
| GPU (A100) | ~50 | ~50 |
| TensorRT FP16 | ~30 | ~80 |

## Troubleshooting

**Model download fails**:
- Check internet connection
- Set `HF_HOME` env var to cache directory
- Pre-download model: `huggingface-cli download facebook/dinov2-base`

**Out of memory**:
- Use smaller model: `facebook/dinov2-small`
- Reduce batch size in `extract_embeddings()`
- Enable gradient checkpointing (not needed for inference)

**MinIO connection fails**:
- Verify MinIO is running: `curl http://localhost:9000/minio/health/live`
- Check credentials and bucket permissions
- Use `mc` CLI to debug: `mc alias set myminio http://localhost:9000 minioadmin minioadmin`

## License

See repository root LICENSE file.

## Contact

For issues or questions, open a GitHub issue in the THEDIFY/FIREPREVENTION repository.
