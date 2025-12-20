# GUIRA Quick Reference

## 🚀 Essential Commands

### Local Development Startup
```bash
# Start infrastructure
cd integrations/guira_core/infra && ./local_start.sh

# Start backend API
cd integrations/guira_core/orchestrator/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Start DINO service
cd integrations/guira_core/vision/embed_service
uvicorn app:app --host 0.0.0.0 --port 8002 --reload

# Start frontend
cd integrations/guira_core/frontend
npm run dev
```

---

## 📖 Documentation Map

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[README.md](./README.md)** | Project overview & quick start | First time setup |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | System architecture with diagrams | Understanding system design |
| **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** | Step-by-step setup guides | Setting up PhysX/DINO |
| **[COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md)** | Detailed API docs | Developing integrations |
| **[docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)** | Deployment instructions | Production deployment |
| **[Functionality_PhysX_DINOv3.md](./Functionality_PhysX_DINOv3.md)** | Advanced technical specs | Deep technical understanding |

---

## 🔥 PhysX Fire Spread - Quick Reference

### Train Surrogate Model
```bash
cd integrations/guira_core/orchestrator/surrogate

# Generate dataset
python generate_ensemble.py --num-runs 1000 --use-mock

# Train model
python train.py --data-dir physx_dataset --epochs 50

# Evaluate
python evaluate.py --model-path models/fire_spreadnet.pt
```

### Use Surrogate for Prediction
```python
from integrations.guira_core.orchestrator.surrogate import PhysXSurrogate

surrogate = PhysXSurrogate(model_path="models/fire_spreadnet.pt")
prediction = surrogate.predict_fire_spread(
    fire_t0=fire_mask,
    wind_u=wind_u_grid,
    wind_v=wind_v_grid,
    humidity=humidity_grid,
    fuel=fuel_grid,
    slope=slope_grid
)
```

### API Endpoint
```bash
curl -X POST http://localhost:8000/api/surrogate/predict \
  -H "Content-Type: application/json" \
  -d @fire_scenario.json
```

**Performance:**
- Inference: 30-50ms on GPU
- Accuracy: ~95% vs PhysX
- Speedup: 100-1000×

---

## 🔷 DINO v3 Embeddings - Quick Reference

### Extract Embeddings via API
```bash
# Single image
curl -X POST http://localhost:8002/embed \
  -F "file=@image.jpg" \
  -F "store=true" \
  -F "metadata={\"session_id\": \"abc123\"}"

# Response
{
  "embedding_id": "uuid",
  "shape": [768],
  "blob_url": "s3://embeds/uuid.npy",
  "indexed": true
}
```

### Python SDK
```python
from embed_service.app import DINOEmbedder
from PIL import Image

embedder = DINOEmbedder(model_id="facebook/dinov2-base")
result = embedder.embed_image(
    image=Image.open("fire.jpg"),
    store=True,
    metadata={"session_id": "session_123"}
)
```

### RAG Query
```python
from rag_pipeline import query_rag

answer = await query_rag(
    "Show me all images with active fires near water sources",
    top_k=5
)
```

**Performance:**
- Embedding extraction: <500ms on GPU
- Dimension: 768 (base) / 1024 (large)
- Tiling: 518×518 patches with 50% overlap

---

## 🌐 API Endpoints

### Backend API (Port 8000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/surrogate/predict` | POST | Fire spread prediction |
| `/api/ingest/detection` | POST | Ingest detection event |
| `/api/query` | POST | RAG-based query |
| `/api/map/fires` | GET | GeoJSON fire locations |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |

### DINO Service (Port 8002)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/embed` | POST | Extract embedding |
| `/embed/{id}` | GET | Retrieve embedding |
| `/embed/batch` | POST | Batch embedding |
| `/health` | GET | Health check |

### YOLO Probe (Port 8003)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/detect` | POST | Fire/smoke detection |
| `/health` | GET | Health check |

---

## 🗄️ Data Schemas

### Fire Detection Event
```json
{
  "event_id": "uuid",
  "session_id": "string",
  "timestamp": "ISO8601",
  "event_type": "fire",
  "location": {"lat": 40.7128, "lon": -74.0060, "alt": 100},
  "detection": {
    "class": "fire",
    "confidence": 0.95,
    "bbox": [x1, y1, x2, y2]
  },
  "metadata": {}
}
```

### Fire Spread Prediction Request
```json
{
  "fire_t0": [[...]],      // (H, W) current fire state
  "wind_u": [[...]],       // (H, W) wind u-component (m/s)
  "wind_v": [[...]],       // (H, W) wind v-component (m/s)
  "humidity": [[...]],     // (H, W) relative humidity [0-1]
  "fuel": [[...]],         // (H, W) fuel density [0-1]
  "slope": [[...]]         // (H, W) slope (degrees)
}
```

### Embedding Response
```json
{
  "embedding_id": "uuid",
  "shape": [768],
  "num_tiles": 1,
  "blob_url": "s3://embeds/uuid.npy",
  "indexed": true
}
```

---

## 🔧 Environment Variables

### Backend API
```bash
POSTGRES_CONNECTION_STRING=postgresql://...
KAFKA_BROKER=localhost:9092
MINIO_ENDPOINT=localhost:9000
AZURE_SEARCH_ENDPOINT=https://...
AZURE_SEARCH_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_KEY=<key>
```

### DINO Service
```bash
DINO_MODEL_ID=facebook/dinov2-base
MINIO_ENDPOINT=localhost:9000
AZURE_SEARCH_ENDPOINT=https://...
USE_MINIO=true
```

---

## 🐛 Troubleshooting

### Can't connect to PostgreSQL
```bash
docker-compose ps postgres
docker-compose logs postgres
psql -h localhost -U guira -d guira_db
```

### DINO service out of memory
```bash
export DINO_MODEL_ID=facebook/dinov2-small  # Use smaller model
export CUDA_VISIBLE_DEVICES=""              # Use CPU
```

### Kafka consumer lag
```bash
kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group fire-detection-group
```

### Frontend can't connect to backend
Check CORS settings in `backend/api/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Performance Benchmarks

| Component | Latency (P50) | Throughput |
|-----------|---------------|------------|
| YOLO Fire Detection | 35ms | 28 FPS |
| TimeSFormer Smoke | 120ms | 8 FPS |
| DINO Embedding | 280ms | 3.5 images/sec |
| PhysX Surrogate | 42ms | 23 predictions/sec |
| Full PhysX Sim | 5-50 seconds | 0.02-0.2 sims/sec |

---

## 🔐 Security Checklist

- [ ] Use HTTPS/TLS for all external endpoints
- [ ] Store secrets in Azure Key Vault or Kubernetes Secrets
- [ ] Enable authentication (JWT tokens)
- [ ] Configure network policies (K8s) or NSGs (Azure)
- [ ] Enable audit logging for all API calls
- [ ] Implement rate limiting
- [ ] Use private endpoints for databases
- [ ] Rotate credentials regularly

---

## 📦 Docker Compose Services

```bash
cd integrations/guira_core/infra
docker-compose ps
```

| Service | Port | Purpose |
|---------|------|---------|
| postgres | 5432 | PostgreSQL + PostGIS |
| kafka | 9092 | Event streaming |
| zookeeper | 2181 | Kafka coordination |
| minio | 9000, 9001 | Object storage |

---

## 🎯 Development Workflow

1. **Make changes** to code
2. **Run tests** (`pytest tests/`)
3. **Check formatting** (`black . && ruff check .`)
4. **Test locally** (start services and verify)
5. **Commit** with descriptive message
6. **Push** to feature branch
7. **Create PR** with documentation updates

---

## 📚 Additional Resources

- **GitHub Issues:** https://github.com/THEDIFY/FIREPREVENTION/issues
- **Discussions:** https://github.com/THEDIFY/FIREPREVENTION/discussions
- **PhysX SDK:** https://github.com/NVIDIA-Omniverse/PhysX
- **DINOv2 Paper:** https://arxiv.org/abs/2304.07193
- **Azure AI Search:** https://docs.microsoft.com/azure/search/
- **FastAPI Docs:** https://fastapi.tiangolo.com/

---

## 🚀 Next Steps

1. **Explore Components:**
   - Review [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
   - Read [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md) for API details

2. **Set Up Locally:**
   - Follow [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
   - Train PhysX surrogate model
   - Test DINO embedding extraction

3. **Deploy to Production:**
   - Review [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)
   - Configure Azure resources
   - Set up monitoring and alerts

4. **Extend Functionality:**
   - Fine-tune DINO on fire-specific imagery
   - Train surrogate on local terrain data
   - Integrate real-time weather APIs

---

**Last Updated:** 2025-12-20

**Version:** 1.0.0

**Maintainers:** GUIRA Development Team
