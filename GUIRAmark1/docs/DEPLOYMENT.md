# GUIRA Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Deployment Architectures](#deployment-architectures)
- [Local Development](#local-development)
- [Cloud Deployment (Azure)](#cloud-deployment-azure)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration Management](#configuration-management)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides comprehensive deployment instructions for GUIRA fire detection system across different environments:

- **Local Development** - Docker Compose for rapid development
- **Cloud (Azure)** - Production-ready Azure deployment with Bicep IaC
- **Kubernetes** - Scalable container orchestration

**Key Services to Deploy:**
1. **Backend API** - FastAPI orchestrator
2. **PhysX Surrogate** - Fire spread prediction service
3. **DINO Embed Service** - Vision embedding extraction
4. **Vision Probes** - YOLO and TimeSFormer detection services
5. **Data Layer** - Kafka, PostgreSQL/PostGIS, MinIO, Azure AI Search
6. **Frontend** - React dashboard

---

## Deployment Architectures

### Local Development Architecture

```mermaid
graph TB
    subgraph "Docker Compose"
        A[PostgreSQL + PostGIS]
        B[Kafka + Zookeeper]
        C[MinIO]
    end
    
    subgraph "Python Services"
        D[Backend API :8000]
        E[DINO Service :8002]
        F[YOLO Probe :8003]
    end
    
    subgraph "Frontend"
        G[React Dev Server :5173]
    end
    
    D --> A
    D --> B
    D --> C
    E --> C
    F --> B
    G --> D
    G --> E
```

### Cloud (Azure) Architecture

```mermaid
graph TB
    subgraph "Azure App Service"
        A[Backend API Container]
        B[DINO Service Container]
        C[Vision Probes Container]
    end
    
    subgraph "Azure PaaS Services"
        D[PostgreSQL Flexible Server]
        E[Event Hubs - Kafka]
        F[Storage Account - Blob]
        G[AI Search]
        H[Container Registry]
    end
    
    subgraph "Frontend"
        I[Static Web App]
    end
    
    A --> D
    A --> E
    A --> F
    B --> F
    B --> G
    C --> E
    I --> A
    
    H -.->|Pull Images| A
    H -.->|Pull Images| B
    H -.->|Pull Images| C
```

### Kubernetes Architecture

```mermaid
graph TB
    subgraph "Ingress Layer"
        A[NGINX Ingress]
    end
    
    subgraph "guira-vision namespace"
        B[YOLO Probe Deployment]
        C[TimeSFormer Probe Deployment]
        D[DINO Service Deployment]
    end
    
    subgraph "guira-orchestrator namespace"
        E[Backend API Deployment]
        F[PhysX Surrogate Deployment]
    end
    
    subgraph "guira-data namespace"
        G[Kafka StatefulSet]
        H[PostgreSQL StatefulSet]
        I[MinIO Deployment]
    end
    
    A --> E
    E --> B
    E --> C
    E --> D
    E --> G
    E --> H
    E --> I
    D --> I
```

---

## Local Development

### Prerequisites

```bash
# Docker & Docker Compose
docker --version  # >= 20.10
docker-compose --version  # >= 1.29

# Node.js (for frontend)
node --version  # >= 18.x
npm --version   # >= 9.x

# Python
python --version  # >= 3.10
```

### Step 1: Clone Repository

```bash
git clone https://github.com/THEDIFY/FIREPREVENTION.git
cd FIREPREVENTION
```

### Step 2: Environment Configuration

Create `.env` file in `integrations/guira_core/infra/`:

```bash
# Database
POSTGRES_USER=guira
POSTGRES_PASSWORD=guira_secure_password
POSTGRES_DB=guira_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Kafka
KAFKA_BROKER=localhost:9092
KAFKA_ZOOKEEPER=localhost:2181

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin_secure
MINIO_USE_SSL=false

# Azure AI Search (optional for local dev)
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=fire-embeddings

# Model Configuration
DINO_MODEL_ID=facebook/dinov2-base
YOLO_MODEL_PATH=../../../yolov8n.pt

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Step 3: Start Infrastructure Services

```bash
cd integrations/guira_core/infra

# Start PostgreSQL, Kafka, MinIO
./local_start.sh

# Verify services are running
docker-compose ps
```

**Expected Output:**
```
NAME                SERVICE      STATUS
postgres            postgres     Up
kafka               kafka        Up
zookeeper           zookeeper    Up
minio               minio        Up
```

### Step 4: Initialize Database

```bash
cd integrations/guira_core/infra/sql

# Run database initialization
./setup_database.sh

# Verify PostGIS extension
psql -h localhost -U guira -d guira_db -c "SELECT PostGIS_Version();"
```

### Step 5: Start Backend Services

```bash
# Terminal 1: Backend API
cd integrations/guira_core/orchestrator/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: DINO Embedding Service
cd integrations/guira_core/vision/embed_service
uvicorn app:app --host 0.0.0.0 --port 8002 --reload

# Terminal 3: YOLO Fire Probe
cd integrations/guira_core/vision/probes/yolo_probe
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
```

### Step 6: Start Frontend

```bash
cd integrations/guira_core/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Access Points:**
- Frontend: http://localhost:5173
- Backend API Docs: http://localhost:8000/docs
- DINO API Docs: http://localhost:8002/docs
- YOLO Probe Docs: http://localhost:8003/docs
- MinIO Console: http://localhost:9001

### Step 7: Test the System

```bash
# Run integration tests
cd integrations/guira_core/orchestrator/api
python test_integration.py

# Expected output: All tests pass
```

---

## Cloud Deployment (Azure)

### Architecture Overview

```
Azure Resource Group: guira-rg
├── App Service Plan (Linux, P2v3)
│   ├── Backend API Container
│   ├── DINO Service Container
│   └── Vision Probes Container
├── PostgreSQL Flexible Server (PostGIS enabled)
├── Event Hubs (Kafka-compatible)
├── Storage Account
│   ├── Blob Container: embeds
│   └── Blob Container: models
├── AI Search (Standard tier)
├── Container Registry
├── Application Insights
└── Static Web App (Frontend)
```

### Prerequisites

```bash
# Azure CLI
az --version  # >= 2.40.0

# Login to Azure
az login

# Set subscription
az account set --subscription <your-subscription-id>

# Install Bicep
az bicep install
```

### Step 1: Create Resource Group

```bash
az group create \
  --name guira-rg \
  --location eastus
```

### Step 2: Deploy Infrastructure with Bicep

```bash
cd integrations/guira_core/infra/bicep

# Review parameters
cat guira_infra.bicep

# Deploy infrastructure
az deployment group create \
  --resource-group guira-rg \
  --template-file guira_infra.bicep \
  --parameters \
    environment=production \
    postgresAdminPassword=<secure-password> \
    minioAccessKey=<access-key> \
    minioSecretKey=<secret-key>
```

**Deployment Time:** ~15-20 minutes

**Outputs:**
```json
{
  "postgresConnectionString": "postgresql://...",
  "eventHubsConnectionString": "Endpoint=sb://...",
  "storageAccountConnectionString": "DefaultEndpointsProtocol=https;...",
  "searchServiceEndpoint": "https://guira-search.search.windows.net",
  "containerRegistryLoginServer": "guiraacr.azurecr.io"
}
```

### Step 3: Build and Push Container Images

```bash
# Set registry
ACR_NAME=guiraacr

# Login to ACR
az acr login --name $ACR_NAME

# Build and push backend API
docker build -t $ACR_NAME.azurecr.io/guira-api:latest \
  -f integrations/guira_core/orchestrator/api/Dockerfile .
docker push $ACR_NAME.azurecr.io/guira-api:latest

# Build and push DINO service
docker build -t $ACR_NAME.azurecr.io/guira-dino:latest \
  -f integrations/guira_core/vision/embed_service/Dockerfile .
docker push $ACR_NAME.azurecr.io/guira-dino:latest

# Build and push YOLO probe
docker build -t $ACR_NAME.azurecr.io/guira-yolo:latest \
  -f integrations/guira_core/vision/probes/yolo_probe/Dockerfile .
docker push $ACR_NAME.azurecr.io/guira-yolo:latest
```

### Step 4: Deploy Container Apps

```bash
# Create App Service Plan
az appservice plan create \
  --name guira-plan \
  --resource-group guira-rg \
  --is-linux \
  --sku P2v3

# Deploy Backend API
az webapp create \
  --name guira-api \
  --resource-group guira-rg \
  --plan guira-plan \
  --deployment-container-image-name $ACR_NAME.azurecr.io/guira-api:latest

# Configure environment variables
az webapp config appsettings set \
  --name guira-api \
  --resource-group guira-rg \
  --settings \
    POSTGRES_CONNECTION_STRING="<from bicep output>" \
    KAFKA_CONNECTION_STRING="<from bicep output>" \
    STORAGE_CONNECTION_STRING="<from bicep output>" \
    AZURE_SEARCH_ENDPOINT="<from bicep output>" \
    AZURE_SEARCH_KEY="<from portal>"

# Deploy DINO Service
az webapp create \
  --name guira-dino \
  --resource-group guira-rg \
  --plan guira-plan \
  --deployment-container-image-name $ACR_NAME.azurecr.io/guira-dino:latest

# Deploy YOLO Probe
az webapp create \
  --name guira-yolo \
  --resource-group guira-rg \
  --plan guira-plan \
  --deployment-container-image-name $ACR_NAME.azurecr.io/guira-yolo:latest
```

### Step 5: Deploy Frontend

```bash
cd integrations/guira_core/frontend

# Build frontend
npm run build

# Deploy to Static Web App
az staticwebapp create \
  --name guira-frontend \
  --resource-group guira-rg \
  --source ./dist \
  --location eastus \
  --branch main \
  --app-location "/" \
  --api-location "" \
  --output-location "dist"
```

### Step 6: Configure Custom Domain & SSL

```bash
# Add custom domain
az staticwebapp hostname set \
  --name guira-frontend \
  --resource-group guira-rg \
  --hostname fire.yourdomain.com

# SSL is automatically provisioned by Azure
```

### Step 7: Enable Continuous Deployment

```bash
# Enable webhook for automatic updates
az acr webhook create \
  --name guira-webhook \
  --registry $ACR_NAME \
  --uri https://guira-api.scm.azurewebsites.net/docker/hook \
  --actions push \
  --scope guira-api:latest
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# kubectl
kubectl version --client

# Helm
helm version

# Azure AKS (or other K8s cluster)
az aks get-credentials --resource-group guira-rg --name guira-aks
```

### Step 1: Create Namespaces

```bash
kubectl create namespace guira-vision
kubectl create namespace guira-orchestrator
kubectl create namespace guira-data
```

### Step 2: Deploy Data Layer

```bash
cd integrations/guira_core/infra/k8s

# Deploy PostgreSQL
kubectl apply -f postgres-statefulset.yaml -n guira-data

# Deploy Kafka
kubectl apply -f kafka-statefulset.yaml -n guira-data

# Deploy MinIO
kubectl apply -f minio-deployment.yaml -n guira-data
```

### Step 3: Deploy Application Services

```bash
# Deploy Backend API
kubectl apply -f api-deployment.yaml -n guira-orchestrator

# Deploy DINO Service
kubectl apply -f dino-deployment.yaml -n guira-vision

# Deploy YOLO Probe
kubectl apply -f yolo-deployment.yaml -n guira-vision
```

### Step 4: Configure Ingress

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx

# Apply ingress rules
kubectl apply -f ingress.yaml
```

**ingress.yaml Example:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: guira-ingress
  namespace: guira-orchestrator
spec:
  ingressClassName: nginx
  rules:
  - host: api.guira.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: guira-api
            port:
              number: 8000
  - host: dino.guira.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: guira-dino
            port:
              number: 8002
```

### Step 5: Configure Auto-scaling

```bash
# Horizontal Pod Autoscaler for Backend API
kubectl autoscale deployment guira-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n guira-orchestrator

# HPA for DINO Service
kubectl autoscale deployment guira-dino \
  --cpu-percent=80 \
  --min=1 \
  --max=5 \
  -n guira-vision
```

---

## Configuration Management

### Environment Variables

**Backend API:**
```bash
POSTGRES_CONNECTION_STRING=postgresql://user:pass@host:5432/guira_db
KAFKA_BROKER=kafka:9092
MINIO_ENDPOINT=minio:9000
AZURE_SEARCH_ENDPOINT=https://search.search.windows.net
AZURE_SEARCH_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://openai.openai.azure.com
AZURE_OPENAI_KEY=<key>
LOG_LEVEL=INFO
```

**DINO Service:**
```bash
DINO_MODEL_ID=facebook/dinov2-base
MINIO_ENDPOINT=minio:9000
AZURE_SEARCH_ENDPOINT=https://search.search.windows.net
USE_MINIO=true
```

### Secrets Management

**Azure Key Vault:**
```bash
# Create Key Vault
az keyvault create \
  --name guira-kv \
  --resource-group guira-rg \
  --location eastus

# Add secrets
az keyvault secret set \
  --vault-name guira-kv \
  --name postgres-password \
  --value <secure-password>

# Grant access to App Service
az webapp identity assign \
  --name guira-api \
  --resource-group guira-rg

az keyvault set-policy \
  --name guira-kv \
  --object-id <app-service-identity> \
  --secret-permissions get list
```

**Kubernetes Secrets:**
```bash
# Create secret from literal
kubectl create secret generic postgres-secret \
  --from-literal=password=<secure-password> \
  -n guira-data

# Create secret from file
kubectl create secret generic azure-secrets \
  --from-file=search-key=./search-key.txt \
  --from-file=openai-key=./openai-key.txt \
  -n guira-orchestrator
```

---

## Monitoring & Observability

### Azure Application Insights

```bash
# Enable Application Insights
az monitor app-insights component create \
  --app guira-insights \
  --location eastus \
  --resource-group guira-rg

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app guira-insights \
  --resource-group guira-rg \
  --query instrumentationKey -o tsv)

# Configure in App Service
az webapp config appsettings set \
  --name guira-api \
  --resource-group guira-rg \
  --settings APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=$INSTRUMENTATION_KEY"
```

### Prometheus & Grafana (Kubernetes)

```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80

# Default credentials: admin / prom-operator
```

**Key Metrics to Monitor:**
- API request latency (P50, P95, P99)
- Fire detection throughput (detections/sec)
- PhysX surrogate inference time
- DINO embedding extraction time
- Kafka lag
- Database connection pool usage

### Logging

**Structured Logging Format:**
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "service": "guira-api",
  "trace_id": "uuid",
  "user_id": "user123",
  "message": "Fire detected",
  "data": {
    "confidence": 0.95,
    "location": {"lat": 40.7128, "lon": -74.0060}
  }
}
```

---

## Security

### Network Security

**Azure:**
- Use Private Endpoints for PostgreSQL and Storage Account
- Configure Network Security Groups (NSG)
- Enable Azure Firewall for outbound traffic control

**Kubernetes:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: guira-api-policy
  namespace: guira-orchestrator
spec:
  podSelector:
    matchLabels:
      app: guira-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: guira-vision
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: guira-data
```

### Authentication & Authorization

**JWT Tokens:**
```python
# Backend API
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/protected")
async def protected_route(token: str = Depends(security)):
    # Verify JWT token
    payload = verify_jwt(token.credentials)
    return {"user_id": payload["sub"]}
```

### Data Encryption

- **In Transit:** TLS 1.3 for all HTTPS connections
- **At Rest:** Azure Storage encryption (AES-256)
- **Database:** PostgreSQL SSL mode required

---

## Troubleshooting

### Common Issues

**Issue: PostgreSQL connection refused**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
psql -h localhost -U guira -d guira_db
```

**Issue: Kafka consumer lag**
```bash
# Check consumer group lag
kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe \
  --group fire-detection-group
```

**Issue: DINO service OOM (Out of Memory)**
```bash
# Reduce batch size in environment
export DINO_BATCH_SIZE=1

# Use smaller model
export DINO_MODEL_ID=facebook/dinov2-small

# Process images on CPU
export CUDA_VISIBLE_DEVICES=""
```

**Issue: Frontend not connecting to backend**
```bash
# Check CORS configuration in backend
# backend/api/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Health Checks

**Backend API:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "services": {...}}
```

**DINO Service:**
```bash
curl http://localhost:8002/health
# Expected: {"status": "healthy", "model_loaded": true}
```

**Database:**
```bash
curl http://localhost:8000/health/db
# Expected: {"status": "connected", "version": "15.x"}
```

---

## Performance Tuning

### Backend API

```python
# uvicorn configuration for production
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --http httptools \
  --limit-concurrency 1000 \
  --backlog 2048
```

### PostgreSQL

```sql
-- Optimize for geospatial queries
CREATE INDEX CONCURRENTLY fire_locations_gix 
  ON fire_detections USING GIST (location);

-- Partition large tables
CREATE TABLE fire_detections_2025_01 
  PARTITION OF fire_detections 
  FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Configure connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
```

### Kafka

```bash
# Increase partition count for high throughput
kafka-topics.sh --alter \
  --topic fire_detections \
  --partitions 10 \
  --bootstrap-server localhost:9092

# Configure retention
kafka-configs.sh --alter \
  --entity-type topics \
  --entity-name fire_detections \
  --add-config retention.ms=604800000  # 7 days
```

---

## Backup & Disaster Recovery

### Database Backups

```bash
# Automated daily backups (cron)
0 2 * * * pg_dump -h localhost -U guira guira_db | gzip > /backups/guira_$(date +\%Y\%m\%d).sql.gz

# Restore from backup
gunzip < /backups/guira_20250101.sql.gz | psql -h localhost -U guira guira_db
```

### Azure Backup

```bash
# Enable automated backups for PostgreSQL
az postgres flexible-server backup create \
  --resource-group guira-rg \
  --name guira-postgres \
  --backup-name manual-backup-20250101
```

---

## Scaling Guidelines

| Component | Initial | Scale To | Trigger |
|-----------|---------|----------|---------|
| Backend API | 2 instances | 10 instances | CPU > 70% |
| DINO Service | 1 instance | 5 instances | Queue depth > 100 |
| YOLO Probe | 2 instances | 8 instances | Requests/sec > 50 |
| PostgreSQL | 2 vCPU, 8GB | 8 vCPU, 32GB | Connection pool > 80% |
| Kafka | 3 brokers | 9 brokers | Lag > 10000 msgs |

---

## Cost Optimization

**Azure:**
- Use Reserved Instances for predictable workloads (save 30-50%)
- Enable auto-shutdown for dev/test environments
- Use Azure Spot VMs for batch processing
- Implement blob lifecycle management (move to cool/archive tier)

**General:**
- Cache frequently accessed embeddings in Redis
- Batch process images during off-peak hours
- Use smaller DINO model for non-critical queries
- Compress embeddings before storage

---

## References

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) - Integration setup
- [Azure Bicep Templates](./bicep/) - Infrastructure as Code
- [Kubernetes Manifests](./k8s/) - K8s deployment files
