# GUIRA Data Stores & Ingestion Infrastructure

## Overview

This document describes the data storage and message bus infrastructure for the GUIRA fire prevention system. The system uses PostGIS for geospatial data storage, MinIO/Azure Blob for object storage, and Kafka (or Redis Streams) for real-time message processing.

## Components

### 1. PostGIS Database

**Purpose**: Store detection events, forecasts, embeddings, and geospatial data with spatial indexing.

**Schema**: See `sql/init_postgis.sql` for complete schema definition.

**Tables**:
- `detections` - All detection events from fire, smoke, fauna, vegetation models
- `forecasts` - Fire spread predictions and simulation results
- `sessions` - Live ingestion and analysis session tracking
- `embeddings` - Vector embeddings for RAG analysis

**Connection String Format**:
```
postgresql://username:password@host:port/database
```

### 2. Kafka Message Bus

**Purpose**: Real-time event streaming for frames, detections, and simulations.

**Topics**:
- `frames.raw` - Raw video frames (retention: 1 hour)
- `frames.embeddings` - Frame embeddings for RAG (retention: 24 hours)
- `detections` - All detection events (retention: 7 days)
- `simulations` - Fire spread simulations (retention: 7 days)
- `alerts` - High-priority notifications (retention: 30 days)

**Configuration**:
- Partitions: 2-3 per topic for parallelism
- Compression: Snappy for detections, LZ4 for embeddings
- Replication: 1 (local dev), 3 (production)

### 3. Object Storage (MinIO/Azure Blob)

**Purpose**: Store raw video files, model artifacts, simulation outputs.

**Bucket Structure**:
```
guira-storage/
├── raw-video/          # Original video streams
├── processed-frames/   # Processed frame images
├── embeddings/         # Embedding vectors
├── simulations/        # Simulation outputs
└── models/            # Model checkpoints
```

## Local Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- PostgreSQL client tools (psql)

### Step 1: Start Infrastructure Services

**PostGIS, MinIO, Redis**:
```bash
cd integrations/guira_core/infra/local
docker-compose up -d
```

**Kafka (optional, separate compose file)**:
```bash
cd integrations/guira_core/infra/kafka
docker-compose -f docker-compose.kafka.yml up -d
```

This will start:
- PostGIS on port 5432
- MinIO on port 9000
- Redis on port 6379
- Kafka on port 9092
- Zookeeper on port 2181
- Kafka UI on port 8080

### Step 2: Initialize Database Schema

```bash
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -f integrations/guira_core/infra/sql/init_postgis.sql
```

Verify tables:
```bash
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "\dt"
```

### Step 3: Verify Kafka Topics

Access Kafka UI at http://localhost:8080 to view topics, or use CLI:

```bash
docker exec -it guira-kafka kafka-topics --list \
  --bootstrap-server localhost:9092
```

### Step 4: Configure Environment Variables

Copy `.env.example` to `.env` and update:

```bash
cp .env.example .env
```

Key variables:
```
POSTGRES_CONN="postgresql://guira:guira_pass@localhost:5432/guira"
KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
REDIS_URL="redis://localhost:6379/0"
```

## Running the Ingestion Consumer

### Start Detection Ingestion

```bash
python integrations/guira_core/data/ingest/ingest_detection.py \
  --kafka-bootstrap-servers localhost:9092 \
  --postgres-conn "postgresql://guira:guira_pass@localhost:5432/guira" \
  --consumer-group guira-ingestion \
  --topics detections frames.embeddings simulations
```

### Monitor Ingestion

Check logs for processing status:
```
2024-01-15 10:30:00 - INFO - Connected to PostgreSQL database
2024-01-15 10:30:01 - INFO - Connected to Kafka at localhost:9092
2024-01-15 10:30:01 - INFO - Subscribed to topics: ['detections', 'frames.embeddings', 'simulations']
2024-01-15 10:30:02 - INFO - Starting message consumption...
2024-01-15 10:30:15 - DEBUG - Stored detection abc123 (class=fire, confidence=0.92)
```

### Test with Sample Data

Send a test detection event:

```bash
# Install kafka-python if not already installed
pip install kafka-python

# Python script to send test message
python << EOF
from kafka import KafkaProducer
import json
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

detection = {
    'id': 'test-detection-001',
    'timestamp': datetime.now().isoformat(),
    'source': 'yolov8-fire',
    'class': 'fire',
    'confidence': 0.95,
    'lat': 40.7128,
    'lon': -74.0060,
    'metadata': {
        'bbox': [100, 100, 200, 200],
        'frame_id': 1234
    }
}

producer.send('detections', detection)
producer.flush()
print("Test detection sent!")
EOF
```

Query the database:
```bash
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT id, ts, source, class, confidence, ST_AsText(geom) FROM detections LIMIT 5;"
```

## Production Deployment

### Security Considerations

1. **Database Connections**:
   - Use SSL/TLS: `sslmode=require` in connection string
   - Store credentials in Azure Key Vault
   - Rotate passwords regularly

2. **Kafka**:
   - Enable SASL authentication
   - Use SSL/TLS encryption
   - Configure ACLs for topic access

3. **Object Storage**:
   - Use managed identities (Azure)
   - Enable encryption at rest
   - Configure lifecycle policies for retention

### Azure Production Setup

#### PostGIS (Azure Database for PostgreSQL)

```bash
# Create resource group
az group create --name guira-rg --location eastus

# Create PostgreSQL server
az postgres flexible-server create \
  --resource-group guira-rg \
  --name guira-postgres \
  --location eastus \
  --admin-user guira \
  --admin-password <secure-password> \
  --sku-name Standard_D2s_v3 \
  --storage-size 128 \
  --version 13

# Enable PostGIS
az postgres flexible-server parameter set \
  --resource-group guira-rg \
  --server-name guira-postgres \
  --name azure.extensions \
  --value POSTGIS

# Initialize schema
psql "postgresql://guira:<password>@guira-postgres.postgres.database.azure.com:5432/guira?sslmode=require" \
  -f integrations/guira_core/infra/sql/init_postgis.sql
```

#### Azure Event Hubs (Kafka-compatible)

```bash
# Create Event Hubs namespace
az eventhubs namespace create \
  --resource-group guira-rg \
  --name guira-events \
  --location eastus \
  --sku Standard

# Create topics (event hubs)
for topic in frames-raw frames-embeddings detections simulations alerts; do
  az eventhubs eventhub create \
    --resource-group guira-rg \
    --namespace-name guira-events \
    --name $topic \
    --partition-count 3 \
    --message-retention 7
done
```

Connection string:
```
guira-events.servicebus.windows.net:9093
```

#### Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --resource-group guira-rg \
  --name guirastorage \
  --location eastus \
  --sku Standard_LRS

# Create containers
for container in raw-video processed-frames embeddings simulations models; do
  az storage container create \
    --account-name guirastorage \
    --name $container
done
```

### Environment Variables (Production)

```bash
# Database
POSTGRES_CONN="postgresql://guira:<password>@guira-postgres.postgres.database.azure.com:5432/guira?sslmode=require"

# Message Bus (Azure Event Hubs)
KAFKA_BOOTSTRAP_SERVERS="guira-events.servicebus.windows.net:9093"
KAFKA_SASL_MECHANISM="PLAIN"
KAFKA_SECURITY_PROTOCOL="SASL_SSL"
KAFKA_SASL_USERNAME="$ConnectionString"
KAFKA_SASL_PASSWORD="<event-hub-connection-string>"

# Object Storage (Azure Blob)
AZURE_STORAGE_ACCOUNT="guirastorage"
AZURE_STORAGE_CONNSTR="<connection-string>"

# Key Vault
USE_KEYVAULT=true
AZURE_KEYVAULT_URL="https://guira-keyvault.vault.azure.net/"
```

## Monitoring and Operations

### Health Checks

**Database**:
```bash
psql $POSTGRES_CONN -c "SELECT count(*) FROM detections;"
```

**Kafka**:
```bash
kafka-consumer-groups --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --group guira-ingestion --describe
```

**MinIO/Blob**:
```bash
# MinIO
mc alias set guira http://localhost:9000 minioadmin minioadmin
mc ls guira/

# Azure Blob
az storage blob list --account-name guirastorage --container-name raw-video
```

### Performance Metrics

Track these metrics:
- **Ingestion latency**: Time from Kafka message to DB insertion
- **Throughput**: Messages per second processed
- **Database query performance**: Query execution times
- **Storage usage**: Disk space and blob storage growth

### Troubleshooting

**Consumer not processing messages**:
```bash
# Check consumer group status
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group guira-ingestion --describe

# Reset offsets if needed
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group guira-ingestion --reset-offsets --to-latest --all-topics --execute
```

**Database connection issues**:
```bash
# Test connection
psql $POSTGRES_CONN -c "SELECT version();"

# Check active connections
psql $POSTGRES_CONN -c "SELECT count(*) FROM pg_stat_activity;"
```

**Kafka broker unreachable**:
```bash
# Check if Kafka is running
docker ps | grep kafka

# View logs
docker logs guira-kafka

# Restart if needed
docker-compose -f docker-compose.kafka.yml restart kafka
```

## Data Retention Policies

### Kafka Topics

- `frames.raw`: 1 hour (high volume, temporary)
- `frames.embeddings`: 24 hours
- `detections`: 7 days
- `simulations`: 7 days
- `alerts`: 30 days

### Database

Configure archival for old data:
```sql
-- Archive detections older than 90 days
CREATE TABLE detections_archive (LIKE detections INCLUDING ALL);

INSERT INTO detections_archive 
SELECT * FROM detections 
WHERE ts < NOW() - INTERVAL '90 days';

DELETE FROM detections 
WHERE ts < NOW() - INTERVAL '90 days';
```

### Object Storage

Use lifecycle policies:
```bash
# Azure Blob lifecycle management
az storage account management-policy create \
  --account-name guirastorage \
  --policy @lifecycle-policy.json
```

Example `lifecycle-policy.json`:
```json
{
  "rules": [
    {
      "enabled": true,
      "name": "move-old-to-cool",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {"daysAfterModificationGreaterThan": 30},
            "tierToArchive": {"daysAfterModificationGreaterThan": 90},
            "delete": {"daysAfterModificationGreaterThan": 365}
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["raw-video/"]
        }
      }
    }
  ]
}
```

## Integration with Existing Pipeline

The ingestion consumer integrates with the existing GUIRA pipeline:

1. **run_pipeline.py** → Produces detection events → Kafka `detections` topic
2. **Ingestion Consumer** → Consumes from Kafka → Writes to PostGIS
3. **RAG/Analysis** → Queries PostGIS → Performs spatial analysis

Example producer integration in `run_pipeline.py`:

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def publish_detection(detection_result):
    """Publish detection to Kafka topic."""
    message = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'source': 'yolov8-fire',
        'class': detection_result['class'],
        'confidence': detection_result['confidence'],
        'lat': detection_result.get('lat'),
        'lon': detection_result.get('lon'),
        'metadata': detection_result.get('metadata', {})
    }
    producer.send('detections', message)
```

## References

- PostGIS Documentation: https://postgis.net/documentation/
- Kafka Documentation: https://kafka.apache.org/documentation/
- Azure Event Hubs (Kafka): https://docs.microsoft.com/azure/event-hubs/
- MinIO Documentation: https://min.io/docs/
- Azure PostgreSQL: https://docs.microsoft.com/azure/postgresql/
