# PH-10: Data Stores & Ingestion - Quick Start Guide

This guide provides step-by-step instructions to get the data stores and ingestion pipeline running.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+
- PostgreSQL client tools (psql)

## Step 1: Start Infrastructure Services

### Option A: Start All Services (Recommended for Testing)

```bash
cd integrations/guira_core/infra

# Start PostGIS, MinIO, Redis
cd local
docker-compose up -d

# Start Kafka (in a separate compose file)
cd ../kafka
docker-compose -f docker-compose.kafka.yml up -d

# Wait for services to be ready (about 30 seconds)
sleep 30
```

### Option B: Use Existing Infrastructure

If you already have PostgreSQL and Kafka running, skip to Step 2 and configure environment variables.

## Step 2: Configure Environment Variables

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION

# Copy example environment file
cp .env.example .env

# Edit .env with your settings (or use defaults for local dev)
# Key variables for PH-10:
# - POSTGRES_CONN="postgresql://guira:guira_pass@localhost:5432/guira"
# - KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
```

## Step 3: Initialize Database Schema

```bash
# Option A: Use the setup script (recommended)
cd integrations/guira_core/infra/sql
./setup_database.sh

# Option B: Manual setup
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -f integrations/guira_core/infra/sql/init_postgis.sql
```

**Expected output:**
```
==========================================
GUIRA Database Setup
==========================================
Host: localhost
Port: 5432
User: guira
Database: guira

Testing database connection...
✓ Connection successful

Initializing database schema...
✓ Schema initialized successfully

Verifying tables...
✓ All required tables created
...
```

## Step 4: Verify Kafka Topics

```bash
# List Kafka topics
docker exec -it guira-kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Expected output:**
```
alerts
detections
frames.embeddings
frames.raw
simulations
```

Or view in Kafka UI at http://localhost:8080

## Step 5: Install Python Dependencies

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION

# Install required packages
pip install kafka-python psycopg2-binary redis minio

# Or install all requirements
pip install -r requirements.txt
```

## Step 6: Start the Ingestion Consumer

Open a new terminal and start the consumer:

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION

python integrations/guira_core/data/ingest/ingest_detection.py \
  --kafka-bootstrap-servers localhost:9092 \
  --postgres-conn "postgresql://guira:guira_pass@localhost:5432/guira" \
  --consumer-group guira-ingestion \
  --topics detections frames.embeddings simulations
```

**Expected output:**
```
2024-01-15 10:30:00 - INFO - Initialized DetectionIngestionConsumer for topics: ['detections', 'frames.embeddings', 'simulations']
2024-01-15 10:30:01 - INFO - Connected to PostgreSQL database
2024-01-15 10:30:02 - INFO - Connected to Kafka at localhost:9092
2024-01-15 10:30:02 - INFO - Subscribed to topics: ['detections', 'frames.embeddings', 'simulations']
2024-01-15 10:30:03 - INFO - Starting message consumption...
```

Leave this terminal running.

## Step 7: Send Test Events

Open another terminal and send test events:

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION

python integrations/guira_core/data/ingest/example_producer.py \
  --bootstrap-servers localhost:9092 \
  --count 20 \
  --type all
```

**Expected output:**
```
Connecting to Kafka at localhost:9092...
Sending 20 test events...
  ✓ Sent detection 1/20 (type=fire, id=a1b2c3d4...)
  ✓ Sent detection 2/20 (type=smoke, id=e5f6g7h8...)
  ✓ Sent embedding (id=i9j0k1l2...)
  ...

==================================================
Summary:
  Detections sent: 20
  Simulations sent: 4
  Embeddings sent: 7
==================================================

Events sent successfully!
```

## Step 8: Verify Data in Database

```bash
# Check detection count
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT count(*) FROM detections;"

# View recent detections
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT id, ts, source, class, confidence, ST_AsText(geom) FROM detections ORDER BY ts DESC LIMIT 5;"

# Check forecast/simulation count
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT count(*) FROM forecasts;"

# Check embeddings count
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT count(*) FROM embeddings;"
```

**Expected output:**
```
 count 
-------
    20
(1 row)
```

## Step 9: Run Unit Tests

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION

# Run ingestion consumer tests
python -m unittest tests.unit.test_ingest_detection -v
```

**Expected output:**
```
test_detection_event_creation ... ok
test_consumer_initialization ... ok
test_process_detection_event ... ok
...

----------------------------------------------------------------------
Ran 10 tests in 0.007s

OK
```

## Architecture Overview

```
┌─────────────────┐
│  Detection      │
│  Pipeline       │ ──┐
│  (run_pipeline) │   │
└─────────────────┘   │
                      │
┌─────────────────┐   │   ┌──────────────┐
│  Simulation     │   ├──▶│    Kafka     │
│  Engine         │   │   │   Topics     │
└─────────────────┘   │   └──────┬───────┘
                      │          │
┌─────────────────┐   │          │
│  Embedding      │   │          │
│  Service        │ ──┘          │
└─────────────────┘              │
                                 ▼
                      ┌──────────────────┐
                      │   Ingestion      │
                      │   Consumer       │
                      │ (ingest_detection)│
                      └──────┬───────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   PostGIS    │
                      │   Database   │
                      └──────────────┘
```

## Topics and Data Flow

| Topic              | Purpose                          | Retention | Compression |
|--------------------|----------------------------------|-----------|-------------|
| `frames.raw`       | Raw video frames                 | 1 hour    | Snappy      |
| `frames.embeddings`| Frame embeddings for RAG         | 24 hours  | LZ4         |
| `detections`       | All detection events             | 7 days    | Snappy      |
| `simulations`      | Fire spread predictions          | 7 days    | LZ4         |
| `alerts`           | High-priority notifications      | 30 days   | None        |

## Database Schema

### Tables Created

1. **detections**: Stores all detection events with geospatial data
   - Columns: id, ts, source, class, confidence, geom (POINT), embedding_uri, metadata
   - Indexes: Spatial index on geom, indexes on ts, source, class, confidence

2. **forecasts**: Stores fire spread predictions
   - Columns: request_id, created_at, results_uri, meta, status, updated_at
   - Indexes: status, created_at

3. **sessions**: Tracks live analysis sessions
   - Columns: session_id, user_id, source, source_platform, start_ts, end_ts, status, metadata
   - Indexes: user_id, status, start_ts

4. **embeddings**: Stores vector embeddings for RAG
   - Columns: id, session_id, detection_id, embedding_vector, text_content, metadata
   - Indexes: session_id, detection_id

## Troubleshooting

### Issue: Cannot connect to Kafka

```bash
# Check if Kafka is running
docker ps | grep kafka

# View Kafka logs
docker logs guira-kafka

# Restart Kafka
cd integrations/guira_core/infra/kafka
docker-compose -f docker-compose.kafka.yml restart kafka
```

### Issue: Cannot connect to PostgreSQL

```bash
# Check if PostgreSQL is running
docker ps | grep postgis

# View PostgreSQL logs
docker logs $(docker ps -qf "name=postgis")

# Restart PostgreSQL
cd integrations/guira_core/infra/local
docker-compose restart postgis
```

### Issue: Consumer not receiving messages

```bash
# Check consumer group status
docker exec -it guira-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group guira-ingestion \
  --describe

# Reset consumer offset if needed
docker exec -it guira-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group guira-ingestion \
  --reset-offsets --to-latest --all-topics --execute
```

### Issue: Port conflicts

If ports 5432, 6379, 9000, or 9092 are already in use:

1. Stop the conflicting service
2. Or modify the docker-compose files to use different ports
3. Update your .env file with the new ports

## Integration with Existing Pipeline

To integrate with `run_pipeline.py`, add this code:

```python
import os
from kafka import KafkaProducer
import json
import uuid
from datetime import datetime

# Initialize Kafka producer (once at startup)
producer = KafkaProducer(
    bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# After each detection, publish to Kafka
def publish_detection(detection_result, lat=None, lon=None):
    """Publish detection to Kafka topic."""
    message = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'source': 'yolov8-fire',
        'class': detection_result['class'],
        'confidence': detection_result['confidence'],
        'lat': lat,
        'lon': lon,
        'metadata': {
            'bbox': detection_result.get('bbox'),
            'frame_id': detection_result.get('frame_id')
        }
    }
    producer.send('detections', message)

# Flush on shutdown
producer.flush()
```

## Next Steps

1. ✅ Verify all services are running
2. ✅ Test with sample data
3. ✅ Integrate with existing detection pipeline
4. 📋 Configure production credentials (Azure)
5. 📋 Set up monitoring and alerting
6. 📋 Configure retention policies
7. 📋 Enable SSL/TLS for production

## Production Deployment

For production deployment to Azure, see:
- [README_DATA_STORES.md](README_DATA_STORES.md) - Detailed Azure setup
- Azure Database for PostgreSQL with PostGIS
- Azure Event Hubs (Kafka-compatible)
- Azure Blob Storage

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in Docker containers
3. Verify environment variables in .env
4. Check [README_DATA_STORES.md](README_DATA_STORES.md) for detailed documentation

## Acceptance Criteria (PH-10)

✅ **Deliverables Completed:**

1. ✅ `integrations/guira_core/infra/sql/init_postgis.sql` - PostGIS schema with detections and forecasts tables
2. ✅ `integrations/guira_core/infra/kafka/docker-compose.kafka.yml` - Kafka configuration with all required topics
3. ✅ `integrations/guira_core/data/ingest/ingest_detection.py` - Consumer that writes to PostGIS
4. ✅ Infrastructure docs with env vars and connection strings

✅ **Acceptance Criteria Met:**

1. ✅ Ingestion consumer writes detection rows to PostGIS
2. ✅ Forecasts table receives rows with results_uri
3. ✅ Security: SSL connections supported for production, KeyVault integration available
4. ✅ All unit tests passing

✅ **Additional Deliverables:**

- Setup script for database initialization
- Example producer for testing
- Comprehensive documentation
- Integration tests
- Troubleshooting guide
