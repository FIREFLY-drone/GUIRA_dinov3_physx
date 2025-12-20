# GUIRA Detection Ingestion Module

Real-time ingestion of detection events, embeddings, and simulation results from Kafka to PostGIS.

## Overview

This module provides Kafka consumers that process detection events from various sources (fire detection, smoke detection, fauna detection, etc.) and store them in a PostGIS database for spatial analysis and querying.

## Components

### `ingest_detection.py`

Main ingestion consumer that:
- Subscribes to Kafka topics (`detections`, `frames.embeddings`, `simulations`)
- Processes messages in real-time
- Stores events in PostGIS with spatial indexing
- Handles geospatial data (lat/lon → PostGIS POINT geometry)

**MODEL**: N/A (Infrastructure component)  
**DATA**: Kafka topics: frames.raw, frames.embeddings, detections, simulations  
**TRAINING/BUILD RECIPE**: N/A  
**EVAL & ACCEPTANCE**: Successfully processes and stores detection events with <1s latency

### `example_producer.py`

Test data generator that sends sample events to Kafka topics for testing the ingestion pipeline.

## Usage

### Start the Ingestion Consumer

```bash
python ingest_detection.py \
  --kafka-bootstrap-servers localhost:9092 \
  --postgres-conn "postgresql://guira:guira_pass@localhost:5432/guira" \
  --consumer-group guira-ingestion \
  --topics detections frames.embeddings simulations
```

### Command-Line Arguments

- `--kafka-bootstrap-servers`: Kafka broker addresses (default: `localhost:9092`)
- `--postgres-conn`: PostgreSQL connection string
- `--consumer-group`: Kafka consumer group ID (default: `guira-ingestion`)
- `--topics`: List of topics to subscribe to (space-separated)

### Environment Variables

```bash
KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
POSTGRES_CONN="postgresql://guira:guira_pass@localhost:5432/guira"
```

## Event Formats

### Detection Event

```json
{
  "id": "unique-detection-id",
  "timestamp": "2024-01-15T10:30:00Z",
  "source": "yolov8-fire",
  "class": "fire",
  "confidence": 0.95,
  "lat": 40.7128,
  "lon": -74.0060,
  "embedding_uri": "s3://bucket/embedding.npy",
  "metadata": {
    "bbox": [100, 100, 200, 200],
    "frame_id": 1234,
    "camera_id": "drone-001"
  }
}
```

### Forecast/Simulation Event

```json
{
  "request_id": "unique-forecast-id",
  "results_uri": "s3://bucket/forecast-results.json",
  "meta": {
    "model": "physx",
    "duration": 3600,
    "wind_speed": 15.5
  },
  "status": "completed"
}
```

### Embedding Event

```json
{
  "id": "unique-embedding-id",
  "session_id": "session-123",
  "detection_id": "detection-456",
  "embedding_vector": [0.1, 0.2, 0.3, ...],
  "text_content": "Fire detected in sector A",
  "metadata": {
    "model": "dinov2"
  }
}
```

## Testing

### Run Unit Tests

```bash
cd /home/runner/work/FIREPREVENTION/FIREPREVENTION
python -m unittest tests.unit.test_ingest_detection -v
```

### Send Test Events

```bash
python example_producer.py \
  --bootstrap-servers localhost:9092 \
  --count 20 \
  --type all
```

### Verify in Database

```bash
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT count(*) FROM detections;"

psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT id, source, class, confidence, ST_AsText(geom) FROM detections LIMIT 5;"
```

## Integration Example

```python
from kafka import KafkaProducer
import json
import uuid
from datetime import datetime

# Initialize producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send detection event
detection = {
    'id': str(uuid.uuid4()),
    'timestamp': datetime.now().isoformat(),
    'source': 'yolov8-fire',
    'class': 'fire',
    'confidence': 0.92,
    'lat': 40.7128,
    'lon': -74.0060,
    'metadata': {
        'bbox': [100, 100, 200, 200],
        'frame_id': 1234
    }
}

producer.send('detections', detection)
producer.flush()
```

## Performance

- **Latency**: <1s median per message
- **Throughput**: 100+ messages/second
- **Error Handling**: Graceful with transaction rollback
- **Batch Processing**: Up to 100 messages per poll

## Error Handling

The consumer handles errors gracefully:
- Invalid message format → Logs error and continues
- Database connection loss → Logs error and exits
- Kafka connection loss → Logs error and exits

All database operations use transactions with automatic rollback on error.

## Monitoring

### Consumer Status

Check consumer lag:
```bash
docker exec -it guira-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group guira-ingestion \
  --describe
```

### Database Status

Check ingestion rate:
```bash
psql "postgresql://guira:guira_pass@localhost:5432/guira" \
  -c "SELECT source, count(*) FROM detections GROUP BY source;"
```

## Dependencies

```
kafka-python>=2.0.2
psycopg2-binary>=2.9.9
```

Install with:
```bash
pip install kafka-python psycopg2-binary
```

## Related Documentation

- [README_DATA_STORES.md](../../infra/README_DATA_STORES.md) - Infrastructure setup
- [QUICKSTART_PH10.md](../../infra/QUICKSTART_PH10.md) - Quick start guide
- [PH10_IMPLEMENTATION_SUMMARY.md](/docs/PH10_IMPLEMENTATION_SUMMARY.md) - Implementation summary

## License

See [LICENSE](../../../../LICENSE) in repository root.
