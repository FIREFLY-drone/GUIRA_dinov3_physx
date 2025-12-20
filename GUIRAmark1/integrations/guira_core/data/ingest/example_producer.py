"""
Example Kafka producer for testing the detection ingestion consumer.

This script demonstrates how to send detection events to Kafka topics
that will be consumed by the ingestion consumer and stored in PostGIS.

Usage:
    python example_producer.py --bootstrap-servers localhost:9092 --count 10
"""

import argparse
import json
import os
import uuid
from datetime import datetime
from typing import Optional
import time

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("kafka-python is required. Install with: pip install kafka-python")
    exit(1)


def create_sample_detection(
    detection_type: str = "fire",
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> dict:
    """
    Create a sample detection event.
    
    Args:
        detection_type: Type of detection (fire, smoke, fauna, vegetation)
        lat: Latitude (optional)
        lon: Longitude (optional)
    
    Returns:
        Detection event dictionary
    """
    # Random coordinates around San Francisco if not provided
    if lat is None:
        import random
        lat = 37.7749 + random.uniform(-0.1, 0.1)
    if lon is None:
        import random
        lon = -122.4194 + random.uniform(-0.1, 0.1)
    
    return {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'source': f'yolov8-{detection_type}',
        'class': detection_type,
        'confidence': 0.85 + (0.15 * (hash(str(uuid.uuid4())) % 100) / 100),
        'lat': lat,
        'lon': lon,
        'metadata': {
            'bbox': [100, 100, 300, 300],
            'frame_id': int(time.time() * 1000) % 10000,
            'camera_id': 'drone-001',
            'altitude': 100.0
        }
    }


def create_sample_forecast() -> dict:
    """Create a sample forecast/simulation event."""
    return {
        'request_id': str(uuid.uuid4()),
        'results_uri': f's3://guira-storage/simulations/{uuid.uuid4()}.json',
        'meta': {
            'model': 'physx',
            'duration': 3600,
            'wind_speed': 15.5,
            'temperature': 35.0,
            'humidity': 0.25,
            'grid_size': '1000x1000'
        },
        'status': 'completed'
    }


def create_sample_embedding() -> dict:
    """Create a sample embedding event."""
    import random
    return {
        'id': str(uuid.uuid4()),
        'session_id': 'session-' + str(uuid.uuid4())[:8],
        'detection_id': str(uuid.uuid4()),
        'embedding_vector': [random.random() for _ in range(768)],
        'text_content': 'Fire detected in sector A with high confidence',
        'metadata': {
            'model': 'dinov2',
            'extraction_time': datetime.now().isoformat()
        }
    }


def send_test_events(
    bootstrap_servers: str,
    num_events: int = 10,
    event_type: str = "all"
):
    """
    Send test events to Kafka topics.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        num_events: Number of events to send
        event_type: Type of events (all, detections, simulations, embeddings)
    """
    print(f"Connecting to Kafka at {bootstrap_servers}...")
    
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        compression_type='snappy'
    )
    
    print(f"Sending {num_events} test events...")
    
    sent_counts = {'detections': 0, 'simulations': 0, 'embeddings': 0}
    
    for i in range(num_events):
        if event_type in ['all', 'detections']:
            # Send detection event
            detection_types = ['fire', 'smoke', 'fauna', 'vegetation']
            det_type = detection_types[i % len(detection_types)]
            detection = create_sample_detection(detection_type=det_type)
            
            future = producer.send('detections', detection)
            try:
                future.get(timeout=10)
                sent_counts['detections'] += 1
                print(f"  ✓ Sent detection {i+1}/{num_events} (type={det_type}, id={detection['id'][:8]}...)")
            except KafkaError as e:
                print(f"  ✗ Failed to send detection: {e}")
        
        if event_type in ['all', 'simulations'] and i % 5 == 0:
            # Send simulation event (less frequently)
            forecast = create_sample_forecast()
            
            future = producer.send('simulations', forecast)
            try:
                future.get(timeout=10)
                sent_counts['simulations'] += 1
                print(f"  ✓ Sent simulation (id={forecast['request_id'][:8]}...)")
            except KafkaError as e:
                print(f"  ✗ Failed to send simulation: {e}")
        
        if event_type in ['all', 'embeddings'] and i % 3 == 0:
            # Send embedding event
            embedding = create_sample_embedding()
            
            future = producer.send('frames.embeddings', embedding)
            try:
                future.get(timeout=10)
                sent_counts['embeddings'] += 1
                print(f"  ✓ Sent embedding (id={embedding['id'][:8]}...)")
            except KafkaError as e:
                print(f"  ✗ Failed to send embedding: {e}")
        
        # Small delay between events
        time.sleep(0.1)
    
    producer.flush()
    producer.close()
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Detections sent: {sent_counts['detections']}")
    print(f"  Simulations sent: {sent_counts['simulations']}")
    print(f"  Embeddings sent: {sent_counts['embeddings']}")
    print("="*50)
    print("\nEvents sent successfully!")
    print("\nTo verify in database:")
    print('  psql "postgresql://guira:guira_pass@localhost:5432/guira" \\')
    print('    -c "SELECT count(*) FROM detections;"')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Send test events to Kafka for ingestion testing"
    )
    parser.add_argument(
        '--bootstrap-servers',
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of events to send'
    )
    parser.add_argument(
        '--type',
        choices=['all', 'detections', 'simulations', 'embeddings'],
        default='all',
        help='Type of events to send'
    )
    
    args = parser.parse_args()
    
    try:
        send_test_events(
            bootstrap_servers=args.bootstrap_servers,
            num_events=args.count,
            event_type=args.type
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main()
