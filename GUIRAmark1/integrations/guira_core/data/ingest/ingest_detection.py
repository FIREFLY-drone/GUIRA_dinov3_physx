"""
Detection Ingestion Consumer for GUIRA Fire Prevention System.

Consumes detection events from Kafka topics and writes them to PostGIS database.
Handles frames.raw, frames.embeddings, detections, and simulations topics.

MODEL: N/A (Infrastructure component)
DATA: Kafka topics: frames.raw, frames.embeddings, detections, simulations
TRAINING/BUILD RECIPE: N/A
EVAL & ACCEPTANCE: Successfully processes and stores detection events with <1s latency
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import logging
from dataclasses import dataclass

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None
    KafkaError = Exception

try:
    import psycopg2
    from psycopg2.extras import execute_values, Json
    from psycopg2.extensions import AsIs
except ImportError:
    psycopg2 = None
    Json = None
    AsIs = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Detection event data structure."""
    id: str
    ts: datetime
    source: str
    class_name: str
    confidence: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    embedding_uri: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DetectionIngestionConsumer:
    """
    Kafka consumer that ingests detection events and writes to PostGIS.
    
    Subscribes to Kafka topics and processes messages in real-time,
    storing them in the PostGIS database for geospatial queries and analysis.
    """
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        postgres_conn_string: str = None,
        consumer_group: str = "guira-ingestion",
        topics: Optional[List[str]] = None
    ):
        """
        Initialize the detection ingestion consumer.
        
        Args:
            kafka_bootstrap_servers: Kafka broker addresses
            postgres_conn_string: PostgreSQL connection string
            consumer_group: Kafka consumer group ID
            topics: List of topics to subscribe to
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.postgres_conn_string = postgres_conn_string or os.getenv(
            "POSTGRES_CONN",
            "postgresql://guira:guira_pass@localhost:5432/guira"
        )
        self.consumer_group = consumer_group
        self.topics = topics or ["detections", "frames.embeddings", "simulations"]
        
        # Check dependencies
        if KafkaConsumer is None:
            raise ImportError(
                "kafka-python is required. Install with: pip install kafka-python"
            )
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )
        
        self.consumer: Optional[KafkaConsumer] = None
        self.db_conn = None
        self.running = False
        
        logger.info(
            f"Initialized DetectionIngestionConsumer for topics: {self.topics}"
        )
    
    def connect_database(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.db_conn = psycopg2.connect(self.postgres_conn_string)
            self.db_conn.autocommit = False
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def connect_kafka(self):
        """Initialize Kafka consumer."""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=100,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"Connected to Kafka at {self.kafka_bootstrap_servers}")
            logger.info(f"Subscribed to topics: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def process_detection_event(self, message: Dict[str, Any]) -> Optional[DetectionEvent]:
        """
        Process a detection event message.
        
        Args:
            message: Raw message from Kafka
            
        Returns:
            DetectionEvent object or None if invalid
        """
        try:
            # Extract required fields
            detection_id = message.get('id') or str(uuid.uuid4())
            
            # Parse timestamp
            ts_str = message.get('timestamp') or message.get('ts')
            if isinstance(ts_str, str):
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                ts = datetime.now()
            
            source = message.get('source', 'unknown')
            class_name = message.get('class') or message.get('class_name', 'unknown')
            confidence = float(message.get('confidence', 0.0))
            
            # Extract geospatial data if available
            lat = message.get('lat') or message.get('latitude')
            lon = message.get('lon') or message.get('longitude')
            
            # Extract additional metadata
            metadata = message.get('metadata', {})
            
            # Handle bbox if present (for object detection)
            if 'bbox' in message:
                metadata['bbox'] = message['bbox']
            
            embedding_uri = message.get('embedding_uri')
            
            return DetectionEvent(
                id=detection_id,
                ts=ts,
                source=source,
                class_name=class_name,
                confidence=confidence,
                lat=lat,
                lon=lon,
                embedding_uri=embedding_uri,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to process detection event: {e}")
            logger.debug(f"Message: {message}")
            return None
    
    def insert_detection(self, detection: DetectionEvent) -> bool:
        """
        Insert detection event into PostGIS database.
        
        Args:
            detection: DetectionEvent object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Build geometry from lat/lon if available
            geom_wkt = None
            if detection.lat is not None and detection.lon is not None:
                geom_wkt = f"SRID=4326;POINT({detection.lon} {detection.lat})"
            
            # Insert detection
            # Handle metadata - use Json if available, otherwise pass dict
            metadata_value = Json(detection.metadata or {}) if Json else detection.metadata or {}
            
            cursor.execute(
                """
                INSERT INTO detections 
                (id, ts, source, class, confidence, geom, embedding_uri, metadata)
                VALUES (%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s), %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    ts = EXCLUDED.ts,
                    confidence = EXCLUDED.confidence,
                    metadata = EXCLUDED.metadata
                """,
                (
                    detection.id,
                    detection.ts,
                    detection.source,
                    detection.class_name,
                    detection.confidence,
                    geom_wkt,
                    detection.embedding_uri,
                    metadata_value
                )
            )
            
            self.db_conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert detection: {e}")
            self.db_conn.rollback()
            return False
    
    def process_forecast_event(self, message: Dict[str, Any]) -> bool:
        """
        Process and store forecast/simulation event.
        
        Args:
            message: Raw message from Kafka
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db_conn.cursor()
            
            request_id = message.get('request_id') or str(uuid.uuid4())
            results_uri = message.get('results_uri', '')
            meta = message.get('meta', {})
            status = message.get('status', 'completed')
            
            meta_value = Json(meta) if Json else meta
            
            cursor.execute(
                """
                INSERT INTO forecasts (request_id, results_uri, meta, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET
                    results_uri = EXCLUDED.results_uri,
                    meta = EXCLUDED.meta,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                """,
                (request_id, results_uri, meta_value, status)
            )
            
            self.db_conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to process forecast event: {e}")
            self.db_conn.rollback()
            return False
    
    def process_embedding_event(self, message: Dict[str, Any]) -> bool:
        """
        Process and store embedding event.
        
        Args:
            message: Raw message from Kafka
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db_conn.cursor()
            
            embedding_id = message.get('id') or str(uuid.uuid4())
            session_id = message.get('session_id')
            detection_id = message.get('detection_id')
            embedding_vector = message.get('embedding_vector', [])
            text_content = message.get('text_content', '')
            metadata = message.get('metadata', {})
            
            metadata_value = Json(metadata) if Json else metadata
            
            cursor.execute(
                """
                INSERT INTO embeddings 
                (id, session_id, detection_id, embedding_vector, text_content, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding_vector = EXCLUDED.embedding_vector,
                    text_content = EXCLUDED.text_content,
                    metadata = EXCLUDED.metadata
                """,
                (
                    embedding_id,
                    session_id,
                    detection_id,
                    embedding_vector,
                    text_content,
                    metadata_value
                )
            )
            
            self.db_conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to process embedding event: {e}")
            self.db_conn.rollback()
            return False
    
    def process_message(self, topic: str, message: Dict[str, Any]):
        """
        Route message to appropriate handler based on topic.
        
        Args:
            topic: Kafka topic name
            message: Message payload
        """
        try:
            if topic == "detections":
                detection = self.process_detection_event(message)
                if detection:
                    success = self.insert_detection(detection)
                    if success:
                        logger.debug(
                            f"Stored detection {detection.id} "
                            f"(class={detection.class_name}, "
                            f"confidence={detection.confidence:.2f})"
                        )
            
            elif topic == "simulations":
                success = self.process_forecast_event(message)
                if success:
                    logger.debug(f"Stored forecast event")
            
            elif topic == "frames.embeddings":
                success = self.process_embedding_event(message)
                if success:
                    logger.debug(f"Stored embedding event")
            
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from {topic}: {e}")
    
    def start(self):
        """Start consuming messages from Kafka."""
        self.running = True
        
        # Connect to services
        self.connect_database()
        self.connect_kafka()
        
        logger.info("Starting message consumption...")
        
        try:
            message_count = 0
            for message in self.consumer:
                if not self.running:
                    break
                
                self.process_message(message.topic, message.value)
                message_count += 1
                
                if message_count % 100 == 0:
                    logger.info(f"Processed {message_count} messages")
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the consumer and close connections."""
        logger.info("Shutting down consumer...")
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")
        
        logger.info("Consumer stopped")


def main():
    """Main entry point for running the ingestion consumer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GUIRA Detection Ingestion Consumer"
    )
    parser.add_argument(
        '--kafka-bootstrap-servers',
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers (comma-separated)'
    )
    parser.add_argument(
        '--postgres-conn',
        default=os.getenv('POSTGRES_CONN'),
        help='PostgreSQL connection string'
    )
    parser.add_argument(
        '--consumer-group',
        default='guira-ingestion',
        help='Kafka consumer group ID'
    )
    parser.add_argument(
        '--topics',
        nargs='+',
        default=['detections', 'frames.embeddings', 'simulations'],
        help='Kafka topics to subscribe to'
    )
    
    args = parser.parse_args()
    
    consumer = DetectionIngestionConsumer(
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        postgres_conn_string=args.postgres_conn,
        consumer_group=args.consumer_group,
        topics=args.topics
    )
    
    consumer.start()


if __name__ == "__main__":
    main()
