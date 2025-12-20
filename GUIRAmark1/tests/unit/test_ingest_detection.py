"""
Unit tests for detection ingestion consumer.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'integrations' / 'guira_core'))

try:
    from data.ingest.ingest_detection import (
        DetectionIngestionConsumer,
        DetectionEvent
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipIf(not IMPORTS_AVAILABLE, f"Required imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestDetectionEvent(unittest.TestCase):
    """Test DetectionEvent data structure."""
    
    def test_detection_event_creation(self):
        """Test creating a DetectionEvent."""
        event = DetectionEvent(
            id="test-001",
            ts=datetime.now(),
            source="yolov8",
            class_name="fire",
            confidence=0.95,
            lat=40.7128,
            lon=-74.0060,
            embedding_uri="s3://bucket/embedding.npy",
            metadata={"frame_id": 123}
        )
        
        self.assertEqual(event.id, "test-001")
        self.assertEqual(event.source, "yolov8")
        self.assertEqual(event.class_name, "fire")
        self.assertEqual(event.confidence, 0.95)
        self.assertEqual(event.lat, 40.7128)
        self.assertEqual(event.lon, -74.0060)
        self.assertIsNotNone(event.metadata)


@unittest.skipIf(not IMPORTS_AVAILABLE, f"Required imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestDetectionIngestionConsumer(unittest.TestCase):
    """Test DetectionIngestionConsumer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kafka_servers = "localhost:9092"
        self.postgres_conn = "postgresql://test:test@localhost:5432/test"
        self.consumer_group = "test-group"
        self.topics = ["detections", "simulations"]
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_consumer_initialization(self, mock_kafka, mock_psycopg2):
        """Test consumer initialization."""
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn,
            consumer_group=self.consumer_group,
            topics=self.topics
        )
        
        self.assertEqual(consumer.kafka_bootstrap_servers, self.kafka_servers)
        self.assertEqual(consumer.postgres_conn_string, self.postgres_conn)
        self.assertEqual(consumer.consumer_group, self.consumer_group)
        self.assertEqual(consumer.topics, self.topics)
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_detection_event(self, mock_kafka, mock_psycopg2):
        """Test processing a detection event message."""
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        
        message = {
            'id': 'test-detection-001',
            'timestamp': '2024-01-15T10:30:00',
            'source': 'yolov8-fire',
            'class': 'fire',
            'confidence': 0.92,
            'lat': 40.7128,
            'lon': -74.0060,
            'metadata': {'frame_id': 1234}
        }
        
        detection = consumer.process_detection_event(message)
        
        self.assertIsNotNone(detection)
        self.assertEqual(detection.id, 'test-detection-001')
        self.assertEqual(detection.source, 'yolov8-fire')
        self.assertEqual(detection.class_name, 'fire')
        self.assertEqual(detection.confidence, 0.92)
        self.assertEqual(detection.lat, 40.7128)
        self.assertEqual(detection.lon, -74.0060)
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_detection_event_with_bbox(self, mock_kafka, mock_psycopg2):
        """Test processing detection with bounding box."""
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        
        message = {
            'timestamp': '2024-01-15T10:30:00',
            'source': 'yolov8-fire',
            'class_name': 'smoke',
            'confidence': 0.87,
            'bbox': [100, 100, 200, 200]
        }
        
        detection = consumer.process_detection_event(message)
        
        self.assertIsNotNone(detection)
        self.assertEqual(detection.class_name, 'smoke')
        self.assertIn('bbox', detection.metadata)
        self.assertEqual(detection.metadata['bbox'], [100, 100, 200, 200])
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_detection_event_missing_fields(self, mock_kafka, mock_psycopg2):
        """Test processing detection with missing optional fields."""
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        
        message = {
            'source': 'test-source',
            'confidence': 0.75
        }
        
        detection = consumer.process_detection_event(message)
        
        self.assertIsNotNone(detection)
        self.assertEqual(detection.source, 'test-source')
        self.assertEqual(detection.confidence, 0.75)
        self.assertEqual(detection.class_name, 'unknown')
        self.assertIsNone(detection.lat)
        self.assertIsNone(detection.lon)
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_insert_detection(self, mock_kafka, mock_psycopg2):
        """Test inserting detection into database."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        consumer.db_conn = mock_conn
        
        detection = DetectionEvent(
            id="test-001",
            ts=datetime.now(),
            source="yolov8",
            class_name="fire",
            confidence=0.95,
            lat=40.7128,
            lon=-74.0060,
            metadata={"test": "data"}
        )
        
        result = consumer.insert_detection(detection)
        
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_forecast_event(self, mock_kafka, mock_psycopg2):
        """Test processing forecast/simulation event."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        consumer.db_conn = mock_conn
        
        message = {
            'request_id': 'forecast-001',
            'results_uri': 's3://bucket/forecast-001.json',
            'meta': {'model': 'physx', 'duration': 3600},
            'status': 'completed'
        }
        
        result = consumer.process_forecast_event(message)
        
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_embedding_event(self, mock_kafka, mock_psycopg2):
        """Test processing embedding event."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        consumer.db_conn = mock_conn
        
        message = {
            'id': 'embedding-001',
            'session_id': 'session-123',
            'detection_id': 'detection-456',
            'embedding_vector': [0.1, 0.2, 0.3],
            'text_content': 'Fire detected in sector A',
            'metadata': {'source': 'dinov2'}
        }
        
        result = consumer.process_embedding_event(message)
        
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('data.ingest.ingest_detection.psycopg2')
    @patch('data.ingest.ingest_detection.KafkaConsumer')
    def test_process_message_routing(self, mock_kafka, mock_psycopg2):
        """Test message routing to appropriate handlers."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        consumer = DetectionIngestionConsumer(
            kafka_bootstrap_servers=self.kafka_servers,
            postgres_conn_string=self.postgres_conn
        )
        consumer.db_conn = mock_conn
        
        # Test detection topic
        detection_msg = {
            'timestamp': '2024-01-15T10:30:00',
            'source': 'test',
            'class': 'fire',
            'confidence': 0.9
        }
        consumer.process_message('detections', detection_msg)
        
        # Test simulations topic
        forecast_msg = {
            'request_id': 'test-001',
            'results_uri': 's3://test',
            'status': 'completed'
        }
        consumer.process_message('simulations', forecast_msg)
        
        # Test embeddings topic
        embedding_msg = {
            'id': 'emb-001',
            'embedding_vector': [0.1, 0.2],
            'text_content': 'test'
        }
        consumer.process_message('frames.embeddings', embedding_msg)
        
        # Should have called execute 3 times (one for each message)
        self.assertEqual(mock_cursor.execute.call_count, 3)


class TestWithoutDependencies(unittest.TestCase):
    """Tests that can run without external dependencies."""
    
    def test_imports_status(self):
        """Test whether required dependencies are available."""
        if IMPORTS_AVAILABLE:
            self.assertTrue(True, "All dependencies available")
        else:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")


if __name__ == '__main__':
    unittest.main()
