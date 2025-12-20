"""
Unit tests for TimeSFormer Smoke Detection Probe

Tests the FastAPI service endpoints, temporal analysis, and embedding integration.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import io
import tempfile
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestTimeSFormerProbe(unittest.TestCase):
    """Test cases for TimeSFormer smoke detection probe."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set environment variables for testing
        os.environ["TIMESFORMER_MODEL_PATH"] = "models/timesformer_smoke.pt"
        os.environ["SEQUENCE_LENGTH"] = "8"
        os.environ["FRAME_SIZE"] = "224"
        os.environ["SMOKE_CONF_THRESHOLD"] = "0.5"
        os.environ["USE_EMBEDDING_FUSION"] = "false"
        
        # Mock torch
        sys.modules['torch'] = MagicMock()
        
        # Import app after setting up mocks
        from app import app
        self.client = TestClient(app)
        
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("sequence_length", data)
        self.assertIn("frame_size", data)
        
    def test_analyze_endpoint_no_file(self):
        """Test /analyze endpoint without file upload."""
        response = self.client.post("/analyze")
        self.assertEqual(response.status_code, 400)
        
    def test_analyze_sequence_endpoint(self):
        """Test /analyze_sequence endpoint with image sequence."""
        # Create test images
        images = []
        for i in range(4):
            img = Image.new('RGB', (224, 224), color=(100 + i * 30, 100, 100))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append(("files", (f"frame{i}.jpg", img_bytes, "image/jpeg")))
        
        with patch('app.get_timesformer_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = {
                "smoke_probability": 0.75,
                "frame_scores": [0.7, 0.75, 0.8, 0.72],
                "temporal_consistency": 0.05
            }
            mock_get_model.return_value = mock_model
            
            response = self.client.post("/analyze_sequence", files=images)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("smoke_detected", data)
            self.assertIn("confidence", data)
            self.assertIn("frame_scores", data)
            self.assertIn("temporal_features", data)
            self.assertIn("metadata", data)
            
            # Verify structure
            self.assertEqual(data["smoke_detected"], True)
            self.assertGreater(data["confidence"], 0.5)
            self.assertEqual(len(data["frame_scores"]), 4)
            
            # Verify temporal features
            temporal = data["temporal_features"]
            self.assertIn("temporal_consistency", temporal)
            self.assertIn("peak_frame_index", temporal)
            self.assertIn("peak_confidence", temporal)
    
    def test_analyze_sequence_no_files(self):
        """Test /analyze_sequence endpoint without files."""
        response = self.client.post("/analyze_sequence", files=[])
        self.assertEqual(response.status_code, 400)
    
    def test_mock_timesformer_model(self):
        """Test that mock model works correctly."""
        from app import MockTimeSFormer
        
        mock_model = MockTimeSFormer()
        frames = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        
        predictions = mock_model.predict(frames)
        
        self.assertIn("smoke_probability", predictions)
        self.assertIn("frame_scores", predictions)
        self.assertIn("temporal_consistency", predictions)
        
        # Check ranges
        self.assertGreaterEqual(predictions["smoke_probability"], 0.0)
        self.assertLessEqual(predictions["smoke_probability"], 1.0)
        self.assertEqual(len(predictions["frame_scores"]), 8)


class TestTemporalEmbedding(unittest.TestCase):
    """Test cases for temporal embedding aggregation."""
    
    def test_aggregate_temporal_embeddings(self):
        """Test temporal embedding aggregation."""
        from app import aggregate_temporal_embeddings
        
        # Create mock embeddings
        embeddings_list = [
            np.random.randn(256, 768) for _ in range(8)
        ]
        
        aggregated = aggregate_temporal_embeddings(embeddings_list)
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.shape, (256, 768))
    
    def test_aggregate_empty_embeddings(self):
        """Test aggregation with empty list."""
        from app import aggregate_temporal_embeddings
        
        result = aggregate_temporal_embeddings([])
        self.assertIsNone(result)
    
    def test_load_embedding_from_file_uri(self):
        """Test loading embeddings from file:// URI."""
        # Create a temporary .npz file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            embeddings = np.random.randn(256, 768)
            np.savez(tmp.name, embeddings=embeddings)
            tmp_path = tmp.name
        
        try:
            from app import load_embedding_from_uri
            
            # Test loading
            uri = f"file://{tmp_path}"
            loaded_embeddings = load_embedding_from_uri(uri)
            
            self.assertIsNotNone(loaded_embeddings)
            self.assertEqual(loaded_embeddings.shape, (256, 768))
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_embedding_invalid_uri(self):
        """Test loading embeddings with invalid URI."""
        from app import load_embedding_from_uri
        
        # Test with non-existent file
        result = load_embedding_from_uri("file:///non/existent/path.npz")
        self.assertIsNone(result)
        
        # Test with unsupported scheme
        result = load_embedding_from_uri("http://example.com/embed.npz")
        self.assertIsNone(result)


class TestImageSequenceLoading(unittest.TestCase):
    """Test cases for image sequence loading."""
    
    def test_load_image_sequence(self):
        """Test loading image sequence."""
        # Create mock upload files
        mock_files = []
        for i in range(4):
            img = Image.new('RGB', (640, 480), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            mock_file = MagicMock()
            mock_file.file = img_bytes
            mock_file.filename = f"frame{i}.jpg"
            mock_files.append(mock_file)
        
        from app import load_image_sequence
        
        frames = load_image_sequence(mock_files, max_frames=4)
        
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0].shape, (224, 224, 3))
    
    def test_load_image_sequence_sampling(self):
        """Test that image sequence is sampled when too many frames."""
        # Create 16 mock files
        mock_files = []
        for i in range(16):
            img = Image.new('RGB', (640, 480), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            mock_file = MagicMock()
            mock_file.file = img_bytes
            mock_file.filename = f"frame{i}.jpg"
            mock_files.append(mock_file)
        
        from app import load_image_sequence
        
        # Load with max 8 frames
        frames = load_image_sequence(mock_files, max_frames=8)
        
        # Should sample down to 8 frames
        self.assertEqual(len(frames), 8)


class TestVideoFrameLoading(unittest.TestCase):
    """Test cases for video frame loading."""
    
    @patch('cv2.VideoCapture')
    def test_load_video_frames_mock(self, mock_video_capture):
        """Test video frame loading with mocked OpenCV."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            1: 30,  # CAP_PROP_FRAME_COUNT
            5: 30.0  # CAP_PROP_FPS
        }.get(prop, 0)
        
        # Mock frame reading
        def mock_read():
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return True, frame
        
        mock_cap.read.side_effect = [mock_read() for _ in range(30)]
        mock_video_capture.return_value = mock_cap
        
        from app import load_video_frames
        
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
            frames = load_video_frames(tmp.name, max_frames=8)
            
            # Should load frames
            self.assertGreater(len(frames), 0)
            self.assertLessEqual(len(frames), 8)


if __name__ == "__main__":
    unittest.main()
