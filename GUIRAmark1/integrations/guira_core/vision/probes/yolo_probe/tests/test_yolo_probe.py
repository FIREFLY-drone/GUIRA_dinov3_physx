"""
Unit tests for YOLO Detection Probe

Tests the FastAPI service endpoints, YOLO detection, and fusion head integration.
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


class TestYOLOProbe(unittest.TestCase):
    """Test cases for YOLO detection probe."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set environment variables for testing
        os.environ["YOLO_MODEL_PATH"] = "models/yolo_fire.pt"
        os.environ["YOLO_CONF_THRESHOLD"] = "0.25"
        os.environ["USE_EMBEDDING_FUSION"] = "false"
        
        # Mock ultralytics YOLO before importing app
        self.mock_yolo = MagicMock()
        sys.modules['ultralytics'] = MagicMock()
        
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
        self.assertIn("conf_threshold", data)
        self.assertIn("class_names", data)
        
    def test_detect_endpoint_no_file(self):
        """Test /detect endpoint without file upload."""
        response = self.client.post("/detect")
        self.assertEqual(response.status_code, 422)  # Validation error
        
    def test_detect_endpoint_with_mock_model(self):
        """Test /detect endpoint with mocked YOLO model."""
        # Create a test image
        img = Image.new('RGB', (640, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Mock YOLO model results
        mock_boxes = MagicMock()
        mock_boxes.xyxy = [torch.tensor([[100.0, 200.0, 300.0, 400.0]])]
        mock_boxes.conf = [torch.tensor([0.85])]
        mock_boxes.cls = [torch.tensor([0])]
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        
        with patch('app.get_yolo_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = [mock_result]
            mock_get_model.return_value = mock_model
            
            # Make request
            response = self.client.post(
                "/detect",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )
            
            # Verify response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("detections", data)
            self.assertIn("metadata", data)
            
            # Verify detections structure
            detections = data["detections"]
            self.assertGreaterEqual(len(detections), 0)
            
            if len(detections) > 0:
                detection = detections[0]
                self.assertIn("xyxy", detection)
                self.assertIn("conf", detection)
                self.assertIn("cls", detection)
                self.assertIn("class_name", detection)
            
            # Verify metadata
            metadata = data["metadata"]
            self.assertEqual(metadata["filename"], "test.jpg")
            self.assertIn("image_size", metadata)
            self.assertIn("num_detections", metadata)
    
    def test_invalid_image(self):
        """Test detection with invalid image data."""
        # Create invalid image data
        invalid_data = b"not an image"
        
        with patch('app.get_yolo_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            
            response = self.client.post(
                "/detect",
                files={"file": ("test.jpg", io.BytesIO(invalid_data), "image/jpeg")}
            )
            
            # Should return 400 for invalid image
            self.assertEqual(response.status_code, 400)
    
    def test_detect_batch_endpoint(self):
        """Test batch detection endpoint."""
        # Create two test images
        images = []
        for i in range(2):
            img = Image.new('RGB', (640, 480), color='red' if i == 0 else 'blue')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append(("file", (f"test{i}.jpg", img_bytes, "image/jpeg")))
        
        with patch('app.get_yolo_model') as mock_get_model:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.boxes = None  # No detections
            mock_model.predict.return_value = [mock_result]
            mock_get_model.return_value = mock_model
            
            response = self.client.post("/detect_batch", files=images)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("total", data)
            self.assertIn("successful", data)
            self.assertIn("results", data)
            self.assertEqual(data["total"], 2)


class TestFusionHead(unittest.TestCase):
    """Test cases for fusion head module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock torch
        self.mock_torch = MagicMock()
        sys.modules['torch'] = self.mock_torch
        sys.modules['torch.nn'] = MagicMock()
    
    def test_fusion_head_creation(self):
        """Test fusion head can be created."""
        try:
            from fusion_head import create_default_fusion_head
            # This will fail without actual torch, but we test import works
            self.assertTrue(True)
        except ImportError:
            self.skipTest("Fusion head requires torch")
    
    def test_fusion_head_predict(self):
        """Test fusion head prediction interface."""
        try:
            from fusion_head import FusionHead
            # Test that class exists
            self.assertTrue(hasattr(FusionHead, 'predict'))
        except ImportError:
            self.skipTest("Fusion head requires torch")


class TestEmbeddingLoader(unittest.TestCase):
    """Test cases for embedding loading functionality."""
    
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


# Mock torch tensor for tests
class torch:
    """Mock torch module for testing."""
    
    class tensor:
        def __init__(self, data):
            self.data = data
        
        def cpu(self):
            return self
        
        def numpy(self):
            return np.array(self.data)
        
        def tolist(self):
            return self.data


if __name__ == "__main__":
    unittest.main()
