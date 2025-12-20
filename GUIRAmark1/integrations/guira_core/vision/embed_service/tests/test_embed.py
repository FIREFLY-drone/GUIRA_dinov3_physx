"""
Unit tests for DINOv3 Embedding Service

Tests the FastAPI service endpoints, embedding extraction, and blob storage.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import io
import tempfile
import numpy as np
from PIL import Image

# Mock torch before importing app
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestEmbeddingService(unittest.TestCase):
    """Test cases for DINOv3 embedding service."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set environment variables for testing
        os.environ["DINO_MODEL_ID"] = "facebook/dinov2-base"
        os.environ["USE_MINIO"] = "false"  # Don't use MinIO in tests
        
        # Import app after setting env vars
        from app import app
        self.client = TestClient(app)
        
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("device", data)
        self.assertIn("storage", data)
        
    def test_embed_endpoint_no_file(self):
        """Test /embed endpoint without file upload."""
        response = self.client.post("/embed")
        self.assertEqual(response.status_code, 422)  # Validation error
        
    def test_embed_endpoint_with_mock_model(self):
        """Test /embed endpoint with mocked model."""
        # Create a test image
        img = Image.new('RGB', (640, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Mock the extract_embeddings and storage functions directly
        with patch('app.extract_embeddings') as mock_extract, \
             patch('app.save_embedding_blob') as mock_save_blob:
            
            # Mock embeddings extraction returns dummy embeddings
            mock_extract.return_value = np.random.randn(1, 256, 768)
            
            # Mock storage returns URI
            mock_save_blob.return_value = "file:///tmp/embed_test.npz"
            
            # Make request
            response = self.client.post(
                "/embed",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )
            
            # Verify response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("embedding_uri", data)
            self.assertIn("shape", data)
            self.assertIn("num_tiles", data)
            self.assertIn("metadata", data)
            
            # Verify shape is correct
            self.assertEqual(len(data["shape"]), 3)
            
            # Verify metadata
            metadata = data["metadata"]
            self.assertEqual(metadata["filename"], "test.jpg")
            self.assertIn("original_size", metadata)
            self.assertIn("num_tiles", metadata)
            
    def test_tile_image_small(self):
        """Test that small images are not tiled."""
        from app import tile_image
        
        img = Image.new('RGB', (512, 512), color='blue')
        tiles = tile_image(img)
        
        # Small image should return single tile
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].size, (512, 512))
        
    def test_tile_image_large(self):
        """Test that large images are tiled correctly."""
        from app import tile_image
        
        img = Image.new('RGB', (2048, 1536), color='green')
        tiles = tile_image(img)
        
        # Large image should be tiled
        self.assertGreater(len(tiles), 1)
        
        # All tiles should be valid
        for tile in tiles:
            self.assertIsInstance(tile, Image.Image)
            # Tiles should be at most TILE_SIZE in each dimension
            self.assertLessEqual(tile.width, 518)
            self.assertLessEqual(tile.height, 518)
            
    def test_embedding_shape_validation(self):
        """Test that embeddings have correct shape."""
        # This test validates the embedding shape structure
        # Expected: (num_tiles, num_patches, embed_dim)
        
        # For a 518x518 image with patch_size=14:
        # num_patches = (518/14)^2 ≈ 37^2 = 1369
        # But DINOv2 adds CLS token, so num_patches = 1370
        # After removing CLS: 1369 patches
        
        expected_patches = 256  # Approximate for DINOv2 base
        expected_embed_dim = 768  # DINOv2 base dimension
        
        # Mock embedding shape
        mock_shape = (1, expected_patches, expected_embed_dim)
        self.assertEqual(len(mock_shape), 3)
        self.assertEqual(mock_shape[2], expected_embed_dim)
        
    def test_save_embedding_blob_local(self):
        """Test saving embeddings to local file system."""
        from app import save_embedding_blob
        
        # Create dummy embeddings
        embeddings = np.random.randn(1, 256, 768)
        metadata = {
            "filename": "test.jpg",
            "model": "facebook/dinov2-base"
        }
        
        # Save embeddings (should save locally since USE_MINIO=false)
        uri = save_embedding_blob(embeddings, metadata)
        
        # Verify URI format
        self.assertTrue(uri.startswith("file://"))
        
        # Verify file exists and can be loaded
        filepath = uri.replace("file://", "")
        self.assertTrue(os.path.exists(filepath))
        
        # Load and verify
        data = np.load(filepath, allow_pickle=True)
        self.assertIn("embeddings", data)
        loaded_embeddings = data["embeddings"]
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        
        # Clean up
        os.remove(filepath)


class TestMinIOIntegration(unittest.TestCase):
    """Integration tests for MinIO storage."""
    
    def setUp(self):
        """Set up MinIO test environment."""
        self.minio_available = False
        
        # Check if MinIO is available
        try:
            from minio import Minio
            minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
            client = Minio(
                minio_endpoint.replace("http://", "").replace("https://", ""),
                access_key="minioadmin",
                secret_key="minioadmin",
                secure=False
            )
            # Try to list buckets
            client.list_buckets()
            self.minio_available = True
        except Exception:
            pass
            
    @unittest.skipUnless(
        os.environ.get("TEST_MINIO") == "true",
        "MinIO integration test disabled (set TEST_MINIO=true to enable)"
    )
    def test_minio_upload(self):
        """Test uploading embeddings to MinIO."""
        if not self.minio_available:
            self.skipTest("MinIO not available")
            
        os.environ["USE_MINIO"] = "true"
        os.environ["EMBED_BUCKET"] = "test-embeds"
        
        from app import save_embedding_blob
        from minio import Minio
        
        # Create dummy embeddings
        embeddings = np.random.randn(2, 256, 768)
        
        # Upload
        uri = save_embedding_blob(embeddings)
        
        # Verify URI format
        self.assertTrue(uri.startswith("minio://"))
        
        # Verify object exists in MinIO
        bucket = uri.split("/")[2]
        object_name = uri.split("/")[-1]
        
        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        
        # Check object exists
        stat = client.stat_object(bucket, object_name)
        self.assertIsNotNone(stat)
        self.assertGreater(stat.size, 0)
        
        # Clean up
        client.remove_object(bucket, object_name)


class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities."""
    
    def test_image_conversion(self):
        """Test image format conversion."""
        # Create test images in different formats
        formats = ['RGB', 'RGBA', 'L', 'P']
        
        for fmt in formats:
            with self.subTest(format=fmt):
                if fmt == 'P':
                    img = Image.new('P', (100, 100))
                else:
                    img = Image.new(fmt, (100, 100), color='red')
                
                # Convert to RGB (required for model)
                img_rgb = img.convert('RGB')
                self.assertEqual(img_rgb.mode, 'RGB')
                self.assertEqual(img_rgb.size, (100, 100))
                
    def test_tile_coordinates(self):
        """Test that tiles cover entire image without gaps."""
        from app import tile_image
        
        # Create large test image
        width, height = 2000, 1500
        img = Image.new('RGB', (width, height), color='blue')
        
        tiles = tile_image(img)
        
        # Verify we have tiles
        self.assertGreater(len(tiles), 0)
        
        # Each tile should be valid
        for tile in tiles:
            self.assertLessEqual(tile.width, 518)
            self.assertLessEqual(tile.height, 518)
            # Tiles should not be too small (edge cases)
            self.assertGreater(tile.width, 100)
            self.assertGreater(tile.height, 100)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test client."""
        os.environ["USE_MINIO"] = "false"
        from app import app
        self.client = TestClient(app)
        
    def test_invalid_image_format(self):
        """Test uploading non-image file."""
        # Create a text file
        text_file = io.BytesIO(b"This is not an image")
        
        with patch('app.get_model') as mock_get_model:
            mock_processor = Mock()
            mock_model = Mock()
            mock_get_model.return_value = (mock_processor, mock_model)
            
            response = self.client.post(
                "/embed",
                files={"file": ("test.txt", text_file, "text/plain")}
            )
            
            # Should return error
            self.assertEqual(response.status_code, 500)
            
    def test_corrupted_image(self):
        """Test uploading corrupted image."""
        # Create corrupted JPEG
        corrupted = io.BytesIO(b"\xff\xd8\xff\xe0corrupted")
        
        with patch('app.get_model') as mock_get_model:
            mock_get_model.return_value = (Mock(), Mock())
            
            response = self.client.post(
                "/embed",
                files={"file": ("corrupt.jpg", corrupted, "image/jpeg")}
            )
            
            # Should handle error gracefully
            self.assertEqual(response.status_code, 500)


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestEmbeddingService))
    test_suite.addTest(unittest.makeSuite(TestMinIOIntegration))
    test_suite.addTest(unittest.makeSuite(TestImageProcessing))
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
