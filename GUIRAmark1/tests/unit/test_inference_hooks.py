"""
Unit tests for inference hooks module.
Tests all model inference functions with mock data.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open

# Add src to path for testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from inference_hooks import (
    FireDetectionInference, SmokeDetectionInference, 
    FaunaDetectionInference, VegetationHealthInference,
    FireSpreadPredictor, detect_fire, classify_smoke_clip,
    detect_fauna, classify_veg, predict_spread
)

class TestFireDetectionInference(unittest.TestCase):
    """Test fire detection inference class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "models/fire_yolov8/best.pt"
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    def test_init_with_fusion(self):
        """Test FireDetectionInference initialization with fusion enabled"""
        detector = FireDetectionInference(self.model_path, use_fusion=True)
        self.assertEqual(detector.model_path, self.model_path)
        self.assertTrue(detector.use_fusion)
        self.assertEqual(detector.device, 'cpu')
        
    def test_init_without_fusion(self):
        """Test FireDetectionInference initialization without fusion"""
        detector = FireDetectionInference(self.model_path, use_fusion=False, device='cuda')
        self.assertFalse(detector.use_fusion)
        self.assertEqual(detector.device, 'cuda')
    
    def test_detect_fire_mock_results(self):
        """Test fire detection with mock results"""
        detector = FireDetectionInference(self.model_path)
        
        result = detector.detect_fire(self.test_frame, confidence_threshold=0.5)
        
        # Check result structure
        self.assertIn('boxes', result)
        self.assertIn('classes', result)
        self.assertIn('scores', result)
        self.assertIn('fusion_used', result)
        self.assertIn('detection_count', result)
        
        # Check types
        self.assertIsInstance(result['boxes'], list)
        self.assertIsInstance(result['classes'], list)
        self.assertIsInstance(result['scores'], list)
        self.assertIsInstance(result['fusion_used'], bool)
        self.assertIsInstance(result['detection_count'], int)
        
        # Check consistency
        self.assertEqual(len(result['boxes']), result['detection_count'])
        self.assertEqual(len(result['classes']), result['detection_count'])
        self.assertEqual(len(result['scores']), result['detection_count'])
    
    def test_detect_fire_with_thermal(self):
        """Test fire detection with thermal fusion"""
        detector = FireDetectionInference(self.model_path, use_fusion=True)
        
        result = detector.detect_fire(self.test_frame, thermal_frame=self.thermal_frame)
        
        self.assertIn('fusion_used', result)
        # When thermal frame is provided and fusion is enabled, should be True
        # (This tests the mock implementation logic)
    
    def test_detect_fire_confidence_threshold(self):
        """Test confidence threshold filtering"""
        detector = FireDetectionInference(self.model_path)
        
        # Test with high threshold
        result_high = detector.detect_fire(self.test_frame, confidence_threshold=0.9)
        result_low = detector.detect_fire(self.test_frame, confidence_threshold=0.1)
        
        # All scores should be above threshold (mock implementation guarantees this)
        for score in result_high['scores']:
            self.assertGreaterEqual(score, 0.9)
        
        for score in result_low['scores']:
            self.assertGreaterEqual(score, 0.1)

class TestSmokeDetectionInference(unittest.TestCase):
    """Test smoke detection inference class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "models/smoke_timesformer/best.pt"
        self.sequence_length = 8
        self.test_clip = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        self.test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test SmokeDetectionInference initialization"""
        detector = SmokeDetectionInference(self.model_path, sequence_length=self.sequence_length)
        self.assertEqual(detector.model_path, self.model_path)
        self.assertEqual(detector.sequence_length, self.sequence_length)
        self.assertEqual(len(detector.frame_buffer), 0)
    
    def test_classify_smoke_clip(self):
        """Test smoke classification on video clip"""
        detector = SmokeDetectionInference(self.model_path)
        
        probability = detector.classify_smoke_clip(self.test_clip)
        
        # Check result is a valid probability
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_classify_smoke_clip_short(self):
        """Test smoke classification with clip shorter than sequence length"""
        detector = SmokeDetectionInference(self.model_path, sequence_length=10)
        short_clip = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
        
        probability = detector.classify_smoke_clip(short_clip)
        
        # Should return 0.0 for clips that are too short
        self.assertEqual(probability, 0.0)
    
    def test_add_frame_buffer_fill(self):
        """Test frame addition and buffer management"""
        detector = SmokeDetectionInference(self.model_path, sequence_length=3)
        
        # Add frames one by one
        result1 = detector.add_frame(self.test_frame)
        self.assertIsNone(result1)  # Buffer not full yet
        
        result2 = detector.add_frame(self.test_frame)
        self.assertIsNone(result2)  # Still not full
        
        result3 = detector.add_frame(self.test_frame)
        self.assertIsNotNone(result3)  # Buffer is now full
        self.assertIsInstance(result3, float)
        
        # Add another frame - should still return result
        result4 = detector.add_frame(self.test_frame)
        self.assertIsNotNone(result4)
        
        # Buffer should maintain sequence length
        self.assertEqual(len(detector.frame_buffer), 3)

class TestFaunaDetectionInference(unittest.TestCase):
    """Test fauna detection inference class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.yolo_path = "models/fauna_yolov8_csrnet/yolo_best.pt"
        self.csrnet_path = "models/fauna_yolov8_csrnet/csrnet_best.pth"
        self.test_frame = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test FaunaDetectionInference initialization"""
        detector = FaunaDetectionInference(self.yolo_path, self.csrnet_path)
        self.assertEqual(detector.yolo_path, self.yolo_path)
        self.assertEqual(detector.csrnet_path, self.csrnet_path)
    
    def test_detect_fauna(self):
        """Test fauna detection"""
        detector = FaunaDetectionInference(self.yolo_path, self.csrnet_path)
        
        result = detector.detect_fauna(self.test_frame, confidence_threshold=0.5)
        
        # Check result structure
        expected_keys = ['boxes', 'species', 'scores', 'health_status', 'detection_count']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check types and consistency
        self.assertIsInstance(result['boxes'], list)
        self.assertIsInstance(result['species'], list)
        self.assertIsInstance(result['scores'], list)
        self.assertIsInstance(result['health_status'], list)
        
        detection_count = result['detection_count']
        self.assertEqual(len(result['boxes']), detection_count)
        self.assertEqual(len(result['species']), detection_count)
        self.assertEqual(len(result['scores']), detection_count)
        self.assertEqual(len(result['health_status']), detection_count)
        
        # Check valid species and health status
        valid_species = ['deer', 'elk', 'bear', 'bird', 'other']
        valid_health = ['healthy', 'distressed']
        
        for species in result['species']:
            self.assertIn(species, valid_species)
        
        for health in result['health_status']:
            self.assertIn(health, valid_health)
    
    def test_estimate_density(self):
        """Test density estimation"""
        detector = FaunaDetectionInference(self.yolo_path, self.csrnet_path)
        
        density_map = detector.estimate_density(self.test_frame)
        
        # Check shape and type
        self.assertIsInstance(density_map, np.ndarray)
        self.assertEqual(density_map.dtype, np.float32)
        
        # Check downsampling (should be input_size // 8)
        expected_shape = (self.test_frame.shape[0] // 8, self.test_frame.shape[1] // 8)
        self.assertEqual(density_map.shape, expected_shape)
        
        # Check values are non-negative
        self.assertTrue(np.all(density_map >= 0))

class TestVegetationHealthInference(unittest.TestCase):
    """Test vegetation health inference class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "models/vegetation_resnet_vari/best.pt"
        self.test_patch = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test VegetationHealthInference initialization"""
        detector = VegetationHealthInference(self.model_path)
        self.assertEqual(detector.model_path, self.model_path)
        self.assertEqual(detector.classes, ['healthy', 'dry', 'burned'])
    
    def test_compute_vari(self):
        """Test VARI index computation"""
        detector = VegetationHealthInference(self.model_path)
        
        # Test with known values
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 1] = 200  # High green
        test_image[:, :, 0] = 50   # Low red
        test_image[:, :, 2] = 30   # Low blue
        
        vari = detector.compute_vari(test_image)
        
        self.assertIsInstance(vari, float)
        # With high green and low red, VARI should be positive
        self.assertGreater(vari, 0)
    
    def test_compute_vari_edge_cases(self):
        """Test VARI computation edge cases"""
        detector = VegetationHealthInference(self.model_path)
        
        # Test with all zeros (should not crash)
        zero_image = np.zeros((50, 50, 3), dtype=np.uint8)
        vari_zero = detector.compute_vari(zero_image)
        self.assertIsInstance(vari_zero, float)
        
        # Test with single pixel
        single_pixel = np.array([[[100, 150, 50]]], dtype=np.uint8)
        vari_single = detector.compute_vari(single_pixel)
        self.assertIsInstance(vari_single, float)
        
        # Test with wrong shape (should raise error)
        wrong_shape = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detector.compute_vari(wrong_shape)
    
    def test_classify_veg(self):
        """Test vegetation classification"""
        detector = VegetationHealthInference(self.model_path)
        
        result = detector.classify_veg(self.test_patch)
        
        # Check result structure
        expected_keys = ['health_class', 'probabilities', 'vari_index', 'confidence']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check types
        self.assertIsInstance(result['health_class'], str)
        self.assertIsInstance(result['probabilities'], dict)
        self.assertIsInstance(result['vari_index'], float)
        self.assertIsInstance(result['confidence'], float)
        
        # Check valid health class
        self.assertIn(result['health_class'], detector.classes)
        
        # Check probabilities sum to 1 (approximately)
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

class TestFireSpreadPredictor(unittest.TestCase):
    """Test fire spread prediction class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "models/spread_hybrid/best.pt"
        self.test_sequence = np.random.rand(6, 64, 64)
        self.wind_data = {'speed': 15, 'direction': 45}
        self.terrain_data = {'elevation': np.random.rand(64, 64) * 100}
    
    def test_init(self):
        """Test FireSpreadPredictor initialization"""
        predictor = FireSpreadPredictor(self.model_path)
        self.assertEqual(predictor.model_path, self.model_path)
    
    def test_predict_spread(self):
        """Test fire spread prediction"""
        predictor = FireSpreadPredictor(self.model_path)
        
        future_masks = predictor.predict_spread(
            self.test_sequence, 
            wind_data=self.wind_data,
            terrain_data=self.terrain_data
        )
        
        # Check output shape
        self.assertIsInstance(future_masks, np.ndarray)
        expected_shape = (12, 64, 64)  # 12 future timesteps
        self.assertEqual(future_masks.shape, expected_shape)
        
        # Check values are between 0 and 1 (probabilities)
        self.assertTrue(np.all(future_masks >= 0))
        self.assertTrue(np.all(future_masks <= 1))
    
    def test_predict_spread_insufficient_history(self):
        """Test prediction with insufficient historical data"""
        predictor = FireSpreadPredictor(self.model_path)
        
        short_sequence = np.random.rand(3, 64, 64)  # Less than required 6
        
        with self.assertRaises(ValueError):
            predictor.predict_spread(short_sequence)
    
    def test_predict_spread_no_environmental_data(self):
        """Test prediction without environmental data"""
        predictor = FireSpreadPredictor(self.model_path)
        
        future_masks = predictor.predict_spread(self.test_sequence)
        
        # Should still work without environmental data
        self.assertIsInstance(future_masks, np.ndarray)
        self.assertEqual(future_masks.shape, (12, 64, 64))

class TestInferenceHooksFunctions(unittest.TestCase):
    """Test main inference hook functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_clip = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        self.test_patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.test_sequence = np.random.rand(6, 64, 64)
    
    def test_detect_fire_function(self):
        """Test detect_fire main function"""
        result = detect_fire(self.test_frame, use_fusion=True)
        
        self.assertIsInstance(result, dict)
        self.assertIn('boxes', result)
        self.assertIn('classes', result)
        self.assertIn('scores', result)
    
    def test_classify_smoke_clip_function(self):
        """Test classify_smoke_clip main function"""
        probability = classify_smoke_clip(self.test_clip)
        
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_detect_fauna_function(self):
        """Test detect_fauna main function"""
        detections, density_map = detect_fauna(self.test_frame)
        
        self.assertIsInstance(detections, dict)
        self.assertIsInstance(density_map, np.ndarray)
        self.assertIn('boxes', detections)
        self.assertIn('species', detections)
    
    def test_classify_veg_function(self):
        """Test classify_veg main function"""
        result = classify_veg(self.test_patch)
        
        self.assertIsInstance(result, dict)
        self.assertIn('health_class', result)
        self.assertIn('probabilities', result)
        self.assertIn('vari_index', result)
    
    def test_predict_spread_function(self):
        """Test predict_spread main function"""
        future_masks = predict_spread(self.test_sequence)
        
        self.assertIsInstance(future_masks, np.ndarray)
        self.assertEqual(future_masks.shape, (12, 64, 64))

if __name__ == '__main__':
    unittest.main()