"""
Integration tests for the complete fire prevention pipeline.
Tests end-to-end workflows and module interactions.
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import json
import time
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from inference_hooks import detect_fire, classify_smoke_clip, detect_fauna, classify_veg, predict_spread
from maps_adapter import create_fire_overlay, create_fauna_overlay, CameraIntrinsics, DronePose
from video_streaming import VideoStreamProcessor, StreamConfig, FrameResult

class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.thermal_frame = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        self.test_clip = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        
        self.drone_pose = DronePose(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            roll=2.5,
            pitch=-5.0,
            yaw=120.0,
            timestamp=time.time()
        )
        
        self.camera_intrinsics = CameraIntrinsics(
            fx=800.0, fy=800.0, cx=320.0, cy=320.0,
            k1=-0.1, k2=0.05
        )
    
    def test_fire_detection_to_geojson_pipeline(self):
        """Test fire detection → GeoJSON overlay pipeline"""
        
        # Step 1: Detect fire
        fire_detections = detect_fire(
            self.test_frame, 
            use_fusion=True, 
            thermal_frame=self.thermal_frame
        )
        
        # Verify fire detection results
        self.assertIn('boxes', fire_detections)
        self.assertIn('classes', fire_detections)
        self.assertIn('scores', fire_detections)
        self.assertIn('detection_count', fire_detections)
        
        # Step 2: Convert to GeoJSON overlay
        if fire_detections['detection_count'] > 0:
            geojson_overlay = create_fire_overlay(
                fire_detections, 
                self.drone_pose, 
                self.camera_intrinsics, 
                640, 640
            )
            
            # Verify GeoJSON structure
            self.assertIsInstance(geojson_overlay, str)
            overlay_data = json.loads(geojson_overlay)
            
            self.assertEqual(overlay_data['type'], 'FeatureCollection')
            self.assertIn('features', overlay_data)
            
            for feature in overlay_data['features']:
                self.assertEqual(feature['type'], 'Feature')
                self.assertEqual(feature['geometry']['type'], 'Polygon')
                self.assertIn('detection_type', feature['properties'])
                self.assertEqual(feature['properties']['detection_type'], 'fire_detection')
    
    def test_multi_modal_detection_pipeline(self):
        """Test multiple detection models working together"""
        
        # Run all detection models
        fire_result = detect_fire(self.test_frame)
        smoke_result = classify_smoke_clip(self.test_clip)
        fauna_result, density_map = detect_fauna(self.test_frame)
        veg_result = classify_veg(self.test_frame[:224, :224])  # Extract patch
        
        # Verify each model returns expected structure
        self.assertIsInstance(fire_result, dict)
        self.assertIn('boxes', fire_result)
        
        self.assertIsInstance(smoke_result, float)
        self.assertGreaterEqual(smoke_result, 0.0)
        self.assertLessEqual(smoke_result, 1.0)
        
        self.assertIsInstance(fauna_result, dict)
        self.assertIn('species', fauna_result)
        self.assertIsInstance(density_map, np.ndarray)
        
        self.assertIsInstance(veg_result, dict)
        self.assertIn('health_class', veg_result)
        self.assertIn('vari_index', veg_result)
        
        # Test decision fusion (simple example)
        fire_detected = fire_result['detection_count'] > 0
        smoke_detected = smoke_result > 0.5
        fauna_detected = fauna_result['detection_count'] > 0
        
        # Overall risk assessment
        risk_score = 0.0
        if fire_detected:
            risk_score += 0.6
        if smoke_detected:
            risk_score += 0.3
        if fauna_detected:
            risk_score += 0.1  # Wildlife presence
        
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
    
    def test_temporal_analysis_pipeline(self):
        """Test temporal analysis across multiple frames"""
        
        # Simulate sequence of frames
        frame_sequence = []
        smoke_probabilities = []
        fire_detections_sequence = []
        
        for i in range(10):
            # Generate slightly different frames
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            frame_sequence.append(frame)
            
            # Fire detection
            fire_result = detect_fire(frame)
            fire_detections_sequence.append(fire_result)
            
            # Smoke detection (accumulate over time)
            if len(frame_sequence) >= 8:
                recent_frames = frame_sequence[-8:]  # Last 8 frames
                smoke_clip = np.stack([cv2.resize(f, (224, 224)) for f in recent_frames])
                smoke_prob = classify_smoke_clip(smoke_clip)
                smoke_probabilities.append(smoke_prob)
        
        # Analyze temporal patterns
        fire_counts = [result['detection_count'] for result in fire_detections_sequence]
        
        # Check temporal consistency
        self.assertEqual(len(fire_detections_sequence), 10)
        self.assertGreater(len(smoke_probabilities), 0)
        
        # Simple temporal filtering example
        consistent_fire = sum(1 for count in fire_counts if count > 0) >= 3
        consistent_smoke = len(smoke_probabilities) > 0 and np.mean(smoke_probabilities) > 0.3
        
        # These are boolean checks that should not crash
        self.assertIsInstance(consistent_fire, bool)
        self.assertIsInstance(consistent_smoke, bool)
    
    @patch('cv2.resize')
    def test_spread_prediction_pipeline(self, mock_resize):
        """Test fire spread prediction pipeline"""
        # Mock cv2.resize to avoid import issues
        mock_resize.side_effect = lambda x, size: np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        
        # Create historical fire sequence
        historical_sequence = np.random.rand(6, 128, 128)  # 6 timesteps
        
        # Add some realistic fire patterns
        for t in range(6):
            # Simple growing fire
            center_size = 10 + t * 2
            center_y, center_x = 64, 64
            historical_sequence[t, 
                              center_y-center_size:center_y+center_size,
                              center_x-center_size:center_x+center_size] = 0.8
        
        # Predict future spread
        future_spread = predict_spread(
            historical_sequence,
            wind_data={'speed': 20, 'direction': 45},
            terrain_data={'elevation': np.random.rand(128, 128) * 100}
        )
        
        # Verify prediction structure
        self.assertEqual(future_spread.shape, (12, 128, 128))
        self.assertTrue(np.all(future_spread >= 0))
        self.assertTrue(np.all(future_spread <= 1))
        
        # Check monotonic growth (fire doesn't disappear)
        for t in range(11):
            current_area = np.sum(future_spread[t] > 0.1)
            next_area = np.sum(future_spread[t+1] > 0.1)
            # Allow small decreases due to randomness in mock model
            self.assertGreaterEqual(next_area, current_area * 0.9)

class TestVideoStreamIntegration(unittest.TestCase):
    """Test video streaming integration with detection models"""
    
    def setUp(self):
        """Set up streaming test fixtures"""
        self.config = StreamConfig(
            target_fps=15.0,
            enable_fire_detection=True,
            enable_smoke_detection=True,
            enable_fauna_detection=False,  # Keep disabled for performance
            enable_vegetation_analysis=False,
            max_queue_size=3,
            processing_threads=1  # Single thread for testing
        )
        
        self.intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
        self.results_received = []
        
        def result_callback(result):
            self.results_received.append(result)
        
        self.processor = VideoStreamProcessor(
            self.config, 
            self.intrinsics, 
            result_callback
        )
    
    def test_streaming_detection_integration(self):
        """Test streaming with actual detection processing"""
        self.processor.start()
        
        try:
            # Submit sequence of frames
            frames_submitted = 0
            for i in range(5):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                drone_pose = DronePose(
                    latitude=37.7749 + i * 0.0001,  # Slightly moving
                    longitude=-122.4194 + i * 0.0001,
                    altitude=100.0,
                    roll=0.0, pitch=0.0, yaw=i * 10.0,
                    timestamp=time.time() + i
                )
                
                success = self.processor.process_frame(frame, drone_pose=drone_pose)
                if success:
                    frames_submitted += 1
                
                time.sleep(0.1)  # 10 FPS submission rate
            
            # Wait for processing to complete
            time.sleep(1.0)
            
            # Check results
            self.assertGreater(frames_submitted, 0)
            
            # May have received some results
            for result in self.results_received:
                self.assertIsInstance(result, FrameResult)
                self.assertIsNotNone(result.fire_detections)
                self.assertGreaterEqual(result.processing_time, 0.0)
        
        finally:
            self.processor.stop()
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring during streaming"""
        self.processor.start()
        
        try:
            # Submit frames and monitor performance
            start_time = time.time()
            
            for i in range(3):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                self.processor.process_frame(frame)
                time.sleep(0.1)
            
            # Check performance stats
            stats = self.processor.get_performance_stats()
            
            self.assertIn('current_fps', stats)
            self.assertIn('frames_processed', stats)
            self.assertIn('running', stats)
            
            self.assertTrue(stats['running'])
            self.assertGreaterEqual(stats['frames_processed'], 0)
            
        finally:
            self.processor.stop()

class TestDataFlowIntegration(unittest.TestCase):
    """Test data flow between different system components"""
    
    def test_coordinate_projection_chain(self):
        """Test coordinate projection from pixels to geographic coordinates"""
        
        # Start with detection in pixel coordinates
        fire_detections = {
            'boxes': [[100, 150, 200, 250], [300, 300, 400, 400]],
            'classes': [0, 1],  # fire, smoke
            'scores': [0.85, 0.72],
            'detection_count': 2
        }
        
        # Define camera and pose
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        drone_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=150.0,
            roll=0.0, pitch=-15.0, yaw=90.0
        )
        
        # Convert to geographic overlay
        geojson_str = create_fire_overlay(fire_detections, drone_pose, intrinsics, 640, 480)
        geojson_data = json.loads(geojson_str)
        
        # Verify coordinate transformation
        for feature in geojson_data['features']:
            coordinates = feature['geometry']['coordinates'][0]  # Polygon coordinates
            
            # Check each coordinate pair
            for lon, lat in coordinates:
                # Should be near drone position (within reasonable distance)
                self.assertTrue(abs(lon - drone_pose.longitude) < 0.01)  # ~1 km
                self.assertTrue(abs(lat - drone_pose.latitude) < 0.01)
                
                # Should be valid geographic coordinates
                self.assertGreaterEqual(lon, -180.0)
                self.assertLessEqual(lon, 180.0)
                self.assertGreaterEqual(lat, -90.0)
                self.assertLessEqual(lat, 90.0)
    
    def test_multi_sensor_fusion(self):
        """Test fusion of RGB and thermal sensor data"""
        
        # RGB frame
        rgb_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Thermal frame (single channel)
        thermal_frame = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        
        # Test fusion in fire detection
        fire_result_rgb_only = detect_fire(rgb_frame, use_fusion=False)
        fire_result_fused = detect_fire(rgb_frame, use_fusion=True, thermal_frame=thermal_frame)
        
        # Both should return valid results
        for result in [fire_result_rgb_only, fire_result_fused]:
            self.assertIn('boxes', result)
            self.assertIn('classes', result)
            self.assertIn('scores', result)
            self.assertIn('fusion_used', result)
        
        # Fusion flag should be different
        self.assertFalse(fire_result_rgb_only['fusion_used'])
        self.assertTrue(fire_result_fused['fusion_used'])
    
    def test_error_propagation(self):
        """Test how errors propagate through the pipeline"""
        
        # Test with invalid inputs
        invalid_frames = [
            None,  # None input
            np.array([]),  # Empty array
            np.random.randint(0, 255, (10, 10), dtype=np.uint8),  # Wrong shape
            np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8),  # Wrong channels
        ]
        
        for invalid_frame in invalid_frames:
            try:
                # Should either handle gracefully or raise appropriate exception
                result = detect_fire(invalid_frame)
                
                # If it returns a result, should be properly structured
                if result is not None:
                    self.assertIn('boxes', result)
                    self.assertIn('detection_count', result)
                    
            except (ValueError, TypeError, AttributeError) as e:
                # Expected exceptions are acceptable
                self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
            
            except Exception as e:
                # Unexpected exceptions should be investigated
                self.fail(f"Unexpected exception with invalid input: {e}")

class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration handling across modules"""
    
    def test_consistent_model_paths(self):
        """Test that model paths are consistently handled"""
        
        # Test different model path formats
        model_paths = [
            "models/fire_yolov8/best.pt",
            "/absolute/path/to/model.pt",
            "relative/model.pt",
            "nonexistent/model.pt"
        ]
        
        for path in model_paths:
            try:
                # Should not crash during initialization
                result = detect_fire(
                    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    model_path=path
                )
                
                # Should return valid structure even with nonexistent model
                self.assertIn('boxes', result)
                
            except Exception as e:
                # Some paths might cause expected exceptions
                pass
    
    def test_parameter_validation(self):
        """Test parameter validation across modules"""
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test confidence threshold validation
        for threshold in [0.0, 0.5, 1.0]:
            result = detect_fire(frame, confidence_threshold=threshold)
            
            # All scores should be above threshold
            for score in result['scores']:
                self.assertGreaterEqual(score, threshold)
        
        # Test invalid thresholds
        for invalid_threshold in [-0.1, 1.1, None, "invalid"]:
            try:
                result = detect_fire(frame, confidence_threshold=invalid_threshold)
                # Should either work with default or handle gracefully
            except (ValueError, TypeError):
                # Expected for invalid values
                pass

class TestResourceManagement(unittest.TestCase):
    """Test resource management and cleanup"""
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during processing"""
        
        # Process sequence of frames and monitor memory usage
        initial_arrays = len(gc.get_objects())
        
        for i in range(10):
            frame = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)
            
            # Run all detection models
            fire_result = detect_fire(frame)
            smoke_clip = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
            smoke_result = classify_smoke_clip(smoke_clip)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_arrays = len(gc.get_objects())
        
        # Memory shouldn't grow excessively
        memory_growth = final_arrays - initial_arrays
        self.assertLess(memory_growth, 1000)  # Reasonable threshold
    
    def test_concurrent_access(self):
        """Test concurrent access to detection models"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_thread(thread_id):
            try:
                for i in range(3):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    result = detect_fire(frame)
                    results_queue.put((thread_id, i, result))
                    time.sleep(0.01)  # Small delay
                    
            except Exception as e:
                errors_queue.put((thread_id, e))
        
        # Start multiple worker threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join(timeout=10.0)
        
        # Check results
        self.assertTrue(errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}")
        
        # Should have received results from all threads
        results_count = results_queue.qsize()
        self.assertEqual(results_count, 9)  # 3 threads * 3 frames each

if __name__ == '__main__':
    # Import additional modules for memory testing
    import gc
    import cv2
    
    unittest.main()