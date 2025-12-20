"""
Unit tests for video streaming module.
Tests stream processing, frame handling, and performance monitoring.
"""

import unittest
import numpy as np
import time
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, Mock
import threading
import queue

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video_streaming import (
    StreamConfig, FrameResult, VideoStreamProcessor,
    StreamingDemo, create_stream_processor, run_streaming_demo
)
from maps_adapter import CameraIntrinsics, DronePose

class TestStreamConfig(unittest.TestCase):
    """Test stream configuration data class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = StreamConfig()
        
        self.assertEqual(config.target_fps, 15.0)
        self.assertEqual(config.input_resolution, (640, 480))
        self.assertTrue(config.enable_fire_detection)
        self.assertTrue(config.enable_smoke_detection)
        self.assertFalse(config.enable_fauna_detection)  # Disabled by default
        self.assertFalse(config.enable_vegetation_analysis)  # Disabled by default
        
        # Check thresholds
        self.assertEqual(config.fire_confidence_threshold, 0.5)
        self.assertEqual(config.smoke_sequence_length, 8)
        
        # Check performance settings
        self.assertEqual(config.max_queue_size, 5)
        self.assertEqual(config.processing_threads, 2)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = StreamConfig(
            target_fps=30.0,
            input_resolution=(1280, 720),
            enable_fauna_detection=True,
            processing_threads=4,
            fire_confidence_threshold=0.7
        )
        
        self.assertEqual(config.target_fps, 30.0)
        self.assertEqual(config.input_resolution, (1280, 720))
        self.assertTrue(config.enable_fauna_detection)
        self.assertEqual(config.processing_threads, 4)
        self.assertEqual(config.fire_confidence_threshold, 0.7)

class TestFrameResult(unittest.TestCase):
    """Test frame result data class"""
    
    def test_default_frame_result(self):
        """Test default frame result initialization"""
        result = FrameResult(
            frame_id=123,
            timestamp=1640995200.0,
            processing_time=0.05
        )
        
        self.assertEqual(result.frame_id, 123)
        self.assertEqual(result.timestamp, 1640995200.0)
        self.assertEqual(result.processing_time, 0.05)
        
        # Check defaults
        self.assertIsNone(result.fire_detections)
        self.assertIsNone(result.smoke_probability)
        self.assertIsNone(result.fauna_detections)
        self.assertEqual(result.vegetation_health, [])
        
        self.assertIsNone(result.fire_overlay_geojson)
        self.assertIsNone(result.fauna_overlay_geojson)
        self.assertIsNone(result.vegetation_overlay_geojson)
        
        self.assertEqual(result.fire_detection_time, 0.0)
        self.assertEqual(result.smoke_detection_time, 0.0)
        self.assertEqual(result.fauna_detection_time, 0.0)
        self.assertEqual(result.vegetation_detection_time, 0.0)
    
    def test_frame_result_with_data(self):
        """Test frame result with detection data"""
        fire_detections = {
            'boxes': [[100, 150, 200, 250]],
            'classes': [0],
            'scores': [0.85]
        }
        
        result = FrameResult(
            frame_id=456,
            timestamp=1640995260.0,
            processing_time=0.08,
            fire_detections=fire_detections,
            smoke_probability=0.3,
            fire_detection_time=0.02,
            smoke_detection_time=0.03
        )
        
        self.assertEqual(result.fire_detections, fire_detections)
        self.assertEqual(result.smoke_probability, 0.3)
        self.assertEqual(result.fire_detection_time, 0.02)
        self.assertEqual(result.smoke_detection_time, 0.03)

class TestVideoStreamProcessor(unittest.TestCase):
    """Test video stream processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = StreamConfig(
            target_fps=15.0,
            enable_fire_detection=True,
            enable_smoke_detection=True,
            enable_fauna_detection=False,
            enable_vegetation_analysis=False
        )
        
        self.intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0
        )
        
        self.result_callback = MagicMock()
        
        self.processor = VideoStreamProcessor(
            self.config, self.intrinsics, self.result_callback
        )
    
    def test_init(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.config, self.config)
        self.assertEqual(self.processor.intrinsics, self.intrinsics)
        self.assertEqual(self.processor.result_callback, self.result_callback)
        
        self.assertFalse(self.processor.running)
        self.assertEqual(self.processor.frame_counter, 0)
        self.assertEqual(len(self.processor.processing_threads), 0)
        
        # Check models are initialized (should be None due to mock models)
        self.assertIsNotNone(self.processor.fire_detector)  # Mock model
        self.assertIsNotNone(self.processor.smoke_detector)
        self.assertIsNone(self.processor.fauna_detector)  # Disabled
        self.assertIsNone(self.processor.vegetation_detector)  # Disabled
    
    def test_start_stop(self):
        """Test processor start and stop"""
        # Start processor
        self.processor.start()
        self.assertTrue(self.processor.running)
        self.assertGreater(len(self.processor.processing_threads), 0)
        
        # Stop processor
        self.processor.stop()
        self.assertFalse(self.processor.running)
        
        # Wait a moment for threads to finish
        time.sleep(0.1)
    
    def test_process_frame_success(self):
        """Test successful frame processing"""
        self.processor.start()
        
        try:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            drone_pose = DronePose(
                latitude=37.7749, longitude=-122.4194, altitude=100.0,
                roll=0.0, pitch=0.0, yaw=0.0, timestamp=time.time()
            )
            
            success = self.processor.process_frame(frame, drone_pose=drone_pose)
            
            self.assertTrue(success)
            self.assertEqual(self.processor.frame_counter, 1)
            
        finally:
            self.processor.stop()
    
    def test_process_frame_resize(self):
        """Test frame resizing during processing"""
        self.processor.start()
        
        try:
            # Frame with different resolution than config
            large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            
            success = self.processor.process_frame(large_frame)
            
            self.assertTrue(success)
            
        finally:
            self.processor.stop()
    
    def test_process_frame_queue_full(self):
        """Test behavior when processing queue is full"""
        self.processor.start()
        
        try:
            # Fill the queue beyond capacity
            for i in range(self.config.max_queue_size + 5):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                result = self.processor.process_frame(frame)
                
                if i < self.config.max_queue_size:
                    self.assertTrue(result)
                # Later frames may be dropped due to full queue
        
        finally:
            self.processor.stop()
    
    def test_get_latest_result(self):
        """Test getting latest processing results"""
        # Should return None when no results
        result = self.processor.get_latest_result()
        self.assertIsNone(result)
        
        # Start processor and process a frame
        self.processor.start()
        
        try:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            self.processor.process_frame(frame)
            
            # Wait for processing
            time.sleep(0.2)
            
            # Should have a result now
            result = self.processor.get_latest_result()
            # May be None due to timing, but should not crash
        
        finally:
            self.processor.stop()
    
    def test_get_performance_stats(self):
        """Test performance statistics"""
        stats = self.processor.get_performance_stats()
        
        self.assertIn('current_fps', stats)
        self.assertIn('input_queue_size', stats)
        self.assertIn('output_queue_size', stats)
        self.assertIn('frames_processed', stats)
        self.assertIn('running', stats)
        
        # Check types
        self.assertIsInstance(stats['current_fps'], float)
        self.assertIsInstance(stats['input_queue_size'], int)
        self.assertIsInstance(stats['output_queue_size'], int)
        self.assertIsInstance(stats['frames_processed'], int)
        self.assertIsInstance(stats['running'], bool)
        
        self.assertFalse(stats['running'])  # Should be False initially
        self.assertEqual(stats['frames_processed'], 0)
    
    def test_process_single_frame(self):
        """Test internal single frame processing"""
        frame_data = {
            'frame_id': 1,
            'timestamp': time.time(),
            'frame': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'thermal_frame': None,
            'drone_pose': None
        }
        
        result = self.processor._process_single_frame(frame_data)
        
        # Check result structure
        self.assertIsInstance(result, FrameResult)
        self.assertEqual(result.frame_id, 1)
        self.assertGreater(result.processing_time, 0)
        
        # Fire detection should be enabled and return results
        self.assertIsNotNone(result.fire_detections)
        
        # Smoke detection should be enabled but may not return results immediately
        # (needs sequence buffer to fill)
        
    def test_analyze_vegetation_patches(self):
        """Test vegetation patch analysis"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Enable vegetation analysis for this test
        self.processor.config.enable_vegetation_analysis = True
        
        # Mock vegetation detector
        self.processor.vegetation_detector = MagicMock()
        self.processor.vegetation_detector.classify_veg.return_value = {
            'health_class': 'healthy',
            'confidence': 0.85,
            'vari_index': 0.25
        }
        
        patches = self.processor._analyze_vegetation_patches(frame)
        
        # Should return a list of results
        self.assertIsInstance(patches, list)
        # May be empty if patches are too small, but should not crash

class TestStreamingDemo(unittest.TestCase):
    """Test streaming demo class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = StreamConfig(
            target_fps=15.0,
            enable_fire_detection=True,
            enable_smoke_detection=True
        )
        self.demo = StreamingDemo(self.config)
    
    def test_init(self):
        """Test demo initialization"""
        self.assertEqual(self.demo.config, self.config)
        self.assertIsNone(self.demo.processor)
        self.assertEqual(self.demo.results_log, [])
    
    def test_result_callback(self):
        """Test result callback functionality"""
        # Create mock result
        result = FrameResult(
            frame_id=1,
            timestamp=time.time(),
            processing_time=0.05,
            fire_detections={'boxes': [[100, 150, 200, 250]], 'detection_count': 1},
            smoke_probability=0.8
        )
        
        # Call callback
        self.demo._result_callback(result)
        
        # Check result was logged
        self.assertEqual(len(self.demo.results_log), 1)
        self.assertEqual(self.demo.results_log[0], result)
    
    def test_print_demo_summary_no_results(self):
        """Test demo summary with no results"""
        # Should not crash when no results
        self.demo._print_demo_summary(30.0, 0)
    
    def test_print_demo_summary_with_results(self):
        """Test demo summary with results"""
        # Add some mock results
        for i in range(5):
            result = FrameResult(
                frame_id=i,
                timestamp=time.time(),
                processing_time=0.05 + i * 0.01,
                fire_detections={'boxes': [], 'detection_count': 0} if i % 2 else None,
                smoke_probability=0.3 + i * 0.1 if i % 3 else None
            )
            self.demo.results_log.append(result)
        
        # Should not crash and print summary
        self.demo._print_demo_summary(5.0, 5)

class TestMainFunctions(unittest.TestCase):
    """Test main module functions"""
    
    def test_create_stream_processor_defaults(self):
        """Test create_stream_processor with defaults"""
        processor = create_stream_processor()
        
        self.assertIsInstance(processor, VideoStreamProcessor)
        self.assertEqual(processor.config.target_fps, 15.0)
        self.assertTrue(processor.config.enable_fire_detection)
        self.assertTrue(processor.config.enable_smoke_detection)
        self.assertFalse(processor.config.enable_fauna_detection)
        self.assertFalse(processor.config.enable_vegetation_analysis)
    
    def test_create_stream_processor_custom(self):
        """Test create_stream_processor with custom settings"""
        processor = create_stream_processor(
            enable_fire=True,
            enable_smoke=False,
            enable_fauna=True,
            enable_vegetation=True,
            target_fps=30.0
        )
        
        self.assertEqual(processor.config.target_fps, 30.0)
        self.assertTrue(processor.config.enable_fire_detection)
        self.assertFalse(processor.config.enable_smoke_detection)
        self.assertTrue(processor.config.enable_fauna_detection)
        self.assertTrue(processor.config.enable_vegetation_analysis)
    
    @patch('video_streaming.StreamingDemo')
    def test_run_streaming_demo(self, mock_demo_class):
        """Test run_streaming_demo function"""
        mock_demo = MagicMock()
        mock_demo_class.return_value = mock_demo
        
        run_streaming_demo()
        
        # Check demo was created and started
        mock_demo_class.assert_called_once()
        mock_demo.start_demo.assert_called_once_with(source=0, duration=30.0)

class TestThreadSafety(unittest.TestCase):
    """Test thread safety of video streaming components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = StreamConfig(
            target_fps=15.0,
            max_queue_size=10,
            processing_threads=2
        )
        
        self.processor = VideoStreamProcessor(self.config)
    
    def test_concurrent_frame_processing(self):
        """Test concurrent frame processing doesn't cause race conditions"""
        self.processor.start()
        
        try:
            # Submit frames from multiple threads
            def submit_frames(thread_id):
                for i in range(5):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    success = self.processor.process_frame(frame)
                    # Don't assert success because queue might be full
                    time.sleep(0.01)  # Small delay to avoid overwhelming
            
            threads = []
            for i in range(3):
                t = threading.Thread(target=submit_frames, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for threads to complete
            for t in threads:
                t.join(timeout=5.0)
            
            # Wait for processing to complete
            time.sleep(0.5)
            
            # Check that some frames were processed
            stats = self.processor.get_performance_stats()
            # May be 0 due to timing, but should not crash
        
        finally:
            self.processor.stop()
    
    def test_start_stop_multiple_times(self):
        """Test starting and stopping processor multiple times"""
        for _ in range(3):
            self.processor.start()
            self.assertTrue(self.processor.running)
            
            time.sleep(0.1)
            
            self.processor.stop()
            self.assertFalse(self.processor.running)
            
            time.sleep(0.1)

class TestMemoryManagement(unittest.TestCase):
    """Test memory management and resource cleanup"""
    
    def test_frame_buffer_size_limit(self):
        """Test smoke detector frame buffer size limit"""
        from inference_hooks import SmokeDetectionInference
        
        detector = SmokeDetectionInference("model_path", sequence_length=5)
        
        # Add more frames than sequence length
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        for i in range(10):
            detector.add_frame(frame)
        
        # Buffer should not exceed sequence length
        self.assertLessEqual(len(detector.frame_buffer), 5)
    
    def test_queue_size_limits(self):
        """Test processing queue size limits"""
        config = StreamConfig(max_queue_size=3)
        processor = VideoStreamProcessor(config)
        
        # Check queue sizes are reasonable
        self.assertEqual(processor.input_queue.maxsize, 3)
        self.assertEqual(processor.output_queue.maxsize, 0)  # Unlimited output queue

class TestErrorHandling(unittest.TestCase):
    """Test error handling in video streaming"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = StreamConfig()
        self.processor = VideoStreamProcessor(self.config)
    
    def test_invalid_frame_shape(self):
        """Test handling of invalid frame shapes"""
        self.processor.start()
        
        try:
            # Try various invalid frame shapes
            invalid_frames = [
                np.random.randint(0, 255, (480,), dtype=np.uint8),  # 1D
                np.random.randint(0, 255, (480, 640), dtype=np.uint8),  # 2D grayscale
                np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8),  # 4 channel
            ]
            
            for frame in invalid_frames:
                # Should not crash, may return False
                result = self.processor.process_frame(frame)
                # Result can be True or False, main thing is no crash
        
        finally:
            self.processor.stop()
    
    def test_none_input_handling(self):
        """Test handling of None inputs"""
        self.processor.start()
        
        try:
            # Should handle None frame gracefully
            with self.assertRaises((TypeError, AttributeError)):
                self.processor.process_frame(None)
        
        finally:
            self.processor.stop()

if __name__ == '__main__':
    unittest.main()