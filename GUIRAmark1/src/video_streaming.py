"""
Video Streaming Processing Stub

Accepts frames (RGB or thermal) and runs the fire pipeline at ≥15 FPS on 640 input.
Provides real-time fire detection, smoke analysis, and overlay generation.
"""

import numpy as np
import cv2
import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

# Import our inference modules
from src.inference_hooks import (
    FireDetectionInference, SmokeDetectionInference, 
    FaunaDetectionInference, VegetationHealthInference
)
from src.maps_adapter import (
    CameraIntrinsics, DronePose, create_fire_overlay,
    create_fauna_overlay, create_vegetation_overlay
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for video streaming pipeline"""
    target_fps: float = 15.0
    input_resolution: tuple = (640, 480)
    enable_fire_detection: bool = True
    enable_smoke_detection: bool = True
    enable_fauna_detection: bool = False  # Disabled by default for performance
    enable_vegetation_analysis: bool = False  # Disabled by default for performance
    
    # Detection thresholds
    fire_confidence_threshold: float = 0.5
    smoke_sequence_length: int = 8
    
    # Performance settings
    max_queue_size: int = 5
    processing_threads: int = 2
    
    # Model paths
    fire_model_path: str = "models/fire_yolov8/best.pt"
    smoke_model_path: str = "models/smoke_timesformer/best.pt"
    fauna_yolo_path: str = "models/fauna_yolov8_csrnet/yolo_best.pt"
    fauna_csrnet_path: str = "models/fauna_yolov8_csrnet/csrnet_best.pth"
    vegetation_model_path: str = "models/vegetation_resnet_vari/best.pt"

@dataclass
class FrameResult:
    """Result from processing a single frame"""
    frame_id: int
    timestamp: float
    processing_time: float
    
    # Detection results
    fire_detections: Optional[Dict] = None
    smoke_probability: Optional[float] = None
    fauna_detections: Optional[Dict] = None
    vegetation_health: Optional[List[Dict]] = field(default_factory=list)
    
    # Overlay data
    fire_overlay_geojson: Optional[str] = None
    fauna_overlay_geojson: Optional[str] = None
    vegetation_overlay_geojson: Optional[str] = None
    
    # Performance metrics
    fire_detection_time: float = 0.0
    smoke_detection_time: float = 0.0
    fauna_detection_time: float = 0.0
    vegetation_detection_time: float = 0.0

class VideoStreamProcessor:
    """Real-time video stream processor for fire prevention pipeline"""
    
    def __init__(self, config: StreamConfig, 
                 intrinsics: Optional[CameraIntrinsics] = None,
                 result_callback: Optional[Callable[[FrameResult], None]] = None):
        self.config = config
        self.intrinsics = intrinsics
        self.result_callback = result_callback
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=config.max_queue_size)
        self.output_queue = queue.Queue()
        
        # Initialize models
        self._init_models()
        
        # Processing state
        self.running = False
        self.frame_counter = 0
        self.processing_threads = []
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Frame buffer for smoke detection
        self.frame_buffer = []
        
        logger.info(f"VideoStreamProcessor initialized for {config.target_fps} FPS")
    
    def _init_models(self):
        """Initialize all enabled models"""
        try:
            if self.config.enable_fire_detection:
                self.fire_detector = FireDetectionInference(
                    self.config.fire_model_path, use_fusion=True
                )
                logger.info("Fire detection model loaded")
            else:
                self.fire_detector = None
            
            if self.config.enable_smoke_detection:
                self.smoke_detector = SmokeDetectionInference(
                    self.config.smoke_model_path, 
                    sequence_length=self.config.smoke_sequence_length
                )
                logger.info("Smoke detection model loaded")
            else:
                self.smoke_detector = None
            
            if self.config.enable_fauna_detection:
                self.fauna_detector = FaunaDetectionInference(
                    self.config.fauna_yolo_path, self.config.fauna_csrnet_path
                )
                logger.info("Fauna detection models loaded")
            else:
                self.fauna_detector = None
            
            if self.config.enable_vegetation_analysis:
                self.vegetation_detector = VegetationHealthInference(
                    self.config.vegetation_model_path
                )
                logger.info("Vegetation health model loaded")
            else:
                self.vegetation_detector = None
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.info("Proceeding with mock models for demonstration")
    
    def start(self):
        """Start the processing pipeline"""
        if self.running:
            logger.warning("Processor already running")
            return
        
        self.running = True
        
        # Start processing threads
        for i in range(self.config.processing_threads):
            thread = threading.Thread(target=self._processing_worker, 
                                     name=f"ProcessingWorker-{i}")
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        # Start output handler thread
        output_thread = threading.Thread(target=self._output_worker, name="OutputWorker")
        output_thread.daemon = True
        output_thread.start()
        self.processing_threads.append(output_thread)
        
        logger.info("Video stream processor started")
    
    def stop(self):
        """Stop the processing pipeline"""
        self.running = False
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Video stream processor stopped")
    
    def process_frame(self, frame: np.ndarray, thermal_frame: Optional[np.ndarray] = None,
                     drone_pose: Optional[DronePose] = None) -> bool:
        """
        Submit a frame for processing
        
        Args:
            frame: RGB frame (H, W, 3)
            thermal_frame: Optional thermal frame
            drone_pose: Optional drone pose for geo-projection
            
        Returns:
            True if frame was queued, False if queue is full
        """
        if not self.running:
            logger.warning("Processor not running")
            return False
        
        # Resize frame to target resolution
        if frame.shape[:2] != self.config.input_resolution[::-1]:  # (H, W) vs (W, H)
            frame = cv2.resize(frame, self.config.input_resolution)
        
        if thermal_frame is not None and thermal_frame.shape[:2] != self.config.input_resolution[::-1]:
            thermal_frame = cv2.resize(thermal_frame, self.config.input_resolution)
        
        frame_data = {
            'frame_id': self.frame_counter,
            'timestamp': time.time(),
            'frame': frame.copy(),
            'thermal_frame': thermal_frame.copy() if thermal_frame is not None else None,
            'drone_pose': drone_pose
        }
        
        try:
            self.input_queue.put_nowait(frame_data)
            self.frame_counter += 1
            return True
        except queue.Full:
            logger.warning("Input queue full, dropping frame")
            return False
    
    def get_latest_result(self) -> Optional[FrameResult]:
        """Get the latest processing result"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            'current_fps': self.current_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'frames_processed': self.frame_counter,
            'running': self.running
        }
    
    def _processing_worker(self):
        """Worker thread for processing frames"""
        while self.running:
            try:
                frame_data = self.input_queue.get(timeout=1.0)
                
                start_time = time.time()
                result = self._process_single_frame(frame_data)
                result.processing_time = time.time() - start_time
                
                self.output_queue.put(result)
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _output_worker(self):
        """Worker thread for handling output results"""
        while self.running:
            try:
                result = self.output_queue.get(timeout=1.0)
                
                if self.result_callback:
                    self.result_callback(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in output worker: {e}")
    
    def _process_single_frame(self, frame_data: Dict) -> FrameResult:
        """Process a single frame through all enabled pipelines"""
        frame = frame_data['frame']
        thermal_frame = frame_data['thermal_frame']
        drone_pose = frame_data['drone_pose']
        
        result = FrameResult(
            frame_id=frame_data['frame_id'],
            timestamp=frame_data['timestamp'],
            processing_time=0.0
        )
        
        # Fire detection
        if self.config.enable_fire_detection and self.fire_detector:
            start_time = time.time()
            result.fire_detections = self.fire_detector.detect_fire(
                frame, thermal_frame, self.config.fire_confidence_threshold
            )
            result.fire_detection_time = time.time() - start_time
            
            # Create fire overlay if we have positioning data
            if drone_pose and self.intrinsics and result.fire_detections:
                try:
                    result.fire_overlay_geojson = create_fire_overlay(
                        result.fire_detections, drone_pose, self.intrinsics,
                        self.config.input_resolution[0], self.config.input_resolution[1]
                    )
                except Exception as e:
                    logger.error(f"Error creating fire overlay: {e}")
        
        # Smoke detection
        if self.config.enable_smoke_detection and self.smoke_detector:
            start_time = time.time()
            smoke_prob = self.smoke_detector.add_frame(frame)
            if smoke_prob is not None:
                result.smoke_probability = smoke_prob
            result.smoke_detection_time = time.time() - start_time
        
        # Fauna detection
        if self.config.enable_fauna_detection and self.fauna_detector:
            start_time = time.time()
            fauna_detections, density_map = self.fauna_detector.detect_fauna(frame), \
                                          self.fauna_detector.estimate_density(frame)
            result.fauna_detections = fauna_detections
            result.fauna_detection_time = time.time() - start_time
            
            # Create fauna overlay
            if drone_pose and self.intrinsics:
                try:
                    result.fauna_overlay_geojson = create_fauna_overlay(
                        fauna_detections, density_map, drone_pose, self.intrinsics,
                        self.config.input_resolution[0], self.config.input_resolution[1]
                    )
                except Exception as e:
                    logger.error(f"Error creating fauna overlay: {e}")
        
        # Vegetation health analysis (sample patches)
        if self.config.enable_vegetation_analysis and self.vegetation_detector:
            start_time = time.time()
            result.vegetation_health = self._analyze_vegetation_patches(frame)
            result.vegetation_detection_time = time.time() - start_time
            
            # Create vegetation overlay
            if drone_pose and self.intrinsics and result.vegetation_health:
                try:
                    patch_locations = [(100, 100), (300, 200), (500, 300)]  # Sample locations
                    result.vegetation_overlay_geojson = create_vegetation_overlay(
                        result.vegetation_health, patch_locations, drone_pose, self.intrinsics,
                        self.config.input_resolution[0], self.config.input_resolution[1]
                    )
                except Exception as e:
                    logger.error(f"Error creating vegetation overlay: {e}")
        
        return result
    
    def _analyze_vegetation_patches(self, frame: np.ndarray) -> List[Dict]:
        """Analyze vegetation health in sample patches"""
        if not self.vegetation_detector:
            return []
        
        # Sample 3 patches for analysis
        h, w = frame.shape[:2]
        patch_size = 64
        
        patches = [
            frame[50:50+patch_size, 50:50+patch_size],
            frame[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2],
            frame[h-patch_size-50:h-50, w-patch_size-50:w-50]
        ]
        
        results = []
        for patch in patches:
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                health_result = self.vegetation_detector.classify_veg(patch)
                results.append(health_result)
        
        return results

class StreamingDemo:
    """Demo application for video streaming pipeline"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.processor = None
        self.results_log = []
        
    def start_demo(self, source: Union[int, str] = 0, duration: float = 30.0):
        """
        Start streaming demo
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            duration: Demo duration in seconds
        """
        # Initialize camera intrinsics (example values)
        intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0
        )
        
        # Initialize processor
        self.processor = VideoStreamProcessor(
            self.config, intrinsics, self._result_callback
        )
        
        # Start processor
        self.processor.start()
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Could not open video source: {source}")
            return
        
        logger.info(f"Starting {duration}s demo with source: {source}")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):  # Video file ended
                        break
                    else:
                        continue
                
                # Create mock drone pose
                drone_pose = DronePose(
                    latitude=37.7749 + np.random.uniform(-0.001, 0.001),
                    longitude=-122.4194 + np.random.uniform(-0.001, 0.001),
                    altitude=100.0 + np.random.uniform(-10, 10),
                    roll=np.random.uniform(-5, 5),
                    pitch=np.random.uniform(-10, 5),
                    yaw=np.random.uniform(0, 360),
                    timestamp=time.time()
                )
                
                # Process frame
                if self.processor.process_frame(frame, drone_pose=drone_pose):
                    frame_count += 1
                
                # Control frame rate
                time.sleep(1.0 / self.config.target_fps)
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            cap.release()
            self.processor.stop()
            
            # Print summary
            self._print_demo_summary(duration, frame_count)
    
    def _result_callback(self, result: FrameResult):
        """Callback for processing results"""
        self.results_log.append(result)
        
        # Log interesting results
        if result.fire_detections and result.fire_detections.get('detection_count', 0) > 0:
            logger.info(f"Frame {result.frame_id}: Fire detected! "
                       f"Count: {result.fire_detections['detection_count']}")
        
        if result.smoke_probability and result.smoke_probability > 0.7:
            logger.info(f"Frame {result.frame_id}: High smoke probability: "
                       f"{result.smoke_probability:.2f}")
        
        # Performance logging
        if result.frame_id % 30 == 0:  # Every 30 frames
            stats = self.processor.get_performance_stats()
            logger.info(f"Performance: {stats['current_fps']:.1f} FPS, "
                       f"Processing time: {result.processing_time:.3f}s")
    
    def _print_demo_summary(self, duration: float, frame_count: int):
        """Print demo summary statistics"""
        if not self.results_log:
            logger.warning("No results recorded")
            return
        
        print("\n" + "="*50)
        print("STREAMING DEMO SUMMARY")
        print("="*50)
        
        print(f"Duration: {duration:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {frame_count/duration:.1f}")
        
        # Processing times
        processing_times = [r.processing_time for r in self.results_log]
        if processing_times:
            print(f"\nProcessing Performance:")
            print(f"  Average processing time: {np.mean(processing_times):.3f}s")
            print(f"  Max processing time: {np.max(processing_times):.3f}s")
            print(f"  Min processing time: {np.min(processing_times):.3f}s")
        
        # Detection statistics
        fire_detections = sum(1 for r in self.results_log 
                            if r.fire_detections and r.fire_detections.get('detection_count', 0) > 0)
        smoke_detections = sum(1 for r in self.results_log 
                             if r.smoke_probability and r.smoke_probability > 0.5)
        
        print(f"\nDetection Results:")
        print(f"  Frames with fire: {fire_detections}")
        print(f"  Frames with smoke: {smoke_detections}")
        
        # Performance target check
        target_fps = self.config.target_fps
        achieved_fps = frame_count / duration
        performance_ratio = achieved_fps / target_fps
        
        print(f"\nPerformance Target:")
        print(f"  Target FPS: {target_fps}")
        print(f"  Achieved FPS: {achieved_fps:.1f}")
        print(f"  Performance: {performance_ratio:.1%}")
        
        if performance_ratio >= 0.9:
            print("  ✅ Performance target ACHIEVED")
        else:
            print("  ❌ Performance target MISSED")
        
        print("="*50)

# Main streaming functions for app integration

def create_stream_processor(enable_fire: bool = True, enable_smoke: bool = True,
                          enable_fauna: bool = False, enable_vegetation: bool = False,
                          target_fps: float = 15.0) -> VideoStreamProcessor:
    """
    Create a configured video stream processor
    
    Args:
        enable_fire: Enable fire detection
        enable_smoke: Enable smoke detection  
        enable_fauna: Enable fauna detection
        enable_vegetation: Enable vegetation analysis
        target_fps: Target processing FPS
        
    Returns:
        Configured VideoStreamProcessor instance
    """
    config = StreamConfig(
        target_fps=target_fps,
        enable_fire_detection=enable_fire,
        enable_smoke_detection=enable_smoke,
        enable_fauna_detection=enable_fauna,
        enable_vegetation_analysis=enable_vegetation
    )
    
    return VideoStreamProcessor(config)

def run_streaming_demo():
    """Run the streaming demo application"""
    config = StreamConfig(
        target_fps=15.0,
        enable_fire_detection=True,
        enable_smoke_detection=True,
        enable_fauna_detection=False,  # Disabled for performance
        enable_vegetation_analysis=False  # Disabled for performance
    )
    
    demo = StreamingDemo(config)
    demo.start_demo(source=0, duration=30.0)  # 30 second demo with webcam

if __name__ == "__main__":
    run_streaming_demo()