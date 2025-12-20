"""
Main pipeline orchestration for real-time fire prevention system.
Processes video streams, runs all detection models, and coordinates responses.
"""

import argparse
import asyncio
import json
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from threading import Thread, Event
import queue
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fire.fire_detection import FireDetectionInference
from smoke.smoke_detection import SmokeDetectionInference
from fauna.fauna_detection import FaunaDetectionInference
from vegetation.vegetation_health import VegetationHealthInference
from geospatial.geospatial_projection import GeospatialProjector
from spread.fire_spread_simulation import FireSpreadSimulator
from utils import setup_logging, load_config, ModelManager, calculate_fire_risk_score, visualize_detections
from loguru import logger


class FrameProcessor:
    """Processes individual frames through all detection models."""
    
    def __init__(self, config: Dict, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        
        # Initialize inference models
        self.fire_detector = self._init_fire_detector()
        self.smoke_detector = self._init_smoke_detector()
        self.fauna_detector = self._init_fauna_detector()
        self.vegetation_detector = self._init_vegetation_detector()
        
        # Frame buffer for temporal models
        self.frame_buffer = []
        self.smoke_sequence_length = config.get('smoke', {}).get('sequence_length', 8)
        
        logger.info("Initialized FrameProcessor with all detection models")
    
    def _init_fire_detector(self) -> Optional[FireDetectionInference]:
        """Initialize fire detection model."""
        try:
            fire_config = self.config.get('fire', {})
            model_path = fire_config.get('model_path', 'models/fire_yolov8.pt')
            return FireDetectionInference(model_path, fire_config)
        except Exception as e:
            logger.error(f"Failed to initialize fire detector: {e}")
            return None
    
    def _init_smoke_detector(self) -> Optional[SmokeDetectionInference]:
        """Initialize smoke detection model."""
        try:
            smoke_config = self.config.get('smoke', {})
            model_path = smoke_config.get('model_path', 'models/smoke_timesformer_best.pt')
            return SmokeDetectionInference(model_path, smoke_config)
        except Exception as e:
            logger.error(f"Failed to initialize smoke detector: {e}")
            return None
    
    def _init_fauna_detector(self) -> Optional[FaunaDetectionInference]:
        """Initialize fauna detection model."""
        try:
            fauna_config = self.config.get('fauna', {})
            detection_model_path = fauna_config.get('model_path', 'models/fauna_yolov8.pt')
            density_model_path = fauna_config.get('density_model_path', 'models/fauna_csrnet_best.pt')
            return FaunaDetectionInference(detection_model_path, density_model_path, fauna_config)
        except Exception as e:
            logger.error(f"Failed to initialize fauna detector: {e}")
            return None
    
    def _init_vegetation_detector(self) -> Optional[VegetationHealthInference]:
        """Initialize vegetation health model."""
        try:
            veg_config = self.config.get('vegetation', {})
            model_path = veg_config.get('model_path', 'models/vegetation_resnet50_best.pt')
            return VegetationHealthInference(model_path, veg_config)
        except Exception as e:
            logger.error(f"Failed to initialize vegetation detector: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict:
        """
        Process a single frame through all detection models.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_id: Frame identifier
        
        Returns:
            Dictionary with all detection results
        """
        results = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'detections': {}
        }
        
        # Fire detection
        if self.fire_detector:
            try:
                fire_detections = self.fire_detector.detect(frame)
                results['detections']['fire'] = fire_detections
                logger.debug(f"Frame {frame_id}: Found {len(fire_detections)} fire detections")
            except Exception as e:
                logger.error(f"Fire detection failed for frame {frame_id}: {e}")
                results['detections']['fire'] = []
        
        # Fauna detection
        if self.fauna_detector:
            try:
                fauna_results = self.fauna_detector.process_image(frame)
                results['detections']['fauna'] = fauna_results['detections']
                results['detections']['fauna_density'] = {
                    'estimated_count': fauna_results['estimated_count'],
                    'health_summary': fauna_results['health_summary']
                }
                logger.debug(f"Frame {frame_id}: Found {len(fauna_results['detections'])} fauna detections")
            except Exception as e:
                logger.error(f"Fauna detection failed for frame {frame_id}: {e}")
                results['detections']['fauna'] = []
                results['detections']['fauna_density'] = {'estimated_count': 0, 'health_summary': {}}
        
        # Vegetation health
        if self.vegetation_detector:
            try:
                veg_analysis = self.vegetation_detector.analyze_vegetation_health(frame)
                results['detections']['vegetation'] = {
                    'overall_health': veg_analysis['overall_health'],
                    'spatial_analysis': veg_analysis['spatial_analysis'],
                    'risk_score': veg_analysis['risk_score']
                }
                logger.debug(f"Frame {frame_id}: Vegetation risk score {veg_analysis['risk_score']:.3f}")
            except Exception as e:
                logger.error(f"Vegetation analysis failed for frame {frame_id}: {e}")
                results['detections']['vegetation'] = {
                    'overall_health': {'predicted_class': 'unknown', 'confidence': 0.0},
                    'spatial_analysis': {'healthy_ratio': 0.0, 'dry_ratio': 0.0, 'burned_ratio': 0.0},
                    'risk_score': 0.0
                }
        
        # Add frame to buffer for smoke detection
        self.frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(self.frame_buffer) > self.smoke_sequence_length:
            self.frame_buffer.pop(0)
        
        # Smoke detection (when we have enough frames)
        if self.smoke_detector and len(self.frame_buffer) == self.smoke_sequence_length:
            try:
                smoke_prob = self.smoke_detector.predict_sequence(self.frame_buffer)
                results['detections']['smoke'] = {'probability': smoke_prob}
                logger.debug(f"Frame {frame_id}: Smoke probability {smoke_prob:.3f}")
            except Exception as e:
                logger.error(f"Smoke detection failed for frame {frame_id}: {e}")
                results['detections']['smoke'] = {'probability': 0.0}
        else:
            results['detections']['smoke'] = {'probability': 0.0}
        
        # Calculate overall fire risk score
        results['fire_risk_score'] = calculate_fire_risk_score(results['detections'])
        
        return results


class VideoStreamer:
    """Handles video input from various sources."""
    
    def __init__(self, input_source: str):
        self.input_source = input_source
        self.cap = None
        self.is_streaming = False
        
    def start_stream(self) -> bool:
        """Start video stream."""
        try:
            # Handle different input types
            if self.input_source.isdigit():
                # Camera index
                self.cap = cv2.VideoCapture(int(self.input_source))
            elif self.input_source.startswith(('rtmp://', 'rtsp://', 'http://')):
                # Network stream
                self.cap = cv2.VideoCapture(self.input_source)
            elif Path(self.input_source).exists():
                # Video file
                self.cap = cv2.VideoCapture(self.input_source)
            else:
                logger.error(f"Invalid input source: {self.input_source}")
                return False
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.input_source}")
                return False
            
            self.is_streaming = True
            logger.info(f"Started video stream from {self.input_source}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video stream: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from stream."""
        if not self.is_streaming or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from stream")
            return None
        
        return frame
    
    def stop_stream(self):
        """Stop video stream."""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            logger.info("Stopped video stream")


class PipelineOrchestrator:
    """Main orchestrator for the fire prevention pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pipeline_config = config.get('pipeline', {})
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.frame_processor = FrameProcessor(config, self.model_manager)
        self.geospatial_projector = GeospatialProjector(config.get('geospatial', {}))
        
        # Initialize fire spread simulator
        spread_config = config.get('spread', {})
        spread_model_path = spread_config.get('model_path', 'models/fire_spread_model_best.pt')
        self.fire_simulator = FireSpreadSimulator(spread_model_path, spread_config)
        
        # Processing parameters
        self.frame_interval = self.pipeline_config.get('frame_interval', 30)
        self.geolocation_interval = self.pipeline_config.get('geolocation_interval', 10)
        self.output_dir = Path(self.pipeline_config.get('output_dir', 'outputs'))
        
        # Statistics
        self.processed_frames = 0
        self.total_detections = 0
        self.high_risk_frames = 0
        
        # Output queues
        self.detection_results = []
        self.geolocation_results = []
        
        logger.info("Initialized PipelineOrchestrator")
    
    def process_stream(self, input_source: str, max_frames: Optional[int] = None):
        """
        Process video stream through the complete pipeline.
        
        Args:
            input_source: Video input source (file, camera, or stream URL)
            max_frames: Maximum frames to process (None for unlimited)
        """
        # Initialize video streamer
        streamer = VideoStreamer(input_source)
        if not streamer.start_stream():
            logger.error("Failed to start video stream")
            return
        
        logger.info(f"Starting pipeline processing on {input_source}")
        start_time = time.time()
        
        try:
            frame_id = 0
            
            while streamer.is_streaming:
                # Get frame
                frame = streamer.get_frame()
                if frame is None:
                    break
                
                # Process every N frames
                if frame_id % self.frame_interval == 0:
                    self._process_single_frame(frame, frame_id)
                
                frame_id += 1
                
                # Check max frames limit
                if max_frames and frame_id >= max_frames:
                    break
                
                # Progress logging
                if frame_id % 1000 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_id / elapsed
                    logger.info(f"Processed {frame_id} frames, {fps:.1f} FPS, "
                              f"{self.total_detections} total detections")
        
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        
        finally:
            streamer.stop_stream()
            self._finalize_processing()
    
    def _process_single_frame(self, frame: np.ndarray, frame_id: int):
        """Process a single frame through the pipeline."""
        try:
            # Run all detections
            results = self.frame_processor.process_frame(frame, frame_id)
            self.detection_results.append(results)
            self.processed_frames += 1
            
            # Count detections
            frame_detections = 0
            for detection_type, detections in results['detections'].items():
                if isinstance(detections, list):
                    frame_detections += len(detections)
            
            self.total_detections += frame_detections
            
            # Check fire risk
            fire_risk = results.get('fire_risk_score', 0.0)
            if fire_risk > 0.7:
                self.high_risk_frames += 1
                logger.warning(f"HIGH FIRE RISK detected in frame {frame_id}: {fire_risk:.3f}")
            
            # Save frame results
            self._save_frame_results(results)
            
            # Geospatial projection (every N processed frames)
            if self.processed_frames % self.geolocation_interval == 0:
                self._perform_geolocation(results)
            
            # Visualization (if enabled)
            if self.pipeline_config.get('enable_visualization', False):
                self._visualize_frame_results(frame, results, frame_id)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
    
    def _save_frame_results(self, results: Dict):
        """Save detection results for a frame."""
        frame_id = results['frame_id']
        
        # Save fire detections
        fire_detections = results['detections'].get('fire', [])
        if fire_detections:
            fire_output_path = self.output_dir / 'fire_predictions' / f'frame_{frame_id:06d}.json'
            fire_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fire_output_path, 'w') as f:
                json.dump(fire_detections, f, indent=2)
        
        # Save fauna detections
        fauna_detections = results['detections'].get('fauna', [])
        if fauna_detections:
            fauna_output_path = self.output_dir / 'fauna_boxes' / f'frame_{frame_id:06d}.json'
            fauna_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fauna_output_path, 'w') as f:
                json.dump(fauna_detections, f, indent=2)
        
        # Save vegetation analysis
        vegetation_data = results['detections'].get('vegetation', {})
        if vegetation_data:
            veg_output_path = self.output_dir / 'vegetation_analysis' / f'frame_{frame_id:06d}.json'
            veg_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(veg_output_path, 'w') as f:
                json.dump(vegetation_data, f, indent=2)
    
    def _perform_geolocation(self, results: Dict):
        """Perform geospatial projection of detections."""
        try:
            frame_id = results['frame_id']
            all_detections = []
            
            # Collect all detections with spatial coordinates
            for detection_type, detections in results['detections'].items():
                if isinstance(detections, list):
                    for det in detections:
                        if all(key in det for key in ['x1', 'y1', 'x2', 'y2']):
                            det_copy = det.copy()
                            det_copy['detection_type'] = detection_type
                            all_detections.append(det_copy)
            
            if all_detections:
                # Project to world coordinates
                geojson_path = self.geospatial_projector.process_frame_detections(
                    all_detections, frame_id, str(self.output_dir / 'geolocations')
                )
                
                if geojson_path:
                    self.geolocation_results.append({
                        'frame_id': frame_id,
                        'geojson_path': geojson_path,
                        'detection_count': len(all_detections)
                    })
                    
                    logger.info(f"Geolocated {len(all_detections)} detections from frame {frame_id}")
        
        except Exception as e:
            logger.error(f"Geolocation failed for frame {results['frame_id']}: {e}")
    
    def _visualize_frame_results(self, frame: np.ndarray, results: Dict, frame_id: int):
        """Create visualization of frame results."""
        try:
            # Collect all detections for visualization
            all_detections = []
            for detection_type, detections in results['detections'].items():
                if isinstance(detections, list):
                    all_detections.extend(detections)
            
            if all_detections:
                vis_frame = visualize_detections(frame, all_detections)
                
                # Add risk score overlay
                fire_risk = results.get('fire_risk_score', 0.0)
                risk_text = f"Fire Risk: {fire_risk:.3f}"
                risk_color = (0, 255, 0) if fire_risk < 0.3 else (0, 165, 255) if fire_risk < 0.7 else (0, 0, 255)
                
                cv2.putText(vis_frame, risk_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 2)
                
                # Save visualization
                vis_output_path = self.output_dir / 'visualizations' / f'frame_{frame_id:06d}.jpg'
                vis_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_output_path), vis_frame)
        
        except Exception as e:
            logger.error(f"Visualization failed for frame {frame_id}: {e}")
    
    def _finalize_processing(self):
        """Finalize processing and generate summary reports."""
        logger.info("Finalizing pipeline processing...")
        
        # Save summary statistics
        summary = {
            'processing_summary': {
                'total_frames_processed': self.processed_frames,
                'total_detections': self.total_detections,
                'high_risk_frames': self.high_risk_frames,
                'risk_ratio': self.high_risk_frames / max(self.processed_frames, 1)
            },
            'detection_summary': self._generate_detection_summary(),
            'geolocation_summary': {
                'geolocated_frames': len(self.geolocation_results),
                'total_geolocated_detections': sum(r['detection_count'] for r in self.geolocation_results)
            }
        }
        
        # Save summary
        summary_path = self.output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detection timeline
        self._save_detection_timeline()
        
        # Generate fire spread prediction if high risk detected
        if self.high_risk_frames > 0:
            self._generate_fire_spread_prediction()
        
        logger.info(f"Pipeline processing completed:")
        logger.info(f"  Processed frames: {self.processed_frames}")
        logger.info(f"  Total detections: {self.total_detections}")
        logger.info(f"  High risk frames: {self.high_risk_frames}")
        logger.info(f"  Results saved to: {self.output_dir}")
    
    def _generate_detection_summary(self) -> Dict:
        """Generate summary of all detections."""
        summary = {
            'fire_detections': 0,
            'smoke_detections': 0,
            'fauna_detections': 0,
            'high_risk_vegetation': 0
        }
        
        for result in self.detection_results:
            detections = result['detections']
            
            # Count fire detections
            fire_dets = detections.get('fire', [])
            summary['fire_detections'] += len(fire_dets)
            
            # Count smoke detections
            smoke_prob = detections.get('smoke', {}).get('probability', 0)
            if smoke_prob > 0.5:
                summary['smoke_detections'] += 1
            
            # Count fauna detections
            fauna_dets = detections.get('fauna', [])
            summary['fauna_detections'] += len(fauna_dets)
            
            # Count high-risk vegetation
            veg_risk = detections.get('vegetation', {}).get('risk_score', 0)
            if veg_risk > 0.6:
                summary['high_risk_vegetation'] += 1
        
        return summary
    
    def _save_detection_timeline(self):
        """Save detection timeline as CSV."""
        timeline_data = []
        
        for result in self.detection_results:
            row = {
                'frame_id': result['frame_id'],
                'timestamp': result['timestamp'],
                'fire_risk_score': result.get('fire_risk_score', 0.0),
                'fire_detections': len(result['detections'].get('fire', [])),
                'smoke_probability': result['detections'].get('smoke', {}).get('probability', 0.0),
                'fauna_detections': len(result['detections'].get('fauna', [])),
                'vegetation_risk': result['detections'].get('vegetation', {}).get('risk_score', 0.0)
            }
            timeline_data.append(row)
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            timeline_path = self.output_dir / 'detection_timeline.csv'
            df.to_csv(timeline_path, index=False)
            logger.info(f"Saved detection timeline to {timeline_path}")
    
    def _generate_fire_spread_prediction(self):
        """Generate fire spread prediction based on detected fire areas."""
        try:
            logger.info("Generating fire spread prediction...")
            
            # Find frames with fire detections
            fire_frames = []
            for result in self.detection_results:
                fire_dets = result['detections'].get('fire', [])
                if fire_dets:
                    fire_frames.append({
                        'frame_id': result['frame_id'],
                        'detections': fire_dets,
                        'fire_risk': result.get('fire_risk_score', 0.0)
                    })
            
            if fire_frames:
                # Use the frame with highest fire risk for simulation
                highest_risk_frame = max(fire_frames, key=lambda x: x['fire_risk'])
                
                # Create initial fire state from detections
                # This is simplified - in practice, you'd use actual geographic coordinates
                ignition_points = [(64, 64), (128, 128)]  # Placeholder coordinates
                initial_fire_state = self.fire_simulator.create_initial_fire_state(ignition_points)
                
                # Create environmental conditions
                env_state = self.fire_simulator.create_environmental_state(
                    wind_speed=8.0,
                    wind_direction=45.0,
                    humidity=0.3,
                    vegetation_density=0.8
                )
                
                # Run simulation
                simulation_steps = self.config.get('spread', {}).get('simulation_steps', 50)
                fire_states = self.fire_simulator.simulate(
                    initial_fire_state,
                    env_state,
                    steps=simulation_steps,
                    use_neural=False  # Use physics model for robustness
                )
                
                # Save simulation results
                spread_output_dir = self.output_dir / 'spread_predictions'
                self.fire_simulator.save_simulation_results(fire_states, str(spread_output_dir))
                self.fire_simulator.visualize_simulation(
                    fire_states, 
                    str(spread_output_dir / 'fire_spread_simulation.gif')
                )
                
                logger.info(f"Fire spread prediction saved to {spread_output_dir}")
        
        except Exception as e:
            logger.error(f"Fire spread prediction failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Fire Prevention Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['process_frames', 'process_video'], 
                      default='process_frames', help='Processing mode')
    parser.add_argument('--input', help='Input source (video file, camera index, or stream URL)')
    parser.add_argument('--max_frames', type=int, help='Maximum frames to process')
    parser.add_argument('--output_dir', help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['pipeline']['output_dir'] = args.output_dir
    
    # Setup logging
    setup_logging(config.get('pipeline', {}).get('log_level', 'INFO'))
    
    logger.info("Starting Fire Prevention Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Input: {args.input}")
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Determine input source
    input_source = args.input or config.get('pipeline', {}).get('input_source', '0')
    
    # Run pipeline
    orchestrator.process_stream(input_source, args.max_frames)


if __name__ == "__main__":
    main()
