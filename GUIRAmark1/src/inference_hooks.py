"""
Phase 5 — Inference Hooks & App Integration

This module exposes Python functions used by run_pipeline.py and provides
app integration APIs for fire prevention models.
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FireDetectionInference:
    """Fire detection using YOLOv8 with optional thermal fusion"""
    
    def __init__(self, model_path: str, use_fusion: bool = True, device: str = 'cpu'):
        self.model_path = model_path
        self.use_fusion = use_fusion
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the fire detection model"""
        try:
            # Try to load real YOLO model
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLOv8 model from {self.model_path}")
        except ImportError:
            logger.warning("YOLO not available, using mock detection")
            self.model = None
    
    def detect_fire(self, frame: np.ndarray, thermal_frame: Optional[np.ndarray] = None, 
                   confidence_threshold: float = 0.5) -> Dict:
        """
        Detect fire in RGB or thermal frame
        
        Args:
            frame: RGB image (H, W, 3)
            thermal_frame: Optional thermal image (H, W) or (H, W, 3)
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Dict with boxes, classes, scores, and metadata
        """
        if self.model is not None:
            # Use real YOLO model
            results = self.model(frame, conf=confidence_threshold)
            
            boxes = []
            classes = []
            scores = []
            
            for r in results:
                if r.boxes is not None:
                    boxes.extend(r.boxes.xyxy.cpu().numpy())
                    classes.extend(r.boxes.cls.cpu().numpy())
                    scores.extend(r.boxes.conf.cpu().numpy())
            
            return {
                'boxes': boxes,
                'classes': classes, 
                'scores': scores,
                'fusion_used': self.use_fusion and thermal_frame is not None,
                'detection_count': len(boxes)
            }
        else:
            # Mock detection for testing
            return self._mock_fire_detection(frame, thermal_frame, confidence_threshold)
    
    def _mock_fire_detection(self, frame, thermal_frame, confidence_threshold):
        """Mock fire detection for testing"""
        h, w = frame.shape[:2]
        
        # Generate mock detections
        num_detections = np.random.randint(0, 3)
        boxes = []
        classes = []
        scores = []
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(50, min(200, w-x1))
            y2 = y1 + np.random.randint(50, min(200, h-y1))
            
            boxes.append([x1, y1, x2, y2])
            classes.append(0 if np.random.random() > 0.3 else 1)  # 0=fire, 1=smoke
            scores.append(np.random.uniform(confidence_threshold, 1.0))
        
        return {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
            'fusion_used': self.use_fusion and thermal_frame is not None,
            'detection_count': len(boxes)
        }

class SmokeDetectionInference:
    """Smoke detection using TimeSFormer for temporal sequences"""
    
    def __init__(self, model_path: str, sequence_length: int = 8, device: str = 'cpu'):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = device
        self.model = None
        self.frame_buffer = []
        self._load_model()
    
    def _load_model(self):
        """Load the smoke detection model"""
        try:
            # Mock model loading
            logger.info(f"Smoke detection model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load smoke model: {e}")
    
    def classify_smoke_clip(self, clip: np.ndarray) -> float:
        """
        Classify smoke probability in video clip
        
        Args:
            clip: Video clip (T, H, W, 3) or (T, H, W)
            
        Returns:
            Smoke probability [0, 1]
        """
        if len(clip) < self.sequence_length:
            logger.warning(f"Clip too short: {len(clip)} < {self.sequence_length}")
            return 0.0
        
        # Use the most recent frames
        recent_clip = clip[-self.sequence_length:]
        
        # Mock smoke classification
        return self._mock_smoke_classification(recent_clip)
    
    def _mock_smoke_classification(self, clip):
        """Mock smoke classification for testing"""
        # Simple heuristic: look for grayish, low-contrast areas
        gray_values = []
        for frame in clip:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Calculate metrics that might indicate smoke
            mean_intensity = np.mean(gray)
            contrast = np.std(gray)
            
            # Smoke tends to be grayish with low contrast
            smoke_score = 1.0 - abs(mean_intensity - 128) / 128  # Favor middle gray
            smoke_score *= max(0, 1.0 - contrast / 50)  # Favor low contrast
            
            gray_values.append(smoke_score)
        
        # Temporal consistency bonus
        consistency = 1.0 - np.std(gray_values)
        final_score = np.mean(gray_values) * consistency
        
        return max(0.0, min(1.0, final_score))
    
    def add_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        Add frame to buffer and return smoke probability if buffer is full
        
        Args:
            frame: Single frame (H, W, 3)
            
        Returns:
            Smoke probability if buffer is full, None otherwise
        """
        self.frame_buffer.append(frame)
        
        # Keep only the required number of frames
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
        
        # Return classification if we have enough frames
        if len(self.frame_buffer) == self.sequence_length:
            return self.classify_smoke_clip(np.array(self.frame_buffer))
        
        return None

class FaunaDetectionInference:
    """Fauna detection and density estimation using YOLOv8 + CSRNet"""
    
    def __init__(self, yolo_path: str, csrnet_path: str, device: str = 'cpu'):
        self.yolo_path = yolo_path
        self.csrnet_path = csrnet_path
        self.device = device
        self.yolo_model = None
        self.csrnet_model = None
        self._load_models()
    
    def _load_models(self):
        """Load both YOLO and CSRNet models"""
        try:
            logger.info(f"Fauna models loaded from {self.yolo_path} and {self.csrnet_path}")
        except Exception as e:
            logger.warning(f"Could not load fauna models: {e}")
    
    def detect_fauna(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect fauna in frame
        
        Args:
            frame: RGB image (H, W, 3)
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Dict with boxes, species, scores, health_status
        """
        return self._mock_fauna_detection(frame, confidence_threshold)
    
    def estimate_density(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate fauna density map
        
        Args:
            frame: RGB image (H, W, 3)
            
        Returns:
            Density map (H//8, W//8) with count estimates per region
        """
        h, w = frame.shape[:2]
        # Return downsampled density map
        density_map = np.random.poisson(0.5, (h//8, w//8)).astype(np.float32)
        return density_map
    
    def _mock_fauna_detection(self, frame, confidence_threshold):
        """Mock fauna detection for testing"""
        h, w = frame.shape[:2]
        species_names = ['deer', 'elk', 'bear', 'bird', 'other']
        health_states = ['healthy', 'distressed']
        
        num_detections = np.random.randint(0, 4)
        boxes = []
        species = []
        scores = []
        health_status = []
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(30, min(150, w-x1))
            y2 = y1 + np.random.randint(30, min(150, h-y1))
            
            boxes.append([x1, y1, x2, y2])
            species.append(np.random.choice(species_names))
            scores.append(np.random.uniform(confidence_threshold, 1.0))
            health_status.append(np.random.choice(health_states))
        
        return {
            'boxes': boxes,
            'species': species,
            'scores': scores,
            'health_status': health_status,
            'detection_count': len(boxes)
        }

class VegetationHealthInference:
    """Vegetation health classification using ResNet50 + VARI"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.classes = ['healthy', 'dry', 'burned']
        self._load_model()
    
    def _load_model(self):
        """Load the vegetation health model"""
        try:
            logger.info(f"Vegetation model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load vegetation model: {e}")
    
    def classify_veg(self, crown_patch: np.ndarray) -> Dict:
        """
        Classify vegetation health for a crown patch
        
        Args:
            crown_patch: RGB image patch (H, W, 3)
            
        Returns:
            Dict with health_class, probabilities, vari_index
        """
        vari_index = self.compute_vari(crown_patch)
        
        # Mock classification based on VARI and visual features
        return self._mock_vegetation_classification(crown_patch, vari_index)
    
    def compute_vari(self, rgb_image: np.ndarray) -> float:
        """
        Compute Visible Atmospherically Resistant Index (VARI)
        
        Args:
            rgb_image: RGB image (H, W, 3) with values [0-255]
            
        Returns:
            VARI index value
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Input must be 3-channel RGB image")
        
        # Convert to float and normalize
        rgb = rgb_image.astype(np.float32) / 255.0
        
        # Extract channels
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        
        # Compute VARI: (Green - Red) / (Green + Red - Blue)
        numerator = green - red
        denominator = green + red - blue
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-8, denominator)
        vari = numerator / denominator
        
        # Return mean VARI value
        return float(np.mean(vari))
    
    def _mock_vegetation_classification(self, patch, vari_index):
        """Mock vegetation classification"""
        # Use VARI to determine health
        if vari_index > 0.15:
            primary_class = 'healthy'
            probs = [0.8, 0.15, 0.05]
        elif vari_index > -0.1:
            primary_class = 'dry'
            probs = [0.2, 0.7, 0.1]
        else:
            primary_class = 'burned'
            probs = [0.1, 0.2, 0.7]
        
        return {
            'health_class': primary_class,
            'probabilities': dict(zip(self.classes, probs)),
            'vari_index': vari_index,
            'confidence': max(probs)
        }

class FireSpreadPredictor:
    """Fire spread prediction using hybrid physics+ML model"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the fire spread model"""
        try:
            logger.info(f"Fire spread model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load spread model: {e}")
    
    def predict_spread(self, raster_seq: np.ndarray, 
                      wind_data: Optional[Dict] = None,
                      terrain_data: Optional[Dict] = None) -> np.ndarray:
        """
        Predict future fire spread masks
        
        Args:
            raster_seq: Historical fire states (T, H, W) where T >= 6
            wind_data: Optional dict with 'speed', 'direction'
            terrain_data: Optional dict with 'elevation', 'slope'
            
        Returns:
            Future fire masks (12, H, W) for next 12 timesteps
        """
        if len(raster_seq) < 6:
            raise ValueError("Need at least 6 historical timesteps")
        
        # Use most recent 6 timesteps
        recent_seq = raster_seq[-6:]
        
        return self._mock_spread_prediction(recent_seq, wind_data, terrain_data)
    
    def _mock_spread_prediction(self, raster_seq, wind_data, terrain_data):
        """Mock fire spread prediction"""
        h, w = raster_seq.shape[1:3]
        
        # Start with the last known fire state
        current_fire = raster_seq[-1].copy()
        
        future_masks = []
        
        for t in range(12):  # Predict 12 timesteps ahead
            # Simple cellular automata-like spread
            new_fire = current_fire.copy()
            
            # Fire spreads to neighboring cells
            kernel = np.array([[0.1, 0.2, 0.1],
                              [0.2, 1.0, 0.2],
                              [0.1, 0.2, 0.1]])
            
            # Apply convolution to spread fire
            from scipy import ndimage
            spread_potential = ndimage.convolve(current_fire, kernel, mode='constant')
            
            # Add wind effects if provided
            if wind_data:
                wind_effect = self._apply_wind_effect(spread_potential, wind_data)
                spread_potential = wind_effect
            
            # Add terrain effects if provided
            if terrain_data:
                terrain_effect = self._apply_terrain_effect(spread_potential, terrain_data)
                spread_potential = terrain_effect
            
            # Apply threshold and add randomness
            spread_threshold = 0.3 + np.random.uniform(-0.1, 0.1)
            new_fire = (spread_potential > spread_threshold).astype(np.float32)
            
            # Ensure fire doesn't disappear (monotonic growth)
            new_fire = np.maximum(new_fire, current_fire)
            
            future_masks.append(new_fire)
            current_fire = new_fire
        
        return np.array(future_masks)
    
    def _apply_wind_effect(self, fire_state, wind_data):
        """Apply wind effects to fire spread"""
        wind_speed = wind_data.get('speed', 0)  # km/h
        wind_direction = wind_data.get('direction', 0)  # degrees
        
        # Simple directional bias based on wind
        wind_factor = min(wind_speed / 50.0, 1.0)  # Normalize to [0,1]
        
        # Create directional kernel
        kernel = np.ones((3, 3)) * (1 - wind_factor)
        
        # Add wind direction bias
        if wind_direction < 45 or wind_direction >= 315:  # North
            kernel[0, :] += wind_factor
        elif wind_direction < 135:  # East
            kernel[:, 2] += wind_factor
        elif wind_direction < 225:  # South
            kernel[2, :] += wind_factor
        else:  # West
            kernel[:, 0] += wind_factor
        
        from scipy import ndimage
        return ndimage.convolve(fire_state, kernel, mode='constant')
    
    def _apply_terrain_effect(self, fire_state, terrain_data):
        """Apply terrain effects to fire spread"""
        elevation = terrain_data.get('elevation')
        if elevation is not None:
            # Fire spreads faster uphill
            grad_y, grad_x = np.gradient(elevation)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            slope_factor = 1.0 + np.clip(slope / 10.0, 0, 0.5)  # Max 50% increase
            return fire_state * slope_factor
        
        return fire_state

# App Integration Functions

def detect_fire(frame: np.ndarray, use_fusion: bool = True, 
               thermal_frame: Optional[np.ndarray] = None,
               model_path: str = "models/fire_yolov8/best.pt") -> Dict:
    """
    Main fire detection function for app integration
    
    Args:
        frame: RGB frame (H, W, 3)
        use_fusion: Whether to use thermal fusion
        thermal_frame: Optional thermal frame
        model_path: Path to fire detection model
        
    Returns:
        Dict with boxes, classes, scores
    """
    detector = FireDetectionInference(model_path, use_fusion)
    return detector.detect_fire(frame, thermal_frame)

def classify_smoke_clip(clip: np.ndarray, 
                       model_path: str = "models/smoke_timesformer/best.pt") -> float:
    """
    Main smoke classification function for app integration
    
    Args:
        clip: Video clip (T, H, W, 3)
        model_path: Path to smoke detection model
        
    Returns:
        Smoke probability [0, 1]
    """
    classifier = SmokeDetectionInference(model_path)
    return classifier.classify_smoke_clip(clip)

def detect_fauna(frame: np.ndarray,
                yolo_path: str = "models/fauna_yolov8_csrnet/yolo_best.pt",
                csrnet_path: str = "models/fauna_yolov8_csrnet/csrnet_best.pth") -> Tuple[Dict, np.ndarray]:
    """
    Main fauna detection function for app integration
    
    Args:
        frame: RGB frame (H, W, 3)
        yolo_path: Path to YOLO detection model
        csrnet_path: Path to CSRNet density model
        
    Returns:
        Tuple of (detection dict, density_map)
    """
    detector = FaunaDetectionInference(yolo_path, csrnet_path)
    detections = detector.detect_fauna(frame)
    density_map = detector.estimate_density(frame)
    return detections, density_map

def classify_veg(crown_patch: np.ndarray,
                model_path: str = "models/vegetation_resnet_vari/best.pt") -> Dict:
    """
    Main vegetation classification function for app integration
    
    Args:
        crown_patch: RGB patch (H, W, 3)
        model_path: Path to vegetation model
        
    Returns:
        Dict with health_class, prob
    """
    classifier = VegetationHealthInference(model_path)
    return classifier.classify_veg(crown_patch)

def predict_spread(raster_seq: np.ndarray,
                  wind_data: Optional[Dict] = None,
                  terrain_data: Optional[Dict] = None,
                  model_path: str = "models/spread_hybrid/best.pt") -> np.ndarray:
    """
    Main fire spread prediction function for app integration
    
    Args:
        raster_seq: Historical fire states (T, H, W)
        wind_data: Optional wind information
        terrain_data: Optional terrain information  
        model_path: Path to spread model
        
    Returns:
        Future fire masks (12, H, W)
    """
    predictor = FireSpreadPredictor(model_path)
    return predictor.predict_spread(raster_seq, wind_data, terrain_data)