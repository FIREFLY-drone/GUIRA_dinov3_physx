"""
Utility functions for the fire prevention system.
"""

import os
import json
import yaml
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from loguru import logger
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.transform import from_bounds
import geojson


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sink=log_file if log_file else "logs/fire_prevention.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days"
    )
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_intrinsics(intrinsics_path: str) -> Dict[str, float]:
    """Load camera intrinsics from JSON file."""
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions to JSON file."""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)


def load_annotations(annotation_path: str) -> List[Dict]:
    """Load annotations from JSON file."""
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def compute_vari(image: np.ndarray) -> np.ndarray:
    """
    Compute Visible Atmospherically Resistant Index (VARI).
    VARI = (Green - Red) / (Green + Red - Blue)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")
    
    # Normalize to [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    # Avoid division by zero
    denominator = g + r - b
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    vari = (g - r) / denominator
    return vari


def create_density_map(bboxes: List[List[float]], image_shape: Tuple[int, int], sigma: float = 15.0) -> np.ndarray:
    """
    Create density map from bounding box centers for crowd counting.
    
    Args:
        bboxes: List of bounding boxes in format [x1, y1, x2, y2]
        image_shape: (height, width) of the image
        sigma: Gaussian kernel sigma
    
    Returns:
        Density map as numpy array
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if 0 <= cx < w and 0 <= cy < h:
            # Create Gaussian kernel
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create coordinate grids
            x_grid, y_grid = np.mgrid[0:kernel_size, 0:kernel_size]
            center = kernel_size // 2
            
            # Gaussian formula
            gaussian = np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2))
            
            # Place Gaussian on density map
            y_start = max(0, cy - center)
            y_end = min(h, cy + center + 1)
            x_start = max(0, cx - center)
            x_end = min(w, cx + center + 1)
            
            # Adjust kernel bounds
            ky_start = max(0, center - cy)
            ky_end = ky_start + (y_end - y_start)
            kx_start = max(0, center - cx)
            kx_end = kx_start + (x_end - x_start)
            
            density_map[y_start:y_end, x_start:x_end] += gaussian[ky_start:ky_end, kx_start:kx_end]
    
    return density_map


def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-maximum suppression using PyTorch.
    
    Args:
        boxes: Tensor of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while sorted_indices.numel() > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current)
        
        if sorted_indices.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.stack(keep)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4)
        box2: Tensor of shape (M, 4)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    # Intersection coordinates
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    # Intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Box areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Union area
    union = area1[:, None] + area2 - intersection
    
    # IoU
    iou = intersection / torch.clamp(union, min=1e-8)
    return iou


def save_geojson(features: List[Dict], output_path: str, crs: str = "EPSG:4326"):
    """Save features to GeoJSON file."""
    ensure_dir(os.path.dirname(output_path))
    
    feature_collection = geojson.FeatureCollection(features)
    
    with open(output_path, 'w') as f:
        geojson.dump(feature_collection, f, indent=2)
    
    logger.info(f"Saved {len(features)} features to {output_path}")


def create_point_feature(lat: float, lon: float, properties: Dict) -> Dict:
    """Create a GeoJSON point feature."""
    return geojson.Feature(
        geometry=geojson.Point((lon, lat)),
        properties=properties
    )


def create_polygon_feature(coordinates: List[List[float]], properties: Dict) -> Dict:
    """Create a GeoJSON polygon feature."""
    return geojson.Feature(
        geometry=geojson.Polygon([coordinates]),
        properties=properties
    )


def visualize_detections(image: np.ndarray, detections: List[Dict], save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize detections on image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries with x1, y1, x2, y2, score, class
        save_path: Optional path to save visualization
    
    Returns:
        Image with visualizations
    """
    vis_image = image.copy()
    
    colors = {
        'fire': (0, 0, 255),      # Red
        'smoke': (128, 128, 128),  # Gray
        'healthy': (0, 255, 0),    # Green
        'distressed': (255, 0, 0), # Blue
        'dry': (0, 165, 255),      # Orange
        'burned': (0, 0, 0)        # Black
    }
    
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        class_name = det.get('class', 'unknown')
        score = det.get('score', 0.0)
        
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, vis_image)
    
    return vis_image


def calculate_fire_risk_score(detections: Dict) -> float:
    """
    Calculate overall fire risk score based on all detections.
    
    Args:
        detections: Dictionary containing all detection results
    
    Returns:
        Fire risk score between 0 and 1
    """
    risk_score = 0.0
    
    # Fire detection weight
    fire_dets = detections.get('fire', [])
    if fire_dets:
        fire_scores = [d.get('score', 0) for d in fire_dets]
        risk_score += 0.4 * max(fire_scores)
    
    # Smoke detection weight
    smoke_prob = detections.get('smoke_prob', 0)
    risk_score += 0.3 * smoke_prob
    
    # Vegetation health weight
    veg_health = detections.get('vegetation_health', {})
    dry_ratio = veg_health.get('dry_ratio', 0)
    burned_ratio = veg_health.get('burned_ratio', 0)
    risk_score += 0.2 * (dry_ratio + burned_ratio)
    
    # Fauna distress weight (animals fleeing)
    fauna_dets = detections.get('fauna', [])
    if fauna_dets:
        distressed_count = sum(1 for d in fauna_dets if d.get('health') == 'distressed')
        distress_ratio = distressed_count / len(fauna_dets)
        risk_score += 0.1 * distress_ratio
    
    return min(risk_score, 1.0)


class ModelManager:
    """Manage loading and inference of multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_name: str, model_path: str):
        """Load a model from checkpoint."""
        if not os.path.exists(model_path):
            logger.warning(f"Model {model_name} not found at {model_path}")
            return None
            
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            self.models[model_name] = model
            logger.info(f"Loaded model {model_name} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str):
        """Get loaded model."""
        return self.models.get(model_name)
    
    def unload_model(self, model_name: str):
        """Unload model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model {model_name}")


def download_sample_data():
    """Download sample data for testing."""
    logger.info("Sample data download would be implemented here")
    logger.info("Please refer to the dataset URLs in config.yaml for manual download")


if __name__ == "__main__":
    # Test utility functions
    config = load_config("config.yaml")
    setup_logging()
    logger.info("Utility functions loaded successfully")
