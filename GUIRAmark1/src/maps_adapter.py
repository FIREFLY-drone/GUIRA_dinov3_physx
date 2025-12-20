"""
Maps Adapter for GeoJSON Overlays

Converts detection and prediction outputs to GeoJSON format for map visualization.
Projects pixel coordinates to geographic coordinates using drone pose/intrinsics/extrinsics.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import cv2
import math

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0

@dataclass
class DronePose:
    """Drone pose information"""
    latitude: float    # GPS latitude (degrees)
    longitude: float   # GPS longitude (degrees)
    altitude: float    # Altitude above ground level (meters)
    roll: float        # Roll angle (degrees)
    pitch: float       # Pitch angle (degrees)
    yaw: float         # Yaw/heading angle (degrees, 0=North)
    timestamp: Optional[float] = None

class MapProjector:
    """Projects pixel coordinates to geographic coordinates"""
    
    def __init__(self, intrinsics: CameraIntrinsics, image_width: int, image_height: int):
        self.intrinsics = intrinsics
        self.image_width = image_width
        self.image_height = image_height
        
    def pixel_to_world(self, pixel_coords: np.ndarray, drone_pose: DronePose, 
                      ground_elevation: float = 0.0) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates (lat/lon)
        
        Args:
            pixel_coords: Array of pixel coordinates (N, 2) as [x, y]
            drone_pose: Current drone pose information
            ground_elevation: Ground elevation in meters
            
        Returns:
            Array of world coordinates (N, 2) as [longitude, latitude]
        """
        if len(pixel_coords) == 0:
            return np.array([])
        
        # Convert pixel coordinates to normalized camera coordinates
        normalized_coords = self._pixel_to_normalized(pixel_coords)
        
        # Project to ground plane
        world_coords = self._project_to_ground(normalized_coords, drone_pose, ground_elevation)
        
        return world_coords
    
    def _pixel_to_normalized(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera coordinates"""
        # Account for principal point offset
        x_norm = (pixel_coords[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        y_norm = (pixel_coords[:, 1] - self.intrinsics.cy) / self.intrinsics.fy
        
        # Apply radial distortion correction (simplified)
        if abs(self.intrinsics.k1) > 1e-6:
            r2 = x_norm**2 + y_norm**2
            distortion_factor = 1 + self.intrinsics.k1 * r2
            x_norm /= distortion_factor
            y_norm /= distortion_factor
        
        return np.column_stack([x_norm, y_norm])
    
    def _project_to_ground(self, normalized_coords: np.ndarray, drone_pose: DronePose, 
                          ground_elevation: float) -> np.ndarray:
        """Project normalized coordinates to ground plane"""
        # Convert angles to radians
        roll_rad = np.radians(drone_pose.roll)
        pitch_rad = np.radians(drone_pose.pitch)
        yaw_rad = np.radians(drone_pose.yaw)
        
        # Height above ground
        height_agl = drone_pose.altitude - ground_elevation
        if height_agl <= 0:
            height_agl = 1.0  # Minimum height to avoid division by zero
        
        # Rotation matrices
        R_roll = np.array([[1, 0, 0],
                          [0, np.cos(roll_rad), -np.sin(roll_rad)],
                          [0, np.sin(roll_rad), np.cos(roll_rad)]])
        
        R_pitch = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
        
        R_yaw = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                         [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                         [0, 0, 1]])
        
        # Combined rotation matrix (yaw * pitch * roll)
        R = R_yaw @ R_pitch @ R_roll
        
        world_coords = []
        
        for x_norm, y_norm in normalized_coords:
            # Ray direction in camera frame (z forward, x right, y down)
            ray_camera = np.array([x_norm, y_norm, 1.0])
            
            # Transform to world frame
            ray_world = R @ ray_camera
            
            # Intersect with ground plane
            # Ground plane equation: z = ground_elevation
            # Ray equation: P = drone_pos + t * ray_world
            # Solve for t when z = ground_elevation
            
            if abs(ray_world[2]) < 1e-6:  # Ray parallel to ground
                # Use approximate projection
                ground_x = x_norm * height_agl
                ground_y = y_norm * height_agl
            else:
                t = -height_agl / ray_world[2]  # Negative because camera z points down
                ground_x = t * ray_world[0]
                ground_y = t * ray_world[1]
            
            # Convert to lat/lon offset from drone position
            lat_offset = ground_y / 111320.0  # Approximate meters per degree latitude
            lon_offset = ground_x / (111320.0 * np.cos(np.radians(drone_pose.latitude)))
            
            world_lat = drone_pose.latitude + lat_offset
            world_lon = drone_pose.longitude + lon_offset
            
            world_coords.append([world_lon, world_lat])
        
        return np.array(world_coords)

class GeoJSONAdapter:
    """Converts detection and prediction outputs to GeoJSON format"""
    
    def __init__(self, projector: MapProjector):
        self.projector = projector
    
    def fire_detections_to_geojson(self, detections: Dict, drone_pose: DronePose) -> Dict:
        """
        Convert fire detections to GeoJSON FeatureCollection
        
        Args:
            detections: Fire detection results from detect_fire()
            drone_pose: Current drone pose
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])
        scores = detections.get('scores', [])
        
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            # Convert box to polygon coordinates
            x1, y1, x2, y2 = box
            pixel_coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            
            # Project to world coordinates
            world_coords = self.projector.pixel_to_world(pixel_coords, drone_pose)
            
            # Create polygon (close the ring)
            coordinates = [world_coords.tolist() + [world_coords[0].tolist()]]
            
            # Determine class name
            class_name = "fire" if class_id == 0 else "smoke"
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {
                    "detection_type": "fire_detection",
                    "class": class_name,
                    "confidence": float(score),
                    "detection_id": f"fire_{i}",
                    "timestamp": drone_pose.timestamp,
                    "fusion_used": detections.get('fusion_used', False)
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def fauna_detections_to_geojson(self, detections: Dict, density_map: np.ndarray, 
                                   drone_pose: DronePose) -> Dict:
        """
        Convert fauna detections and density to GeoJSON
        
        Args:
            detections: Fauna detection results
            density_map: Density estimation map
            drone_pose: Current drone pose
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        # Add individual detections as points
        boxes = detections.get('boxes', [])
        species = detections.get('species', [])
        scores = detections.get('scores', [])
        health_status = detections.get('health_status', [])
        
        for i, (box, spec, score, health) in enumerate(zip(boxes, species, scores, health_status)):
            # Use box center as point location
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            pixel_coords = np.array([[center_x, center_y]])
            world_coords = self.projector.pixel_to_world(pixel_coords, drone_pose)
            
            if len(world_coords) > 0:
                lon, lat = world_coords[0]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "detection_type": "fauna_detection",
                        "species": spec,
                        "health_status": health,
                        "confidence": float(score),
                        "detection_id": f"fauna_{i}",
                        "timestamp": drone_pose.timestamp
                    }
                }
                features.append(feature)
        
        # Add density heatmap as grid polygons
        density_features = self._density_to_geojson(density_map, drone_pose)
        features.extend(density_features)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def vegetation_health_to_geojson(self, health_patches: List[Dict], 
                                   patch_locations: List[Tuple[int, int]], 
                                   drone_pose: DronePose) -> Dict:
        """
        Convert vegetation health classifications to GeoJSON
        
        Args:
            health_patches: List of vegetation health results
            patch_locations: List of (x, y) pixel locations for each patch
            drone_pose: Current drone pose
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for i, (health_result, (x, y)) in enumerate(zip(health_patches, patch_locations)):
            # Convert patch location to world coordinates
            pixel_coords = np.array([[x, y]])
            world_coords = self.projector.pixel_to_world(pixel_coords, drone_pose)
            
            if len(world_coords) > 0:
                lon, lat = world_coords[0]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "detection_type": "vegetation_health",
                        "health_class": health_result['health_class'],
                        "confidence": health_result['confidence'],
                        "vari_index": health_result['vari_index'],
                        "probabilities": health_result['probabilities'],
                        "patch_id": f"veg_{i}",
                        "timestamp": drone_pose.timestamp
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def fire_spread_to_geojson(self, spread_masks: np.ndarray, drone_pose: DronePose,
                              time_horizons: List[int] = None) -> Dict:
        """
        Convert fire spread predictions to GeoJSON
        
        Args:
            spread_masks: Fire spread predictions (T, H, W)
            drone_pose: Current drone pose
            time_horizons: Time horizons to include (default: [1, 3, 6, 12])
            
        Returns:
            GeoJSON FeatureCollection
        """
        if time_horizons is None:
            time_horizons = [1, 3, 6, 12]
        
        features = []
        
        for t_idx in time_horizons:
            if t_idx <= len(spread_masks):
                mask = spread_masks[t_idx - 1]  # Convert to 0-based indexing
                
                # Convert mask to polygons
                contours = self._mask_to_contours(mask)
                
                for j, contour in enumerate(contours):
                    # Project contour to world coordinates
                    world_coords = self.projector.pixel_to_world(contour, drone_pose)
                    
                    if len(world_coords) >= 3:  # Need at least 3 points for polygon
                        # Close the polygon
                        coordinates = [world_coords.tolist() + [world_coords[0].tolist()]]
                        
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": coordinates
                            },
                            "properties": {
                                "detection_type": "fire_spread_prediction",
                                "time_horizon": t_idx,
                                "polygon_id": f"spread_{t_idx}h_{j}",
                                "timestamp": drone_pose.timestamp,
                                "prediction_time": drone_pose.timestamp + (t_idx * 3600) if drone_pose.timestamp else None
                            }
                        }
                        features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _density_to_geojson(self, density_map: np.ndarray, drone_pose: DronePose) -> List[Dict]:
        """Convert density map to grid of polygons"""
        features = []
        h, w = density_map.shape
        
        # Calculate grid cell size in pixels
        cell_height = self.projector.image_height // h
        cell_width = self.projector.image_width // w
        
        for i in range(h):
            for j in range(w):
                density_value = density_map[i, j]
                
                if density_value > 0.1:  # Only include cells with significant density
                    # Cell corners in pixels
                    x1 = j * cell_width
                    y1 = i * cell_height
                    x2 = (j + 1) * cell_width
                    y2 = (i + 1) * cell_height
                    
                    pixel_coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    world_coords = self.projector.pixel_to_world(pixel_coords, drone_pose)
                    
                    if len(world_coords) == 4:
                        coordinates = [world_coords.tolist() + [world_coords[0].tolist()]]
                        
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": coordinates
                            },
                            "properties": {
                                "detection_type": "fauna_density",
                                "density_value": float(density_value),
                                "grid_i": i,
                                "grid_j": j,
                                "timestamp": drone_pose.timestamp
                            }
                        }
                        features.append(feature)
        
        return features
    
    def _mask_to_contours(self, mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
        """Convert binary mask to contour polygons"""
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and simplify
        simplified_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Simplify contour
                epsilon = 0.005 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # Reshape to (N, 2)
                simplified = simplified.reshape(-1, 2)
                simplified_contours.append(simplified)
        
        return simplified_contours

# Main adapter functions for app integration

def create_fire_overlay(detections: Dict, drone_pose: DronePose, 
                       intrinsics: CameraIntrinsics, image_width: int, image_height: int) -> str:
    """
    Create GeoJSON overlay for fire detections
    
    Args:
        detections: Fire detection results
        drone_pose: Current drone pose
        intrinsics: Camera intrinsic parameters
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        GeoJSON string
    """
    projector = MapProjector(intrinsics, image_width, image_height)
    adapter = GeoJSONAdapter(projector)
    geojson = adapter.fire_detections_to_geojson(detections, drone_pose)
    return json.dumps(geojson, indent=2)

def create_fauna_overlay(detections: Dict, density_map: np.ndarray, drone_pose: DronePose,
                        intrinsics: CameraIntrinsics, image_width: int, image_height: int) -> str:
    """
    Create GeoJSON overlay for fauna detections and density
    
    Returns:
        GeoJSON string
    """
    projector = MapProjector(intrinsics, image_width, image_height)
    adapter = GeoJSONAdapter(projector)
    geojson = adapter.fauna_detections_to_geojson(detections, density_map, drone_pose)
    return json.dumps(geojson, indent=2)

def create_vegetation_overlay(health_patches: List[Dict], patch_locations: List[Tuple[int, int]],
                             drone_pose: DronePose, intrinsics: CameraIntrinsics,
                             image_width: int, image_height: int) -> str:
    """
    Create GeoJSON overlay for vegetation health
    
    Returns:
        GeoJSON string
    """
    projector = MapProjector(intrinsics, image_width, image_height)
    adapter = GeoJSONAdapter(projector)
    geojson = adapter.vegetation_health_to_geojson(health_patches, patch_locations, drone_pose)
    return json.dumps(geojson, indent=2)

def create_spread_overlay(spread_masks: np.ndarray, drone_pose: DronePose,
                         intrinsics: CameraIntrinsics, image_width: int, image_height: int) -> str:
    """
    Create GeoJSON overlay for fire spread predictions
    
    Returns:
        GeoJSON string
    """
    projector = MapProjector(intrinsics, image_width, image_height)
    adapter = GeoJSONAdapter(projector)
    geojson = adapter.fire_spread_to_geojson(spread_masks, drone_pose)
    return json.dumps(geojson, indent=2)

# Example usage and testing
if __name__ == "__main__":
    # Example camera intrinsics
    intrinsics = CameraIntrinsics(
        fx=800.0, fy=800.0, cx=640.0, cy=480.0,
        k1=-0.1, k2=0.05
    )
    
    # Example drone pose
    drone_pose = DronePose(
        latitude=37.7749, longitude=-122.4194, altitude=100.0,
        roll=0.0, pitch=-10.0, yaw=45.0, timestamp=1640995200
    )
    
    # Example fire detection
    fire_detections = {
        'boxes': [[100, 150, 200, 250], [300, 300, 400, 400]],
        'classes': [0, 1],  # fire, smoke
        'scores': [0.85, 0.72],
        'fusion_used': True
    }
    
    # Create fire overlay
    fire_geojson = create_fire_overlay(fire_detections, drone_pose, intrinsics, 1280, 960)
    print("Fire Detection GeoJSON:")
    print(fire_geojson[:500] + "..." if len(fire_geojson) > 500 else fire_geojson)