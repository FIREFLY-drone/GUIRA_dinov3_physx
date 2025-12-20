"""
Geospatial Projection and Mapping Module.
Projects detections from image coordinates to world coordinates using camera pose and DEM.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point, Polygon
import geojson
from loguru import logger
import pyproj
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class CameraModel:
    """Camera model for projection calculations."""
    
    def __init__(self, intrinsics: Dict):
        """
        Initialize camera model.
        
        Args:
            intrinsics: Dictionary with fx, fy, cx, cy, and optionally distortion parameters
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        # Camera intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Distortion parameters (if available)
        self.distortion = intrinsics.get('distortion', {})
        self.k1 = self.distortion.get('k1', 0)
        self.k2 = self.distortion.get('k2', 0)
        self.p1 = self.distortion.get('p1', 0)
        self.p2 = self.distortion.get('p2', 0)
        self.k3 = self.distortion.get('k3', 0)
        
        self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
        
        logger.info(f"Initialized camera model: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coordinates to camera ray direction.
        
        Args:
            u, v: Pixel coordinates
        
        Returns:
            Ray direction in camera coordinate system (normalized)
        """
        # Undistort pixel coordinates if distortion is present
        if np.any(self.dist_coeffs != 0):
            # Simple distortion correction (more complex models available)
            pixel_coords = np.array([[u, v]], dtype=np.float32)
            undistorted = cv2.undistortPoints(
                pixel_coords.reshape(1, 1, 2),
                self.K,
                self.dist_coeffs,
                P=self.K
            )
            u_undist, v_undist = undistorted[0, 0]
        else:
            u_undist, v_undist = u, v
        
        # Convert to homogeneous coordinates
        pixel_homo = np.array([u_undist, v_undist, 1.0])
        
        # Convert to camera ray
        ray_camera = np.linalg.inv(self.K) @ pixel_homo
        
        # Normalize ray
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        return ray_camera


class PoseManager:
    """Manages drone pose data for geospatial projection."""
    
    def __init__(self, pose_file: str):
        """
        Initialize pose manager.
        
        Args:
            pose_file: Path to CSV file with pose data (frame, lat, lon, alt, yaw, pitch, roll)
        """
        self.pose_file = pose_file
        self.pose_data = None
        
        if os.path.exists(pose_file):
            self.pose_data = pd.read_csv(pose_file)
            logger.info(f"Loaded {len(self.pose_data)} pose entries from {pose_file}")
        else:
            logger.warning(f"Pose file not found: {pose_file}")
    
    def get_pose(self, frame_id: int) -> Optional[Dict]:
        """
        Get pose data for specific frame.
        
        Args:
            frame_id: Frame identifier
        
        Returns:
            Pose dictionary or None if not found
        """
        if self.pose_data is None:
            return None
        
        # Find closest frame
        frame_data = self.pose_data[self.pose_data['frame'] == frame_id]
        
        if len(frame_data) == 0:
            # Find nearest frame
            closest_idx = np.argmin(np.abs(self.pose_data['frame'] - frame_id))
            frame_data = self.pose_data.iloc[closest_idx:closest_idx+1]
        
        if len(frame_data) > 0:
            row = frame_data.iloc[0]
            return {
                'frame': int(row['frame']),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'alt': float(row['alt']),
                'yaw': float(row['yaw']),
                'pitch': float(row['pitch']),
                'roll': float(row['roll'])
            }
        
        return None
    
    def pose_to_transformation_matrix(self, pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pose to rotation matrix and translation vector.
        
        Args:
            pose: Pose dictionary
        
        Returns:
            Rotation matrix (3x3) and translation vector (3,)
        """
        # Convert angles to radians
        yaw = np.radians(pose['yaw'])
        pitch = np.radians(pose['pitch'])
        roll = np.radians(pose['roll'])
        
        # Create rotation matrices for each axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Combined rotation matrix (ZYX order)
        R = R_yaw @ R_pitch @ R_roll
        
        # Convert GPS coordinates to local coordinate system
        # For simplicity, using approximate conversion
        # In practice, use proper geodetic transformations
        lat_rad = np.radians(pose['lat'])
        lon_rad = np.radians(pose['lon'])
        
        # Earth radius in meters
        R_earth = 6371000
        
        # Convert to local Cartesian coordinates (approximate)
        x = R_earth * lon_rad * np.cos(lat_rad)
        y = R_earth * lat_rad
        z = pose['alt']
        
        t = np.array([x, y, z])
        
        return R, t


class DEMManager:
    """Manages Digital Elevation Model data for terrain intersection."""
    
    def __init__(self, dem_dir: str):
        """
        Initialize DEM manager.
        
        Args:
            dem_dir: Directory containing DEM tiles (GeoTIFF format)
        """
        self.dem_dir = Path(dem_dir)
        self.dem_tiles = {}
        self.dem_bounds = {}
        
        # Load DEM tiles
        if self.dem_dir.exists():
            for dem_file in self.dem_dir.glob('*.tif'):
                try:
                    with rasterio.open(dem_file) as src:
                        self.dem_tiles[dem_file.stem] = {
                            'path': str(dem_file),
                            'bounds': src.bounds,
                            'transform': src.transform,
                            'crs': src.crs
                        }
                        self.dem_bounds[dem_file.stem] = src.bounds
                        
                    logger.info(f"Loaded DEM tile: {dem_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load DEM tile {dem_file}: {e}")
        
        logger.info(f"Loaded {len(self.dem_tiles)} DEM tiles")
    
    def get_elevation(self, lon: float, lat: float) -> Optional[float]:
        """
        Get elevation at specific coordinates.
        
        Args:
            lon, lat: Longitude and latitude
        
        Returns:
            Elevation in meters or None if not available
        """
        # Find appropriate DEM tile
        for tile_name, bounds in self.dem_bounds.items():
            if (bounds.left <= lon <= bounds.right and 
                bounds.bottom <= lat <= bounds.top):
                
                tile_info = self.dem_tiles[tile_name]
                
                try:
                    with rasterio.open(tile_info['path']) as src:
                        # Sample elevation at point
                        row, col = src.index(lon, lat)
                        if (0 <= row < src.height and 0 <= col < src.width):
                            elevation = src.read(1)[row, col]
                            if not np.isnan(elevation):
                                return float(elevation)
                except Exception as e:
                    logger.warning(f"Error reading elevation from {tile_name}: {e}")
        
        return None
    
    def interpolate_elevation(self, lon: float, lat: float) -> Optional[float]:
        """
        Interpolate elevation with bilinear interpolation.
        
        Args:
            lon, lat: Longitude and latitude
        
        Returns:
            Interpolated elevation or None if not available
        """
        # Find appropriate DEM tile
        for tile_name, bounds in self.dem_bounds.items():
            if (bounds.left <= lon <= bounds.right and 
                bounds.bottom <= lat <= bounds.top):
                
                tile_info = self.dem_tiles[tile_name]
                
                try:
                    with rasterio.open(tile_info['path']) as src:
                        # Get transform
                        transform = src.transform
                        
                        # Convert to pixel coordinates
                        col, row = ~transform * (lon, lat)
                        
                        # Get integer parts
                        col_int = int(col)
                        row_int = int(row)
                        
                        # Check bounds
                        if (0 <= col_int < src.width - 1 and 
                            0 <= row_int < src.height - 1):
                            
                            # Read 2x2 window
                            window = src.read(1, window=((row_int, row_int + 2), 
                                                        (col_int, col_int + 2)))
                            
                            # Bilinear interpolation
                            dx = col - col_int
                            dy = row - row_int
                            
                            # Check for NaN values
                            if not np.any(np.isnan(window)):
                                elev = (window[0, 0] * (1 - dx) * (1 - dy) +
                                       window[0, 1] * dx * (1 - dy) +
                                       window[1, 0] * (1 - dx) * dy +
                                       window[1, 1] * dx * dy)
                                return float(elev)
                            
                except Exception as e:
                    logger.warning(f"Error interpolating elevation from {tile_name}: {e}")
        
        return None


class GeospatialProjector:
    """Main class for geospatial projection of detections."""
    
    def __init__(self, dem_file: str, intrinsics_file: str):
        """
        Initialize geospatial projector.
        
        Args:
            dem_file: Path to DEM file
            intrinsics_file: Path to camera intrinsics file
        """
        self.dem_file = dem_file
        self.intrinsics_file = intrinsics_file
        
        # Initialize components
        with open(intrinsics_file, 'r') as f:
            intrinsics = json.load(f)
        self.camera = CameraModel(intrinsics)
        
        # Initialize DEM manager with single file
        self.dem_manager = DEMManager([dem_file])
        
        # Default pose manager (will be updated per detection)
        pose_file = 'data/pose/pose.csv'
        if Path(pose_file).exists():
            self.pose_manager = PoseManager(pose_file)
        else:
            self.pose_manager = None
        
        # Output CRS
        self.output_crs = 'EPSG:4326'
        
        logger.info("Initialized GeospatialProjector")
    
    def pixel_to_world(self, u: float, v: float, frame_id: int) -> Optional[Tuple[float, float, float]]:
        """
        Project pixel coordinates to world coordinates.
        
        Args:
            u, v: Pixel coordinates
            frame_id: Frame identifier
        
        Returns:
            World coordinates (lon, lat, elevation) or None if projection fails
        """
        # Get pose for frame
        pose = self.pose_manager.get_pose(frame_id)
        if pose is None:
            logger.warning(f"No pose data for frame {frame_id}")
            return None
        
        # Get camera ray
        ray_camera = self.camera.pixel_to_ray(u, v)
        
        # Transform ray to world coordinates
        R, t = self.pose_manager.pose_to_transformation_matrix(pose)
        ray_world = R @ ray_camera
        camera_pos_world = t
        
        # Intersect ray with terrain
        intersection = self._intersect_ray_with_terrain(camera_pos_world, ray_world)
        
        return intersection
    
    def _intersect_ray_with_terrain(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Intersect camera ray with terrain using DEM.
        
        Args:
            ray_origin: Ray origin in world coordinates
            ray_direction: Ray direction (normalized)
        
        Returns:
            Intersection point (lon, lat, elevation) or None
        """
        # Use numerical optimization to find intersection
        def height_difference(t):
            # Point along ray
            point = ray_origin + t * ray_direction
            
            # Convert to lon/lat (approximate)
            R_earth = 6371000
            lat = np.degrees(point[1] / R_earth)
            lon = np.degrees(point[0] / (R_earth * np.cos(np.radians(lat))))
            
            # Get terrain elevation
            terrain_elev = self.dem_manager.interpolate_elevation(lon, lat)
            if terrain_elev is None:
                # Assume sea level if no DEM data
                terrain_elev = 0.0
            
            # Difference between ray height and terrain height
            return abs(point[2] - terrain_elev)
        
        try:
            # Find minimum distance to terrain
            result = minimize_scalar(height_difference, bounds=(0, 10000), method='bounded')
            
            if result.success:
                t_optimal = result.x
                intersection_point = ray_origin + t_optimal * ray_direction
                
                # Convert back to lon/lat
                R_earth = 6371000
                lat = np.degrees(intersection_point[1] / R_earth)
                lon = np.degrees(intersection_point[0] / (R_earth * np.cos(np.radians(lat))))
                elevation = intersection_point[2]
                
                return lon, lat, elevation
            
        except Exception as e:
            logger.warning(f"Ray-terrain intersection failed: {e}")
        
        return None
    
    def project_detections(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Project all detections from a frame to world coordinates.
        
        Args:
            detections: List of detection dictionaries with bounding boxes
            frame_id: Frame identifier
        
        Returns:
            List of detections with added geospatial coordinates
        """
        projected_detections = []
        
        for detection in detections:
            # Use center of bounding box
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            center_u = (x1 + x2) / 2
            center_v = (y1 + y2) / 2
            
            # Project to world coordinates
            world_coords = self.pixel_to_world(center_u, center_v, frame_id)
            
            if world_coords is not None:
                lon, lat, elevation = world_coords
                
                # Create projected detection
                projected_detection = detection.copy()
                projected_detection.update({
                    'longitude': lon,
                    'latitude': lat,
                    'elevation': elevation,
                    'frame_id': frame_id,
                    'pixel_center': [center_u, center_v]
                })
                
                projected_detections.append(projected_detection)
            else:
                logger.warning(f"Failed to project detection in frame {frame_id}")
        
        return projected_detections
    
    def create_geojson_features(self, projected_detections: List[Dict]) -> List[Dict]:
        """
        Create GeoJSON features from projected detections.
        
        Args:
            projected_detections: List of detections with geospatial coordinates
        
        Returns:
            List of GeoJSON features
        """
        features = []
        
        for detection in projected_detections:
            if 'longitude' in detection and 'latitude' in detection:
                # Create point geometry
                geometry = geojson.Point((detection['longitude'], detection['latitude']))
                
                # Create properties
                properties = {
                    'class': detection.get('class', 'unknown'),
                    'confidence': detection.get('score', detection.get('confidence', 0.0)),
                    'elevation': detection.get('elevation', 0.0),
                    'frame_id': detection.get('frame_id', 0),
                    'detection_type': 'point'
                }
                
                # Add type-specific properties
                if 'species' in detection:
                    properties['species'] = detection['species']
                if 'health' in detection:
                    properties['health'] = detection['health']
                
                # Create feature
                feature = geojson.Feature(
                    geometry=geometry,
                    properties=properties
                )
                
                features.append(feature)
        
        return features
    
    def create_polygon_features(self, projected_detections: List[Dict], buffer_meters: float = 10.0) -> List[Dict]:
        """
        Create polygon features from point detections with buffer.
        
        Args:
            projected_detections: List of detections with geospatial coordinates
            buffer_meters: Buffer distance in meters
        
        Returns:
            List of GeoJSON polygon features
        """
        features = []
        
        # Convert buffer from meters to degrees (approximate)
        buffer_degrees = buffer_meters / 111000  # Rough conversion
        
        for detection in projected_detections:
            if 'longitude' in detection and 'latitude' in detection:
                lon, lat = detection['longitude'], detection['latitude']
                
                # Create simple square buffer
                polygon_coords = [
                    [lon - buffer_degrees, lat - buffer_degrees],
                    [lon + buffer_degrees, lat - buffer_degrees],
                    [lon + buffer_degrees, lat + buffer_degrees],
                    [lon - buffer_degrees, lat + buffer_degrees],
                    [lon - buffer_degrees, lat - buffer_degrees]  # Close polygon
                ]
                
                geometry = geojson.Polygon([polygon_coords])
                
                properties = {
                    'class': detection.get('class', 'unknown'),
                    'confidence': detection.get('score', detection.get('confidence', 0.0)),
                    'elevation': detection.get('elevation', 0.0),
                    'frame_id': detection.get('frame_id', 0),
                    'detection_type': 'polygon',
                    'buffer_meters': buffer_meters
                }
                
                feature = geojson.Feature(
                    geometry=geometry,
                    properties=properties
                )
                
                features.append(feature)
        
        return features
    
    def save_geojson(self, features: List[Dict], output_path: str):
        """
        Save features to GeoJSON file.
        
        Args:
            features: List of GeoJSON features
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        feature_collection = geojson.FeatureCollection(features)
        
        with open(output_path, 'w') as f:
            geojson.dump(feature_collection, f, indent=2)
        
        logger.info(f"Saved {len(features)} features to {output_path}")
    
    def process_frame_detections(self, detections: List[Dict], frame_id: int, 
                                output_dir: str = 'outputs/geolocations') -> str:
        """
        Process detections from a single frame and save as GeoJSON.
        
        Args:
            detections: List of detection dictionaries
            frame_id: Frame identifier
            output_dir: Output directory
        
        Returns:
            Path to saved GeoJSON file
        """
        # Project detections
        projected_detections = self.project_detections(detections, frame_id)
        
        if not projected_detections:
            logger.warning(f"No projectable detections in frame {frame_id}")
            return None
        
        # Create features
        point_features = self.create_geojson_features(projected_detections)
        polygon_features = self.create_polygon_features(projected_detections, buffer_meters=20.0)
        
        all_features = point_features + polygon_features
        
        # Save to file
        output_path = Path(output_dir) / f'frame_{frame_id:06d}.geojson'
        self.save_geojson(all_features, str(output_path))
        
        logger.info(f"Processed {len(projected_detections)} detections from frame {frame_id}")
        
        return str(output_path)


def create_sample_pose_data():
    """Create sample pose data for testing."""
    # Generate sample GPS track
    frames = range(0, 1000, 10)
    
    # Sample flight path (rectangular)
    base_lat = 40.7128  # New York City area
    base_lon = -74.0060
    
    poses = []
    for i, frame in enumerate(frames):
        # Simple rectangular flight pattern
        t = i / len(frames) * 4  # 4 segments
        
        if t < 1:
            lat = base_lat + t * 0.01
            lon = base_lon
        elif t < 2:
            lat = base_lat + 0.01
            lon = base_lon + (t - 1) * 0.01
        elif t < 3:
            lat = base_lat + 0.01 - (t - 2) * 0.01
            lon = base_lon + 0.01
        else:
            lat = base_lat
            lon = base_lon + 0.01 - (t - 3) * 0.01
        
        poses.append({
            'frame': frame,
            'lat': lat,
            'lon': lon,
            'alt': 100.0 + np.random.normal(0, 5),  # 100m altitude with noise
            'yaw': np.random.normal(0, 10),
            'pitch': np.random.normal(0, 5),
            'roll': np.random.normal(0, 5)
        })
    
    return pd.DataFrame(poses)


if __name__ == "__main__":
    # Test geospatial projection
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--detections', help='Path to detection JSON file')
    parser.add_argument('--frame_id', type=int, default=0, help='Frame ID')
    parser.add_argument('--output', default='outputs/geolocations/test.geojson', help='Output GeoJSON path')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['geospatial']
    
    # Initialize projector
    projector = GeospatialProjector(config)
    
    if args.detections:
        # Load detections
        with open(args.detections, 'r') as f:
            detections = json.load(f)
        
        # Process detections
        output_path = projector.process_frame_detections(
            detections, 
            args.frame_id, 
            os.path.dirname(args.output)
        )
        
        if output_path:
            logger.info(f"Geospatial projection completed. Output: {output_path}")
        else:
            logger.warning("No detections could be projected")
    else:
        # Create sample data for testing
        logger.info("Creating sample pose data...")
        sample_poses = create_sample_pose_data()
        pose_file = Path(config['pose_file'])
        pose_file.parent.mkdir(parents=True, exist_ok=True)
        sample_poses.to_csv(pose_file, index=False)
        logger.info(f"Sample pose data saved to {pose_file}")
        
        # Test with sample detection
        sample_detection = [
            {
                'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200,
                'score': 0.85, 'class': 'fire'
            }
        ]
        
        output_path = projector.process_frame_detections(
            sample_detection, 
            0, 
            os.path.dirname(args.output)
        )
        
        logger.info(f"Test completed. Output: {output_path}")
