"""
Unit tests for maps adapter module.
Tests GeoJSON conversion, coordinate projection, and map overlay generation.
"""

import unittest
import numpy as np
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from maps_adapter import (
    CameraIntrinsics, DronePose, MapProjector, GeoJSONAdapter,
    create_fire_overlay, create_fauna_overlay, create_vegetation_overlay,
    create_spread_overlay
)

class TestCameraIntrinsics(unittest.TestCase):
    """Test camera intrinsics data class"""
    
    def test_init_basic(self):
        """Test basic initialization"""
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        
        self.assertEqual(intrinsics.fx, 800.0)
        self.assertEqual(intrinsics.fy, 800.0)
        self.assertEqual(intrinsics.cx, 320.0)
        self.assertEqual(intrinsics.cy, 240.0)
        self.assertEqual(intrinsics.k1, 0.0)  # Default
        self.assertEqual(intrinsics.k2, 0.0)
        self.assertEqual(intrinsics.p1, 0.0)
        self.assertEqual(intrinsics.p2, 0.0)
    
    def test_init_with_distortion(self):
        """Test initialization with distortion parameters"""
        intrinsics = CameraIntrinsics(
            fx=800.0, fy=800.0, cx=320.0, cy=240.0,
            k1=-0.1, k2=0.05, p1=0.01, p2=0.02
        )
        
        self.assertEqual(intrinsics.k1, -0.1)
        self.assertEqual(intrinsics.k2, 0.05)
        self.assertEqual(intrinsics.p1, 0.01)
        self.assertEqual(intrinsics.p2, 0.02)

class TestDronePose(unittest.TestCase):
    """Test drone pose data class"""
    
    def test_init_basic(self):
        """Test basic initialization"""
        pose = DronePose(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            roll=0.0,
            pitch=-10.0,
            yaw=45.0
        )
        
        self.assertEqual(pose.latitude, 37.7749)
        self.assertEqual(pose.longitude, -122.4194)
        self.assertEqual(pose.altitude, 100.0)
        self.assertEqual(pose.roll, 0.0)
        self.assertEqual(pose.pitch, -10.0)
        self.assertEqual(pose.yaw, 45.0)
        self.assertIsNone(pose.timestamp)
    
    def test_init_with_timestamp(self):
        """Test initialization with timestamp"""
        timestamp = 1640995200.0
        pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-10.0, yaw=45.0, timestamp=timestamp
        )
        
        self.assertEqual(pose.timestamp, timestamp)

class TestMapProjector(unittest.TestCase):
    """Test map projector for coordinate conversion"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        self.projector = MapProjector(self.intrinsics, 640, 480)
        self.drone_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-10.0, yaw=45.0
        )
    
    def test_init(self):
        """Test projector initialization"""
        self.assertEqual(self.projector.intrinsics, self.intrinsics)
        self.assertEqual(self.projector.image_width, 640)
        self.assertEqual(self.projector.image_height, 480)
    
    def test_pixel_to_world_empty_input(self):
        """Test pixel to world conversion with empty input"""
        result = self.projector.pixel_to_world(np.array([]), self.drone_pose)
        self.assertEqual(len(result), 0)
    
    def test_pixel_to_world_single_point(self):
        """Test pixel to world conversion with single point"""
        pixel_coords = np.array([[320, 240]])  # Center of image
        
        world_coords = self.projector.pixel_to_world(pixel_coords, self.drone_pose)
        
        # Should return one coordinate pair
        self.assertEqual(world_coords.shape, (1, 2))
        self.assertEqual(len(world_coords[0]), 2)  # [longitude, latitude]
        
        # Check coordinates are reasonable (near drone position)
        lon, lat = world_coords[0]
        self.assertTrue(abs(lon - self.drone_pose.longitude) < 0.01)  # Within ~1km
        self.assertTrue(abs(lat - self.drone_pose.latitude) < 0.01)
    
    def test_pixel_to_world_multiple_points(self):
        """Test pixel to world conversion with multiple points"""
        pixel_coords = np.array([
            [100, 100],  # Top-left quadrant
            [320, 240],  # Center
            [540, 380]   # Bottom-right quadrant
        ])
        
        world_coords = self.projector.pixel_to_world(pixel_coords, self.drone_pose)
        
        # Should return three coordinate pairs
        self.assertEqual(world_coords.shape, (3, 2))
        
        # All coordinates should be near the drone position
        for lon, lat in world_coords:
            self.assertTrue(abs(lon - self.drone_pose.longitude) < 0.1)
            self.assertTrue(abs(lat - self.drone_pose.latitude) < 0.1)
    
    def test_pixel_to_normalized(self):
        """Test pixel to normalized camera coordinates conversion"""
        pixel_coords = np.array([[320, 240], [420, 340]])  # Center and offset
        
        normalized = self.projector._pixel_to_normalized(pixel_coords)
        
        # Check shape
        self.assertEqual(normalized.shape, (2, 2))
        
        # Center pixel should map to (0, 0) in normalized coordinates
        self.assertAlmostEqual(normalized[0, 0], 0.0, places=5)  # x
        self.assertAlmostEqual(normalized[0, 1], 0.0, places=5)  # y
        
        # Offset pixel should be positive
        self.assertGreater(normalized[1, 0], 0)  # x
        self.assertGreater(normalized[1, 1], 0)  # y
    
    def test_project_to_ground_zero_height(self):
        """Test projection with zero height (edge case)"""
        zero_height_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=0.0,  # Ground level
            roll=0.0, pitch=0.0, yaw=0.0
        )
        
        pixel_coords = np.array([[320, 240]])
        
        # Should not crash and return reasonable coordinates
        world_coords = self.projector.pixel_to_world(pixel_coords, zero_height_pose)
        self.assertEqual(world_coords.shape, (1, 2))

class TestGeoJSONAdapter(unittest.TestCase):
    """Test GeoJSON adapter for converting detections to map overlays"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        self.projector = MapProjector(self.intrinsics, 640, 480)
        self.adapter = GeoJSONAdapter(self.projector)
        
        self.drone_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-10.0, yaw=45.0, timestamp=1640995200.0
        )
        
        self.fire_detections = {
            'boxes': [[100, 150, 200, 250], [300, 300, 400, 400]],
            'classes': [0, 1],  # fire, smoke
            'scores': [0.85, 0.72],
            'fusion_used': True,
            'detection_count': 2
        }
    
    def test_init(self):
        """Test adapter initialization"""
        self.assertEqual(self.adapter.projector, self.projector)
    
    def test_fire_detections_to_geojson(self):
        """Test fire detections to GeoJSON conversion"""
        geojson = self.adapter.fire_detections_to_geojson(
            self.fire_detections, self.drone_pose
        )
        
        # Check top-level structure
        self.assertIsInstance(geojson, dict)
        self.assertEqual(geojson['type'], 'FeatureCollection')
        self.assertIn('features', geojson)
        
        # Check features
        features = geojson['features']
        self.assertEqual(len(features), 2)  # Two detections
        
        for i, feature in enumerate(features):
            # Check feature structure
            self.assertEqual(feature['type'], 'Feature')
            self.assertIn('geometry', feature)
            self.assertIn('properties', feature)
            
            # Check geometry
            geometry = feature['geometry']
            self.assertEqual(geometry['type'], 'Polygon')
            self.assertIn('coordinates', geometry)
            
            # Check properties
            properties = feature['properties']
            self.assertEqual(properties['detection_type'], 'fire_detection')
            self.assertIn(properties['class'], ['fire', 'smoke'])
            self.assertIn('confidence', properties)
            self.assertIn('detection_id', properties)
            self.assertEqual(properties['timestamp'], 1640995200.0)
            self.assertEqual(properties['fusion_used'], True)
    
    def test_fire_detections_to_geojson_empty(self):
        """Test fire detections to GeoJSON with empty detections"""
        empty_detections = {
            'boxes': [], 'classes': [], 'scores': [],
            'fusion_used': False, 'detection_count': 0
        }
        
        geojson = self.adapter.fire_detections_to_geojson(empty_detections, self.drone_pose)
        
        self.assertEqual(geojson['type'], 'FeatureCollection')
        self.assertEqual(len(geojson['features']), 0)
    
    def test_fauna_detections_to_geojson(self):
        """Test fauna detections to GeoJSON conversion"""
        fauna_detections = {
            'boxes': [[100, 150, 200, 250]],
            'species': ['deer'],
            'scores': [0.85],
            'health_status': ['healthy'],
            'detection_count': 1
        }
        
        density_map = np.random.poisson(0.5, (60, 80)).astype(np.float32)
        
        geojson = self.adapter.fauna_detections_to_geojson(
            fauna_detections, density_map, self.drone_pose
        )
        
        # Check structure
        self.assertEqual(geojson['type'], 'FeatureCollection')
        features = geojson['features']
        self.assertGreaterEqual(len(features), 1)  # At least the individual detection
        
        # Find the individual detection feature
        detection_feature = None
        for feature in features:
            if feature['properties']['detection_type'] == 'fauna_detection':
                detection_feature = feature
                break
        
        self.assertIsNotNone(detection_feature)
        
        # Check individual detection
        self.assertEqual(detection_feature['geometry']['type'], 'Point')
        properties = detection_feature['properties']
        self.assertEqual(properties['species'], 'deer')
        self.assertEqual(properties['health_status'], 'healthy')
        self.assertEqual(properties['confidence'], 0.85)
    
    def test_vegetation_health_to_geojson(self):
        """Test vegetation health to GeoJSON conversion"""
        health_patches = [
            {
                'health_class': 'healthy',
                'confidence': 0.85,
                'vari_index': 0.25,
                'probabilities': {'healthy': 0.85, 'dry': 0.10, 'burned': 0.05}
            },
            {
                'health_class': 'dry',
                'confidence': 0.70,
                'vari_index': 0.10,
                'probabilities': {'healthy': 0.15, 'dry': 0.70, 'burned': 0.15}
            }
        ]
        
        patch_locations = [(100, 100), (300, 200)]
        
        geojson = self.adapter.vegetation_health_to_geojson(
            health_patches, patch_locations, self.drone_pose
        )
        
        # Check structure
        self.assertEqual(geojson['type'], 'FeatureCollection')
        features = geojson['features']
        self.assertEqual(len(features), 2)
        
        for i, feature in enumerate(features):
            self.assertEqual(feature['geometry']['type'], 'Point')
            
            properties = feature['properties']
            self.assertEqual(properties['detection_type'], 'vegetation_health')
            self.assertEqual(properties['health_class'], health_patches[i]['health_class'])
            self.assertEqual(properties['confidence'], health_patches[i]['confidence'])
            self.assertEqual(properties['vari_index'], health_patches[i]['vari_index'])
    
    def test_fire_spread_to_geojson(self):
        """Test fire spread predictions to GeoJSON conversion"""
        # Create mock spread masks (3 time steps)
        spread_masks = np.zeros((3, 64, 64), dtype=np.float32)
        
        # Add some fire areas
        spread_masks[0, 20:25, 20:25] = 1.0  # t=1
        spread_masks[1, 18:27, 18:27] = 1.0  # t=2 (expanded)
        spread_masks[2, 15:30, 15:30] = 1.0  # t=3 (further expanded)
        
        geojson = self.adapter.fire_spread_to_geojson(spread_masks, self.drone_pose)
        
        # Check structure
        self.assertEqual(geojson['type'], 'FeatureCollection')
        features = geojson['features']
        
        # Should have features for the spread predictions
        self.assertGreater(len(features), 0)
        
        for feature in features:
            self.assertEqual(feature['type'], 'Feature')
            self.assertEqual(feature['geometry']['type'], 'Polygon')
            
            properties = feature['properties']
            self.assertEqual(properties['detection_type'], 'fire_spread_prediction')
            self.assertIn('time_horizon', properties)
            self.assertIn('polygon_id', properties)
    
    @patch('maps_adapter.cv2.findContours')
    def test_mask_to_contours(self, mock_find_contours):
        """Test mask to contours conversion"""
        # Mock cv2.findContours to return simple contours
        mock_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        mock_find_contours.return_value = ([mock_contour], None)
        
        # Create test mask
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        
        contours = self.adapter._mask_to_contours(mask)
        
        # Check that cv2.findContours was called
        mock_find_contours.assert_called_once()
        
        # Check result
        self.assertEqual(len(contours), 1)
        self.assertIsInstance(contours[0], np.ndarray)

class TestMainAdapterFunctions(unittest.TestCase):
    """Test main adapter functions for app integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        self.drone_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-10.0, yaw=45.0, timestamp=1640995200.0
        )
        
        self.fire_detections = {
            'boxes': [[100, 150, 200, 250]],
            'classes': [0],
            'scores': [0.85],
            'fusion_used': True,
            'detection_count': 1
        }
    
    def test_create_fire_overlay(self):
        """Test create_fire_overlay function"""
        geojson_str = create_fire_overlay(
            self.fire_detections, self.drone_pose, 
            self.intrinsics, 640, 480
        )
        
        # Check it returns a valid JSON string
        self.assertIsInstance(geojson_str, str)
        
        # Parse JSON to verify structure
        geojson = json.loads(geojson_str)
        self.assertEqual(geojson['type'], 'FeatureCollection')
        self.assertIn('features', geojson)
        
        # Check indentation (should be formatted)
        self.assertIn('\n', geojson_str)  # Multi-line formatted
    
    def test_create_fauna_overlay(self):
        """Test create_fauna_overlay function"""
        fauna_detections = {
            'boxes': [[100, 150, 200, 250]],
            'species': ['deer'],
            'scores': [0.85],
            'health_status': ['healthy'],
            'detection_count': 1
        }
        
        density_map = np.random.poisson(0.5, (60, 80)).astype(np.float32)
        
        geojson_str = create_fauna_overlay(
            fauna_detections, density_map, self.drone_pose,
            self.intrinsics, 640, 480
        )
        
        self.assertIsInstance(geojson_str, str)
        
        # Verify JSON validity
        geojson = json.loads(geojson_str)
        self.assertEqual(geojson['type'], 'FeatureCollection')
    
    def test_create_vegetation_overlay(self):
        """Test create_vegetation_overlay function"""
        health_patches = [{
            'health_class': 'healthy',
            'confidence': 0.85,
            'vari_index': 0.25,
            'probabilities': {'healthy': 0.85, 'dry': 0.10, 'burned': 0.05}
        }]
        
        patch_locations = [(100, 100)]
        
        geojson_str = create_vegetation_overlay(
            health_patches, patch_locations, self.drone_pose,
            self.intrinsics, 640, 480
        )
        
        self.assertIsInstance(geojson_str, str)
        
        # Verify JSON validity
        geojson = json.loads(geojson_str)
        self.assertEqual(geojson['type'], 'FeatureCollection')
    
    def test_create_spread_overlay(self):
        """Test create_spread_overlay function"""
        spread_masks = np.random.rand(12, 64, 64)
        
        geojson_str = create_spread_overlay(
            spread_masks, self.drone_pose, self.intrinsics, 640, 480
        )
        
        self.assertIsInstance(geojson_str, str)
        
        # Verify JSON validity
        geojson = json.loads(geojson_str)
        self.assertEqual(geojson['type'], 'FeatureCollection')

class TestExampleUsage(unittest.TestCase):
    """Test the example usage from the module"""
    
    def test_example_usage(self):
        """Test the example usage code doesn't crash"""
        # This tests the code in the __main__ section
        
        intrinsics = CameraIntrinsics(
            fx=800.0, fy=800.0, cx=640.0, cy=480.0,
            k1=-0.1, k2=0.05
        )
        
        drone_pose = DronePose(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-10.0, yaw=45.0, timestamp=1640995200
        )
        
        fire_detections = {
            'boxes': [[100, 150, 200, 250], [300, 300, 400, 400]],
            'classes': [0, 1],  # fire, smoke
            'scores': [0.85, 0.72],
            'fusion_used': True
        }
        
        # Should not raise any exceptions
        fire_geojson = create_fire_overlay(fire_detections, drone_pose, intrinsics, 1280, 960)
        
        self.assertIsInstance(fire_geojson, str)
        self.assertGreater(len(fire_geojson), 100)  # Should be substantial JSON

if __name__ == '__main__':
    unittest.main()