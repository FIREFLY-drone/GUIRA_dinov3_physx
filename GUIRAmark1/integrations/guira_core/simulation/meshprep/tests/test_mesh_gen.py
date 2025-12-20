"""
Unit tests for DEM to mesh conversion.

Tests the convert_dem_to_mesh module functionality including:
- Mesh generation from DEM
- Vertex and face count validation
- Metadata generation
- Mesh loadability
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_dem_to_mesh import dem_to_mesh


class TestMeshGeneration(unittest.TestCase):
    """Test suite for mesh generation from DEM."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not RASTERIO_AVAILABLE:
            raise unittest.SkipTest("rasterio not available")
        if not TRIMESH_AVAILABLE:
            raise unittest.SkipTest("trimesh not available")
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)
        
        # Create sample DEM
        cls.dem_path = cls.temp_path / "test_dem.tif"
        cls._create_sample_dem(cls.dem_path)
        
        # Create sample vegetation raster
        cls.veg_path = cls.temp_path / "test_veg.tif"
        cls._create_sample_vegetation(cls.veg_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    @staticmethod
    def _create_sample_dem(output_path: Path, width: int = 50, height: int = 50):
        """Create a sample DEM GeoTIFF for testing."""
        # Create synthetic elevation data (simple gradient with some noise)
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        xx, yy = np.meshgrid(x, y)
        
        # Elevation: gradient + sinusoidal pattern
        elevation = 100 + 10 * xx + 5 * yy + 3 * np.sin(xx) * np.cos(yy)
        elevation = elevation.astype(np.float32)
        
        # Define geographic bounds
        xmin, ymin, xmax, ymax = -120.0, 35.0, -119.0, 36.0
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=elevation.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(elevation, 1)
    
    @staticmethod
    def _create_sample_vegetation(output_path: Path, width: int = 50, height: int = 50):
        """Create a sample vegetation raster for testing."""
        # Create synthetic vegetation/fuel load data
        veg_data = np.random.rand(height, width) * 10.0  # Fuel load 0-10
        veg_data = veg_data.astype(np.float32)
        
        # Define geographic bounds (same as DEM)
        xmin, ymin, xmax, ymax = -120.0, 35.0, -119.0, 36.0
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=veg_data.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(veg_data, 1)
    
    def test_basic_mesh_generation(self):
        """Test basic mesh generation from DEM."""
        output_obj = self.temp_path / "test_mesh.obj"
        output_meta = self.temp_path / "test_mesh_meta.json"
        
        # Generate mesh
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Check outputs exist
        self.assertTrue(Path(mesh_path).exists(), "Mesh file should exist")
        self.assertTrue(Path(meta_path).exists(), "Metadata file should exist")
    
    def test_mesh_loadable(self):
        """Test that generated mesh is loadable by trimesh."""
        output_obj = self.temp_path / "test_loadable.obj"
        output_meta = self.temp_path / "test_loadable_meta.json"
        
        # Generate mesh
        mesh_path, _ = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Verify mesh properties
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 0, "Mesh should have vertices")
        self.assertGreater(len(mesh.faces), 0, "Mesh should have faces")
    
    def test_vertex_count(self):
        """Test that vertex count matches DEM dimensions."""
        output_obj = self.temp_path / "test_vertices.obj"
        output_meta = self.temp_path / "test_vertices_meta.json"
        
        # Get DEM dimensions
        with rasterio.open(self.dem_path) as ds:
            h, w = ds.shape
        
        # Generate mesh
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Check vertex count: should be h * w
        expected_vertices = h * w
        self.assertEqual(
            len(mesh.vertices),
            expected_vertices,
            f"Vertex count should be {expected_vertices} (h={h}, w={w})"
        )
        
        # Load metadata and verify
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.assertEqual(meta['vertices'], expected_vertices)
    
    def test_face_count(self):
        """Test that face count matches expected triangulation."""
        output_obj = self.temp_path / "test_faces.obj"
        output_meta = self.temp_path / "test_faces_meta.json"
        
        # Get DEM dimensions
        with rasterio.open(self.dem_path) as ds:
            h, w = ds.shape
        
        # Generate mesh
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Check face count: should be 2 * (h-1) * (w-1)
        expected_faces = 2 * (h - 1) * (w - 1)
        self.assertEqual(
            len(mesh.faces),
            expected_faces,
            f"Face count should be {expected_faces} (2*(h-1)*(w-1))"
        )
        
        # Load metadata and verify
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.assertEqual(meta['faces'], expected_faces)
    
    def test_metadata_content(self):
        """Test that metadata contains expected fields."""
        output_obj = self.temp_path / "test_metadata.obj"
        output_meta = self.temp_path / "test_metadata_meta.json"
        
        # Generate mesh
        _, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check required fields
        required_fields = [
            'vertices', 'faces', 'dem_shape', 'dem_path',
            'crs', 'bounds', 'elevation_range', 'scale', 'subsample'
        ]
        
        for field in required_fields:
            self.assertIn(field, meta, f"Metadata should contain '{field}'")
        
        # Check specific values
        self.assertIsInstance(meta['vertices'], int)
        self.assertIsInstance(meta['faces'], int)
        self.assertGreater(meta['vertices'], 0)
        self.assertGreater(meta['faces'], 0)
    
    def test_with_vegetation(self):
        """Test mesh generation with vegetation data."""
        output_obj = self.temp_path / "test_veg.obj"
        output_meta = self.temp_path / "test_veg_meta.json"
        
        # Generate mesh with vegetation
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            veg_path=str(self.veg_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta)
        )
        
        # Check outputs exist
        self.assertTrue(Path(mesh_path).exists())
        self.assertTrue(Path(meta_path).exists())
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check vegetation flag
        self.assertTrue(meta['has_vegetation'], "Should indicate vegetation data was used")
    
    def test_scale_parameter(self):
        """Test vertical scaling of elevation."""
        output_obj = self.temp_path / "test_scale.obj"
        output_meta = self.temp_path / "test_scale_meta.json"
        
        scale_factor = 2.0
        
        # Generate mesh with scaling
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta),
            scale=scale_factor
        )
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check scale is recorded
        self.assertEqual(meta['scale'], scale_factor)
    
    def test_subsample_parameter(self):
        """Test mesh subsampling."""
        output_obj = self.temp_path / "test_subsample.obj"
        output_meta = self.temp_path / "test_subsample_meta.json"
        
        subsample = 2
        
        # Get original DEM dimensions
        with rasterio.open(self.dem_path) as ds:
            h, w = ds.shape
        
        # Generate mesh with subsampling
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(self.dem_path),
            out_obj=str(output_obj),
            out_meta=str(output_meta),
            subsample=subsample
        )
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Calculate expected dimensions after subsampling
        h_sub = h // subsample
        w_sub = w // subsample
        expected_vertices = h_sub * w_sub
        
        # Check vertex count is reduced
        self.assertLess(
            len(mesh.vertices),
            h * w,
            "Subsampling should reduce vertex count"
        )
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.assertEqual(meta['subsample'], subsample)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMeshGeneration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
