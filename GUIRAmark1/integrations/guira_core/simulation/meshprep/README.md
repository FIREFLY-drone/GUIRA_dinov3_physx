# Mesh Preparation for PhysX Fire Simulation

This module converts Digital Elevation Model (DEM) GeoTIFF files into 3D terrain meshes suitable for PhysX-based fire spread simulation in NVIDIA Omniverse.

## Features

- Convert DEM GeoTIFF to 3D mesh (.obj format)
- Optional vegetation/fuel load integration
- Per-vertex attribute support (fuel load, moisture)
- Configurable vertical scaling and mesh density
- Output metadata for simulation parameters
- PhysX/Omniverse compatible mesh format

## MODEL

**Model**: Geometric mesh generation from raster data  
**Version**: 1.0  
**Algorithm**: Grid triangulation with bilinear interpolation  
**Output formats**: OBJ mesh + JSON metadata  
**Future formats**: USD (Universal Scene Description) for Omniverse

## DATA

**Input data**:
- DEM GeoTIFF: Digital Elevation Model with CRS metadata
- Optional vegetation raster: Fuel load/moisture data
- Coordinate Reference System (CRS): EPSG codes embedded in GeoTIFF

**Local paths**:
- Sample DEM: `data/dem/sample_dem.tif`
- Sample outputs: `integrations/guira_core/samples/mesh/`

## TRAINING/BUILD RECIPE

N/A - This is a geometric conversion module with no training required.

**Build requirements**:
- Python 3.8+
- rasterio >= 1.3.0
- trimesh >= 3.21.0
- numpy >= 1.24.0

**Installation**:
```bash
pip install rasterio trimesh numpy
```

## EVALUATION & ACCEPTANCE

### Acceptance Criteria

1. **Mesh loadability**: Output mesh must be loadable via `trimesh.load()`
2. **Vertex count**: Must equal DEM dimensions: `vertices = height × width`
3. **Face count**: Must equal grid triangulation: `faces = 2 × (height-1) × (width-1)`
4. **Metadata completeness**: JSON must include vertices, faces, DEM shape, CRS, bounds, elevation range
5. **Attribute preservation**: Vegetation/fuel data preserved as vertex attributes

### Test Suite

Run unit tests:
```bash
cd integrations/guira_core/simulation/meshprep/tests
python test_mesh_gen.py
```

Expected output: All 8 tests pass

## Usage

### Command Line Interface

#### Basic usage (Python script):
```bash
python convert_dem_to_mesh.py --dem terrain.tif --out mesh.obj
```

#### Basic usage (Shell wrapper):
```bash
./make_mesh.sh terrain.tif mesh.obj
```

#### With vegetation data:
```bash
python convert_dem_to_mesh.py \
    --dem terrain.tif \
    --veg vegetation.tif \
    --out mesh.obj
```

#### With scaling and subsampling:
```bash
python convert_dem_to_mesh.py \
    --dem terrain.tif \
    --out mesh.obj \
    --scale 2.0 \
    --subsample 2
```

### Example: Generate sample mesh

```bash
cd integrations/guira_core/simulation/meshprep

# Using existing sample DEM
python convert_dem_to_mesh.py \
    --dem ../../../../data/dem/sample_dem.tif \
    --out ../../samples/mesh/sample_tile.obj \
    --subsample 10

# Verify mesh is loadable
python -c "import trimesh; mesh = trimesh.load('../../samples/mesh/sample_tile.obj'); print(f'Loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces')"
```

### Python API

```python
from convert_dem_to_mesh import dem_to_mesh

# Convert DEM to mesh
mesh_path, meta_path = dem_to_mesh(
    dem_path='terrain.tif',
    veg_path='vegetation.tif',  # Optional
    out_obj='output.obj',
    out_meta='output_meta.json',
    scale=1.0,                   # Vertical exaggeration
    subsample=1                  # Reduce mesh density
)

# Load and inspect
import trimesh
mesh = trimesh.load(mesh_path)
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.faces)}")
```

## Parameters

### `dem_to_mesh()`

- `dem_path` (str, required): Path to input DEM GeoTIFF
- `veg_path` (str, optional): Path to vegetation/fuel raster
- `out_obj` (str): Output mesh file path (.obj) [default: 'out.obj']
- `out_meta` (str): Output metadata JSON path [default: 'out_meta.json']
- `scale` (float): Vertical scale factor for elevation [default: 1.0]
- `subsample` (int): Subsample factor to reduce mesh density [default: 1]

## Output Format

### Mesh File (.obj)

Standard Wavefront OBJ format with:
- Vertices: 3D coordinates (X, Y, Z) in source CRS
- Faces: Triangulated grid (counter-clockwise winding)
- Vertex attributes: Fuel load (if vegetation data provided)

### Metadata File (.json)

```json
{
  "vertices": 10000,
  "faces": 19602,
  "dem_shape": {
    "height": 100,
    "width": 100
  },
  "dem_path": "/path/to/dem.tif",
  "crs": "EPSG:4326",
  "bounds": {
    "left": -122.5,
    "bottom": 37.7,
    "right": -122.3,
    "top": 37.9
  },
  "elevation_range": {
    "min": 0.0,
    "max": 722.27
  },
  "scale": 1.0,
  "subsample": 10,
  "has_vegetation": false
}
```

## PhysX/Omniverse Integration

### Current Support
- OBJ mesh format (compatible with PhysX and Omniverse)
- Vertex attributes for per-point properties
- Geographic coordinate preservation

### Future: USD Export

For native Omniverse integration, USD (Universal Scene Description) export will be added:

```python
# Future API
from convert_dem_to_mesh import dem_to_mesh_usd

mesh_path = dem_to_mesh_usd(
    dem_path='terrain.tif',
    out_usd='terrain.usd',
    # USD-specific options
    meters_per_unit=1.0,
    up_axis='Z'
)
```

## Performance Considerations

### Large DEMs

For large DEMs (>1000×1000), use subsampling:

```bash
# Full resolution: 1M vertices, ~2M faces
python convert_dem_to_mesh.py --dem large.tif --out full.obj

# Subsampled 10x: 10K vertices, ~20K faces
python convert_dem_to_mesh.py --dem large.tif --out reduced.obj --subsample 10
```

### Memory Usage

- **Vertices**: ~24 bytes per vertex (3 float64 coordinates)
- **Faces**: ~12 bytes per face (3 int32 indices)
- **Example**: 1000×1000 DEM → ~60 MB memory

## Security & Privacy

- DEM and vegetation data may be sensitive (infrastructure locations, protected lands)
- Access via secure blob storage and KeyVault
- Embed appropriate metadata licenses in outputs
- Never commit raw geospatial data to version control

## References

- **PhysX Documentation**: https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/Index.html
- **NVIDIA Omniverse**: https://developer.nvidia.com/omniverse
- **USD Format**: https://graphics.pixar.com/usd/docs/index.html
- **Rasterio**: https://rasterio.readthedocs.io/
- **Trimesh**: https://trimsh.org/

## Contributing

When modifying mesh generation:
1. Update unit tests in `tests/test_mesh_gen.py`
2. Verify acceptance criteria still met
3. Document any new parameters or outputs
4. Update this README

## License

See repository LICENSE file.
