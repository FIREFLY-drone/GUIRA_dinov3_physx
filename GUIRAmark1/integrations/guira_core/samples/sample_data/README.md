# GUIRA Core Sample Data

This directory contains sample data for testing and development of GUIRA core components.

## Data Types

### Vision Data
- **sample_satellite.jpg**: Sample satellite imagery for DINOv3 processing
- **sample_drone_sequence/**: Video frames for TimeSFormer testing
- **sample_thermal.tif**: Thermal imagery for fire detection

### Simulation Data
- **terrain_dem.tif**: Digital elevation model for mesh preparation
- **vegetation_map.tif**: Vegetation classification raster
- **fuel_load.tif**: Fuel density distribution
- **wind_profile.json**: Sample wind data

### Test Cases
- **test_fire_scenario.json**: Complete fire detection + simulation test
- **test_embeddings.npy**: Pre-computed DINOv3 embeddings
- **expected_results/**: Expected outputs for validation

## Usage

```python
# Load sample data
from pathlib import Path
import numpy as np

data_dir = Path("samples/sample_data")
sample_image = data_dir / "sample_satellite.jpg"
dem_data = data_dir / "terrain_dem.tif"
```

## Data Sources

- Satellite imagery: Landsat 8/9 OLI
- DEM data: SRTM 30m resolution
- Vegetation maps: NLCD 2019
- Weather data: NOAA/NWS forecasts

## Licenses

All sample data is either public domain or used under appropriate licenses for testing purposes. See individual file headers for specific attribution.