# PhysX Fire Spread Prototype (PH-06)

Rapid prototype using NVIDIA Omniverse omni.physx Python API to validate physics interactions and particle models for fire spread simulation.

## Overview

This prototype demonstrates:
- Loading USD terrain scenes
- Creating combustible vegetation proxies
- Seeding ignition polygons and spawning ember particles
- Producing time-series fire perimeter GeoJSON output
- Sanity-checking physics interactions and particle models

## MODEL

**Model:** NVIDIA Omniverse PhysX particle system + physics scene  
**Version:** omni.physx Python API (Omniverse Kit compatible)  
**Fallback:** Cellular automaton fire spread simulator

## DATA

**Input:**
- USD scene files with terrain meshes
- Ignition polygon coordinates
- Environmental parameters (wind speed, direction, vegetation density)

**Output:**
- GeoJSON FeatureCollection with time-series fire perimeters
- Metadata: timestep, time_seconds, num_cells, perimeter_length

**Sample Data:**
- `samples/physx/scene.usd` - Minimal terrain scene (auto-generated)
- `samples/physx/prototype_output.geojson` - Example output

## TRAINING/BUILD RECIPE

**No training required** - physics-based particle simulation.

### Prerequisites

#### Option 1: Full Omniverse Setup (Preferred)

1. Install NVIDIA Omniverse Launcher from [developer.nvidia.com/omniverse](https://developer.nvidia.com/omniverse)
2. Install required Omniverse packages:
   ```bash
   # From Omniverse Launcher, install:
   # - Kit SDK
   # - USD Composer (formerly Create)
   # - PhysX extensions
   ```

3. Install Python dependencies:
   ```bash
   pip install pxr omni-usd omni-physx
   ```

#### Option 2: Fallback Simulation (No Omniverse)

Only requires standard scientific Python packages (already in requirements.txt):
```bash
pip install numpy
```

The script automatically detects if Omniverse is available and falls back to a cellular automaton simulator if not.

## EVALUATION & ACCEPTANCE

### Acceptance Criteria

✅ **Criterion 1:** Script executes without errors  
✅ **Criterion 2:** Produces valid GeoJSON output with fire perimeter evolution  
✅ **Criterion 3:** GeoJSON contains multiple timesteps showing fire spread  
✅ **Criterion 4:** Output includes particle/cell positions and perimeter polygons  
✅ **Criterion 5:** Works with both Omniverse and fallback modes  

### Expected Output Format

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "simulation_type": "physx_prototype",
    "total_timesteps": 20,
    "description": "Fire spread simulation perimeter evolution"
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestep": 0,
        "time_seconds": 0.0,
        "num_cells": 45,
        "perimeter_length_m": 120.5
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], ...]]
      }
    },
    ...
  ]
}
```

## Usage

### Basic Usage

```bash
cd integrations/guira_core/simulation/physx_server/omniverse_prototype

# Run with automatic sample scene creation
python run_prototype.py ../../samples/physx/scene.usd ../../samples/physx/prototype_output.geojson
```

### Advanced Usage

```python
from run_prototype import run_simulation

# Custom simulation
success = run_simulation(
    usd_path="path/to/custom_scene.usd",
    output_geojson="output/fire_perimeter.geojson"
)
```

### Script Parameters

- **usd_path** (arg 1): Path to USD scene file (auto-creates if missing)
- **output_geojson** (arg 2): Output path for GeoJSON results

### Environmental Parameters

Configurable in code (future: CLI arguments):
- `terrain_bounds`: Simulation area in meters (default: 200m x 200m)
- `vegetation_density`: Fuel density 0-1 (default: 0.7)
- `wind_direction`: Wind direction in degrees (default: 45° = NE)
- `wind_speed`: Wind speed in m/s (default: 5.0)
- `num_steps`: Simulation steps (default: 100)

## Implementation Details

### Omniverse Mode

When Omniverse is available:
1. Loads USD scene using `omni.usd` and `pxr.Usd`
2. Creates physics scene with `omni.physx.scripts.utils`
3. Spawns vegetation proxies as physics objects
4. Creates particle emitter at ignition points
5. Steps simulation and tracks fire spread
6. Extracts perimeter from burning objects

### Fallback Mode

When Omniverse is NOT available:
1. Uses cellular automaton (CA) on 2D grid
2. Each cell represents terrain patch (1m²)
3. Fire spread rules:
   - Ignition probability based on burning neighbors
   - Modified by vegetation fuel density
   - Wind direction affects spread rate
   - Fire intensity decays over time
4. Converts burning cells to perimeter polygons

### Fire Spread Model

**Physics-based factors:**
- Neighbor influence (0-4 burning neighbors)
- Fuel availability (vegetation density)
- Wind speed and direction
- Fire intensity decay

**Spread equation (fallback mode):**
```
P(ignition) = 0.2 × n_burning × fuel_density × wind_factor
wind_factor = 1.0 + 0.3 × (wind_speed / 10)
```

## Output Validation

### Validate GeoJSON Structure

```bash
# Check output file exists and is valid JSON
python -c "import json; print(json.load(open('samples/physx/prototype_output.geojson'))['type'])"
# Should output: FeatureCollection
```

### Visualize Output

```python
import json
import matplotlib.pyplot as plt

with open('samples/physx/prototype_output.geojson') as f:
    data = json.load(f)

# Plot fire perimeter evolution
fig, ax = plt.subplots(figsize=(10, 10))
for i, feature in enumerate(data['features']):
    coords = feature['geometry']['coordinates'][0]
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    ax.plot(x, y, label=f"t={feature['properties']['time_seconds']:.1f}s")

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('Fire Perimeter Evolution')
ax.legend()
ax.grid(True)
plt.savefig('fire_perimeter_evolution.png')
```

## Known Limitations

1. **Omniverse Mode:** Requires NVIDIA GPU and Omniverse installation
2. **Particle Rendering:** GIF/screenshot generation requires Omniverse viewport access (manual step)
3. **Terrain Interaction:** Current prototype uses simplified terrain (flat plane)
4. **Performance:** Fallback mode is optimized for speed over accuracy
5. **Wind Model:** Simplified directional bias, not full CFD

## Future Enhancements

- [ ] Real DEM terrain integration
- [ ] Variable vegetation types and burn rates
- [ ] Spotting/ember transport physics
- [ ] Heat transfer and radiation models
- [ ] Multi-GPU scaling for large domains
- [ ] Real-time visualization server
- [ ] Integration with weather data APIs

## Security & Privacy

- **Credentials:** Omniverse may require NVIDIA account login
- **Data:** No credentials stored in code or output files
- **Privacy:** All simulation data is synthetic; no real-world PII

## References

- [NVIDIA Omniverse Documentation](https://docs.omniverse.nvidia.com/)
- [PhysX SDK Documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/index.html)
- [USD (Universal Scene Description)](https://graphics.pixar.com/usd/docs/index.html)
- Fire spread models: Rothermel (1972), Finney (1998)

## Troubleshooting

### "Omniverse modules not available"
- **Cause:** Omniverse not installed or not in Python path
- **Solution:** Install Omniverse or use fallback mode (automatic)

### "USD scene not found"
- **Cause:** Missing input file
- **Solution:** Script auto-creates sample scene; or provide valid USD file

### Empty GeoJSON output
- **Cause:** Simulation not spreading (low ignition probability)
- **Solution:** Increase vegetation density or wind speed parameters

### Permission errors
- **Cause:** Output directory not writable
- **Solution:** Ensure output path is writable; create directories if needed

## Contact

For issues or questions about this prototype, see repository documentation or file an issue.
