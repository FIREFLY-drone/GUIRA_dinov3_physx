# PhysX Prototype Usage Examples (PH-06)

This document provides practical examples for using the PhysX fire spread prototype.

## Quick Start

### Example 1: Basic Simulation Run

```bash
# Navigate to prototype directory
cd integrations/guira_core/simulation/physx_server/omniverse_prototype

# Run with default parameters (creates sample scene if needed)
python run_prototype.py \
  ../../../../samples/physx/scene.usd \
  ../../../../samples/physx/output.geojson

# Expected output:
# - samples/physx/scene.usd (auto-generated if missing)
# - samples/physx/output.geojson (20 timesteps)
```

### Example 2: Visualize Results

```bash
# Create visualization from output
python visualize_output.py \
  ../../../../samples/physx/prototype_output.geojson \
  ../../../../samples/physx/fire_viz.gif

# Creates:
# - fire_viz.png (static plot)
# - fire_viz.gif (animated evolution)
```

## Advanced Examples

### Example 3: Custom Python Script

```python
#!/usr/bin/env python3
"""Custom fire simulation with specific parameters."""

from run_prototype import FallbackPhysicsSimulator, generate_geojson_output

# Initialize simulator with custom grid
simulator = FallbackPhysicsSimulator(
    grid_size=(300, 300),  # Larger 300x300 grid
    cell_size=0.5          # Smaller 0.5m cells
)

# Set up scene
simulator.initialize_scene()

# Create dense vegetation (90% coverage)
simulator.create_vegetation_proxies(
    terrain_bounds=(-75, -75, 75, 75),
    density=0.9
)

# Define custom ignition area (line ignition)
ignition_polygon = [
    (-50, 0), (-40, 0), (-40, 2), (-50, 2)
]

# Spawn with strong wind
simulator.spawn_ember_particles(
    ignition_polygon,
    wind_direction=90.0,   # East wind
    wind_speed=15.0        # Strong 15 m/s wind
)

# Run longer simulation
states = simulator.simulate_steps(num_steps=200)

# Generate output
generate_geojson_output(states, 'custom_simulation.geojson')

print(f"Simulation complete: {len(states)} timesteps")
```

### Example 4: Multiple Ignition Points

```python
"""Simulate multiple spot fires merging."""

from run_prototype import FallbackPhysicsSimulator, generate_geojson_output

simulator = FallbackPhysicsSimulator()
simulator.initialize_scene()
simulator.create_vegetation_proxies((-100, -100, 100, 100), 0.75)

# Create three separate ignition areas
ignition_areas = [
    # Spot fire 1 (northwest)
    [(-40, 30), (-35, 30), (-35, 35), (-40, 35)],
    # Spot fire 2 (northeast)
    [(35, 30), (40, 30), (40, 35), (35, 35)],
    # Spot fire 3 (south)
    [(-5, -40), (5, -40), (5, -35), (-5, -35)]
]

# Ignite all areas
for polygon in ignition_areas:
    simulator.spawn_ember_particles(polygon, wind_direction=0, wind_speed=8)

# Simulate and watch fires merge
states = simulator.simulate_steps(num_steps=150)
generate_geojson_output(states, 'multi_ignition.geojson')
```

### Example 5: Parameter Sweep Study

```python
"""Study fire spread under different wind conditions."""

from run_prototype import FallbackPhysicsSimulator, generate_geojson_output
import json

wind_speeds = [0, 5, 10, 15, 20]  # m/s
results = []

for wind_speed in wind_speeds:
    print(f"Running simulation with wind={wind_speed} m/s...")
    
    simulator = FallbackPhysicsSimulator()
    simulator.initialize_scene()
    simulator.create_vegetation_proxies((-100, -100, 100, 100), 0.7)
    
    # Standard ignition
    ignition = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    simulator.spawn_ember_particles(ignition, 45, wind_speed)
    
    # Run simulation
    states = simulator.simulate_steps(100)
    
    # Save results
    output_path = f'wind_study_{wind_speed}ms.geojson'
    generate_geojson_output(states, output_path)
    
    # Extract metrics
    final_cells = states[-1]['fire_cells']
    results.append({
        'wind_speed_ms': wind_speed,
        'final_burning_cells': len(final_cells),
        'output_file': output_path
    })

# Save summary
with open('wind_study_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nWind Study Results:")
for r in results:
    print(f"  {r['wind_speed_ms']} m/s: {r['final_burning_cells']} cells")
```

### Example 6: Extract Fire Spread Rate

```python
"""Calculate fire spread rate from simulation."""

import json
import numpy as np

def calculate_spread_rate(geojson_path):
    """Calculate average fire spread rate."""
    with open(geojson_path) as f:
        data = json.load(f)
    
    features = data['features']
    times = []
    areas = []
    
    for feature in features:
        time = feature['properties']['time_seconds']
        num_cells = feature['properties']['num_cells']
        
        # Estimate area (assuming 1m² cells)
        area_m2 = num_cells
        
        times.append(time)
        areas.append(area_m2)
    
    # Calculate spread rate (m²/s)
    if len(times) > 1:
        time_diff = times[-1] - times[0]
        area_diff = areas[-1] - areas[0]
        spread_rate = area_diff / time_diff if time_diff > 0 else 0
        
        print(f"Fire Spread Analysis:")
        print(f"  Duration: {time_diff:.0f} seconds")
        print(f"  Initial area: {areas[0]:.0f} m²")
        print(f"  Final area: {areas[-1]:.0f} m²")
        print(f"  Spread rate: {spread_rate:.2f} m²/s")
        print(f"  Equivalent radius growth: {np.sqrt(spread_rate/np.pi):.2f} m/s")
        
        return spread_rate
    
    return 0

# Analyze prototype output
rate = calculate_spread_rate('samples/physx/prototype_output.geojson')
```

### Example 7: Compare with Real Fire Data

```python
"""Compare simulation with historical fire perimeter."""

import json
import numpy as np

def load_historical_perimeter(geojson_path):
    """Load real fire perimeter from field data."""
    with open(geojson_path) as f:
        data = json.load(f)
    # Extract coordinates from first feature
    coords = data['features'][0]['geometry']['coordinates'][0]
    return coords

def compare_perimeters(sim_geojson, historical_geojson):
    """Compare simulated vs. historical perimeter."""
    # Load both datasets
    with open(sim_geojson) as f:
        sim_data = json.load(f)
    
    historical_coords = load_historical_perimeter(historical_geojson)
    
    # Find best matching timestep
    best_match = None
    min_diff = float('inf')
    
    for feature in sim_data['features']:
        sim_coords = feature['geometry']['coordinates'][0]
        
        # Simple comparison: compare number of points
        diff = abs(len(sim_coords) - len(historical_coords))
        
        if diff < min_diff:
            min_diff = diff
            best_match = feature
    
    print(f"Best matching timestep: {best_match['properties']['timestep']}")
    print(f"Simulated cells: {best_match['properties']['num_cells']}")
    print(f"Historical perimeter points: {len(historical_coords)}")
    
    return best_match

# Example usage (with your own historical data)
# compare_perimeters('samples/physx/prototype_output.geojson', 
#                    'data/historical_fires/fire_20230815.geojson')
```

## Integration Examples

### Example 8: REST API Integration

```python
"""Simple Flask API for fire simulation."""

from flask import Flask, request, jsonify, send_file
from run_prototype import run_simulation
import tempfile
import os

app = Flask(__name__)

@app.route('/api/simulate', methods=['POST'])
def simulate_fire():
    """Run fire simulation from API request."""
    data = request.json
    
    # Extract parameters
    ignition_poly = data.get('ignition_polygon', [[-10,-10], [10,-10], [10,10], [-10,10]])
    wind_speed = data.get('wind_speed', 5.0)
    wind_direction = data.get('wind_direction', 45.0)
    num_steps = data.get('num_steps', 100)
    
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.usd', delete=False) as usd_file:
        usd_path = usd_file.name
    with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as geo_file:
        geo_path = geo_file.name
    
    try:
        # Run simulation
        run_simulation(usd_path, geo_path)
        
        # Read results
        with open(geo_path) as f:
            results = json.load(f)
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    finally:
        # Cleanup
        os.unlink(usd_path)
        os.unlink(geo_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Example 9: Batch Processing

```bash
#!/bin/bash
# Batch process multiple scenarios

SCENARIOS=(
  "low_wind:2:0"
  "medium_wind:8:45"
  "high_wind:15:90"
)

for scenario in "${SCENARIOS[@]}"; do
  IFS=':' read -r name wind_speed wind_dir <<< "$scenario"
  
  echo "Processing scenario: $name"
  echo "  Wind: ${wind_speed} m/s @ ${wind_dir}°"
  
  # Run simulation (would need to modify script to accept CLI args)
  python run_prototype.py \
    "samples/physx/scene_${name}.usd" \
    "samples/physx/output_${name}.geojson"
  
  # Visualize
  python visualize_output.py \
    "samples/physx/output_${name}.geojson" \
    "samples/physx/${name}.gif"
done

echo "Batch processing complete!"
```

## Testing Examples

### Example 10: Unit Test for Custom Function

```python
"""Test custom fire behavior."""

import unittest
from run_prototype import FallbackPhysicsSimulator

class TestCustomBehavior(unittest.TestCase):
    """Test specific fire behavior."""
    
    def test_uphill_spread(self):
        """Test that fire spreads faster uphill."""
        # This would require terrain/slope support
        # Placeholder for future enhancement
        pass
    
    def test_wind_directionality(self):
        """Test fire spreads faster downwind."""
        sim = FallbackPhysicsSimulator()
        sim.initialize_scene()
        
        # Set up east wind
        sim.wind_direction = np.radians(90)
        sim.wind_speed = 10
        
        # Ignite center
        sim.grid[100, 100] = 1.0
        
        # Run steps
        for _ in range(10):
            sim._spread_fire()
        
        # Check if more fire to the east
        west_fire = np.sum(sim.grid[90:100, 100])
        east_fire = np.sum(sim.grid[100:110, 100])
        
        self.assertGreater(east_fire, west_fire, 
                          "Fire should spread more downwind")

if __name__ == '__main__':
    unittest.main()
```

## Performance Tips

### Example 11: Optimize for Large Grids

```python
"""Performance optimization for large simulations."""

from run_prototype import FallbackPhysicsSimulator
import time

def benchmark_simulation(grid_size, num_steps):
    """Benchmark simulation performance."""
    print(f"Benchmarking {grid_size[0]}x{grid_size[1]} grid, {num_steps} steps...")
    
    start_time = time.time()
    
    simulator = FallbackPhysicsSimulator(grid_size=grid_size)
    simulator.initialize_scene()
    simulator.create_vegetation_proxies((-100, -100, 100, 100), 0.7)
    
    ignition = [(0, 0), (5, 0), (5, 5), (0, 5)]
    simulator.spawn_ember_particles(ignition)
    
    states = simulator.simulate_steps(num_steps)
    
    elapsed = time.time() - start_time
    
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Steps/sec: {num_steps/elapsed:.1f}")
    print(f"  Final cells: {len(states[-1]['fire_cells'])}")
    
    return elapsed

# Benchmark different grid sizes
for size in [50, 100, 200, 400]:
    benchmark_simulation((size, size), 50)
    print()
```

## Troubleshooting Examples

### Example 12: Debug Slow Spread

```python
"""Debug why fire isn't spreading as expected."""

from run_prototype import FallbackPhysicsSimulator
import numpy as np

def diagnose_spread():
    """Diagnose fire spread issues."""
    simulator = FallbackPhysicsSimulator()
    simulator.initialize_scene()
    
    # Check vegetation
    print("Vegetation Analysis:")
    print(f"  Mean density: {np.mean(simulator.vegetation):.3f}")
    print(f"  Max density: {np.max(simulator.vegetation):.3f}")
    print(f"  Cells with fuel > 0.3: {np.sum(simulator.vegetation > 0.3)}")
    
    # Ignite and check spread
    simulator.spawn_ember_particles([(0, 0), (1, 0), (1, 1), (0, 1)])
    
    print("\nFire Spread Analysis:")
    for step in range(10):
        burning_before = np.sum(simulator.grid > 0.5)
        simulator._spread_fire()
        burning_after = np.sum(simulator.grid > 0.5)
        
        print(f"  Step {step}: {burning_before} -> {burning_after} cells "
              f"(+{burning_after - burning_before})")
        
        if burning_after == burning_before:
            print("  WARNING: Fire stopped spreading!")
            break

diagnose_spread()
```

---

## Additional Resources

- **Documentation**: See `README.md` for full setup and API reference
- **Tests**: Run `python test_prototype.py` for validation
- **Source Code**: Review `run_prototype.py` for implementation details
- **Visualization**: Use `visualize_output.py` for output analysis

## Contributing

To add new examples:
1. Create your example script
2. Test thoroughly
3. Add documentation here
4. Submit PR with working example
