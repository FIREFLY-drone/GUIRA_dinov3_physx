#!/usr/bin/env python3
"""
Example usage of DEM to mesh conversion for fire simulation.

This script demonstrates the complete workflow from DEM to mesh
and how to integrate with PhysX fire simulation.
"""

import sys
from pathlib import Path
import json

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from convert_dem_to_mesh import dem_to_mesh


def example_basic_conversion():
    """Example 1: Basic DEM to mesh conversion."""
    print("=" * 60)
    print("Example 1: Basic DEM to mesh conversion")
    print("=" * 60)
    
    # Convert DEM to mesh
    mesh_path, meta_path = dem_to_mesh(
        dem_path='../../../../data/dem/sample_dem.tif',
        out_obj='../../samples/mesh/example_basic.obj',
        out_meta='../../samples/mesh/example_basic_meta.json',
        subsample=20  # Reduce mesh density for demonstration
    )
    
    print(f"\n✅ Mesh generated: {mesh_path}")
    print(f"✅ Metadata saved: {meta_path}")
    
    # Load and display metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {meta['vertices']:,}")
    print(f"  Faces: {meta['faces']:,}")
    print(f"  DEM shape: {meta['dem_shape']['height']} × {meta['dem_shape']['width']}")
    print(f"  Elevation range: {meta['elevation_range']['min']:.1f} - {meta['elevation_range']['max']:.1f} m")
    print(f"  CRS: {meta['crs']}")
    

def example_with_vegetation():
    """Example 2: Mesh with vegetation/fuel data."""
    print("\n" + "=" * 60)
    print("Example 2: Mesh with vegetation/fuel data")
    print("=" * 60)
    print("\nNote: This example requires a vegetation raster file.")
    print("When available, use:")
    print("""
    mesh_path, meta_path = dem_to_mesh(
        dem_path='terrain.tif',
        veg_path='vegetation.tif',  # Fuel load raster
        out_obj='mesh_with_fuel.obj',
        subsample=10
    )
    """)


def example_with_scaling():
    """Example 3: Vertical scaling for visualization."""
    print("\n" + "=" * 60)
    print("Example 3: Vertical scaling for visualization")
    print("=" * 60)
    
    # Create mesh with vertical exaggeration
    mesh_path, meta_path = dem_to_mesh(
        dem_path='../../../../data/dem/sample_dem.tif',
        out_obj='../../samples/mesh/example_scaled.obj',
        out_meta='../../samples/mesh/example_scaled_meta.json',
        scale=2.0,      # 2x vertical exaggeration
        subsample=25
    )
    
    print(f"\n✅ Scaled mesh generated: {mesh_path}")
    print(f"   (Vertical scale: 2.0x)")


def example_physx_integration():
    """Example 4: Integration with PhysX simulation (conceptual)."""
    print("\n" + "=" * 60)
    print("Example 4: PhysX integration workflow (conceptual)")
    print("=" * 60)
    
    print("""
    # Step 1: Generate mesh from DEM
    mesh_path, meta_path = dem_to_mesh(
        dem_path='terrain.tif',
        veg_path='fuel_load.tif',
        out_obj='simulation_terrain.obj',
        subsample=5  # Balance between detail and performance
    )
    
    # Step 2: Load mesh for PhysX (future implementation)
    # import trimesh
    # mesh = trimesh.load(mesh_path)
    
    # Step 3: Initialize PhysX simulation
    # from ..physx_server import PhysXFireSimulator
    # simulator = PhysXFireSimulator(scene_path=mesh_path)
    
    # Step 4: Set up simulation parameters
    # simulator.initialize_simulation(
    #     wind_params={'speed': 5.0, 'direction': 45.0},
    #     weather_params={'temperature': 30.0, 'humidity': 0.3}
    # )
    
    # Step 5: Run fire spread simulation
    # results = simulator.simulate_fire_spread(
    #     ignition_points=[(100.0, 200.0)],
    #     duration=3600.0,  # 1 hour
    #     time_step=1.0
    # )
    
    # Step 6: Export results
    # simulator.export_results('fire_spread_results.json')
    """)
    print("\n⚠️  Note: Full PhysX integration is in development.")


def example_batch_processing():
    """Example 5: Batch processing multiple DEMs."""
    print("\n" + "=" * 60)
    print("Example 5: Batch processing (template)")
    print("=" * 60)
    
    print("""
    from pathlib import Path
    
    # Process multiple DEM files
    dem_dir = Path('dems/')
    output_dir = Path('meshes/')
    output_dir.mkdir(exist_ok=True)
    
    for dem_file in dem_dir.glob('*.tif'):
        print(f"Processing {dem_file.name}...")
        
        mesh_path, meta_path = dem_to_mesh(
            dem_path=str(dem_file),
            out_obj=str(output_dir / f"{dem_file.stem}.obj"),
            subsample=10
        )
        
        print(f"  ✅ Generated: {mesh_path}")
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DEM to Mesh Conversion - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_conversion()
        example_with_vegetation()
        example_with_scaling()
        example_physx_integration()
        example_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - README.md (module documentation)")
        print("  - convert_dem_to_mesh.py --help (CLI help)")
        print("  - tests/test_mesh_gen.py (unit tests)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
