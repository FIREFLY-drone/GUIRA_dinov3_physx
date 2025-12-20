#!/usr/bin/env python3
"""
Convert DEM to 3D Mesh for PhysX Fire Simulation

Converts DEM GeoTIFF + vegetation raster → 3D terrain mesh (.obj / .usd) 
+ per-vertex attributes (fuel load, moisture).

MODEL: Mesh generation from raster data using trimesh
DATA: DEM GeoTIFF, optional vegetation raster
TRAINING/BUILD RECIPE: N/A (geometric conversion, no training)
EVAL & ACCEPTANCE: 
  - Mesh file loadable via trimesh
  - Vertex count matches DEM cell count (h*w)
  - Face count = 2*(h-1)*(w-1)
  - Metadata JSON includes vertex/face counts
"""

import rasterio
import numpy as np
import trimesh
import json
from pathlib import Path
import argparse
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def dem_to_mesh(
    dem_path: str,
    veg_path: Optional[str] = None,
    out_obj: str = 'out.obj',
    out_meta: str = 'out_meta.json',
    scale: float = 1.0,
    subsample: int = 1
) -> Tuple[str, str]:
    """
    Convert DEM GeoTIFF to 3D mesh with optional vegetation attributes.
    
    Args:
        dem_path: Path to input DEM GeoTIFF
        veg_path: Optional path to vegetation raster
        out_obj: Output mesh file path (.obj)
        out_meta: Output metadata JSON path
        scale: Vertical scale factor for elevation (default 1.0)
        subsample: Subsample factor to reduce mesh density (default 1)
        
    Returns:
        Tuple of (mesh_path, metadata_path)
        
    Raises:
        FileNotFoundError: If DEM file doesn't exist
        ValueError: If DEM data is invalid
    """
    logger.info(f"Loading DEM from {dem_path}")
    
    # Load DEM data
    with rasterio.open(dem_path) as ds:
        arr = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        
        # Get bounds
        bounds = ds.bounds
        
    logger.info(f"DEM shape: {arr.shape}, CRS: {crs}")
    logger.info(f"Elevation range: {np.nanmin(arr):.2f} to {np.nanmax(arr):.2f}")
    
    # Subsample if requested
    if subsample > 1:
        arr = arr[::subsample, ::subsample]
        logger.info(f"Subsampled to shape: {arr.shape}")
    
    h, w = arr.shape
    
    # Load vegetation data if provided
    veg_data = None
    if veg_path:
        logger.info(f"Loading vegetation data from {veg_path}")
        with rasterio.open(veg_path) as veg_ds:
            veg_data = veg_ds.read(1)
            if subsample > 1:
                veg_data = veg_data[::subsample, ::subsample]
            if veg_data.shape != arr.shape:
                logger.warning(
                    f"Vegetation shape {veg_data.shape} doesn't match DEM {arr.shape}. "
                    "Vegetation data will be ignored."
                )
                veg_data = None
    
    # Build grid vertices
    logger.info("Building mesh vertices...")
    verts = []
    fuel_load = []  # Per-vertex fuel load attribute
    
    for i in range(h):
        for j in range(w):
            # Get world coordinates using rasterio transform
            x, y = rasterio.transform.xy(transform, i * subsample, j * subsample, offset='center')
            z = float(arr[i, j]) * scale
            
            # Handle NaN values
            if np.isnan(z):
                z = 0.0
                
            verts.append([x, y, z])
            
            # Add fuel load from vegetation if available
            if veg_data is not None:
                fuel = float(veg_data[i, j])
                if np.isnan(fuel):
                    fuel = 0.0
                fuel_load.append(fuel)
            else:
                fuel_load.append(0.0)
    
    verts = np.array(verts, dtype=np.float64)
    fuel_load = np.array(fuel_load, dtype=np.float32)
    
    logger.info(f"Created {verts.shape[0]} vertices")
    
    # Build faces (simple grid triangulation)
    logger.info("Building mesh faces...")
    faces = []
    
    for i in range(h - 1):
        for j in range(w - 1):
            # Vertex indices for quad
            v0 = i * w + j
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            
            # Two triangles per quad (counter-clockwise winding)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    faces = np.array(faces, dtype=np.int32)
    logger.info(f"Created {len(faces)} faces")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    # Store fuel load as vertex attribute
    if veg_data is not None:
        mesh.vertex_attributes['fuel_load'] = fuel_load
    
    # Export mesh
    logger.info(f"Exporting mesh to {out_obj}")
    mesh.export(out_obj)
    
    # Create metadata
    meta = {
        "vertices": int(verts.shape[0]),
        "faces": int(len(faces)),
        "dem_shape": {"height": int(h), "width": int(w)},
        "dem_path": str(dem_path),
        "crs": str(crs) if crs else None,
        "bounds": {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top)
        } if bounds else None,
        "elevation_range": {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr))
        },
        "scale": float(scale),
        "subsample": int(subsample),
        "has_vegetation": veg_data is not None
    }
    
    # Save metadata
    logger.info(f"Saving metadata to {out_meta}")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info("Mesh generation complete!")
    
    return out_obj, out_meta


def main():
    """Command-line interface for DEM to mesh conversion."""
    parser = argparse.ArgumentParser(
        description='Convert DEM GeoTIFF to 3D mesh for PhysX simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_dem_to_mesh.py --dem terrain.tif --out mesh.obj
  
  # With vegetation data
  python convert_dem_to_mesh.py --dem terrain.tif --veg vegetation.tif --out mesh.obj
  
  # With vertical scaling and subsampling
  python convert_dem_to_mesh.py --dem terrain.tif --out mesh.obj --scale 2.0 --subsample 2
        """
    )
    
    parser.add_argument(
        '--dem',
        required=True,
        help='Path to input DEM GeoTIFF file'
    )
    parser.add_argument(
        '--veg',
        default=None,
        help='Path to optional vegetation raster file'
    )
    parser.add_argument(
        '--out',
        default='tile.obj',
        help='Output mesh file path (default: tile.obj)'
    )
    parser.add_argument(
        '--meta',
        default=None,
        help='Output metadata JSON path (default: <out_base>_meta.json)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Vertical scale factor for elevation (default: 1.0)'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=1,
        help='Subsample factor to reduce mesh density (default: 1, no subsampling)'
    )
    
    args = parser.parse_args()
    
    # Determine metadata path
    if args.meta is None:
        out_path = Path(args.out)
        args.meta = str(out_path.parent / f"{out_path.stem}_meta.json")
    
    # Convert DEM to mesh
    try:
        dem_to_mesh(
            dem_path=args.dem,
            veg_path=args.veg,
            out_obj=args.out,
            out_meta=args.meta,
            scale=args.scale,
            subsample=args.subsample
        )
    except Exception as e:
        logger.error(f"Failed to convert DEM to mesh: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
