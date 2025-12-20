"""
Initialize sample forecast data for API testing

Creates sample forecast metadata and copies sample GeoJSON from the samples directory.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Paths
API_DIR = Path(__file__).parent
DATA_DIR = API_DIR / "data" / "forecasts"
SAMPLE_GEOJSON = Path(__file__).parent.parent.parent / "samples" / "physx" / "prototype_output.geojson"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_forecast():
    """Create a sample forecast with metadata and GeoJSON data"""
    
    # Generate forecast ID
    forecast_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create metadata
    metadata = {
        "id": forecast_id,
        "created_at": datetime.now().isoformat(),
        "model": "physx_prototype",
        "region": "north_forest_test",
        "timesteps": 20,
        "duration_seconds": 95.0,
        "description": "Fire spread simulation using physics-based model"
    }
    
    # Save metadata
    metadata_path = DATA_DIR / f"{forecast_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy sample GeoJSON
    if SAMPLE_GEOJSON.exists():
        geojson_path = DATA_DIR / f"{forecast_id}.geojson"
        shutil.copy(SAMPLE_GEOJSON, geojson_path)
        print(f"✓ Created forecast: {forecast_id}")
        print(f"  Metadata: {metadata_path}")
        print(f"  GeoJSON: {geojson_path}")
    else:
        print(f"⚠ Sample GeoJSON not found at {SAMPLE_GEOJSON}")
        print(f"  Metadata created but GeoJSON is missing")
    
    return forecast_id


if __name__ == "__main__":
    print("Initializing sample forecast data...")
    forecast_id = create_sample_forecast()
    print(f"\nSample forecast created successfully!")
    print(f"Forecast ID: {forecast_id}")
    print(f"\nTest the API:")
    print(f"  curl -H 'X-API-Key: dev-key-change-in-production' http://localhost:8000/forecast/{forecast_id}")
    print(f"  curl -H 'X-API-Key: dev-key-change-in-production' http://localhost:8000/latest")
