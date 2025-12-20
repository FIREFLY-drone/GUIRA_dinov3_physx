"""
Tests for GUIRA Core API endpoints

Run with: pytest test_api.py -v
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Import app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app, DATA_DIR, save_forecast_metadata

# Test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "dev-key-change-in-production"


@pytest.fixture(autouse=True)
def setup_test_data():
    """Setup test data before each test"""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a test forecast
    test_forecast_id = "test_forecast_001"
    test_metadata = {
        "id": test_forecast_id,
        "created_at": "2024-01-01T00:00:00",
        "model": "test_model",
        "region": "test_region",
        "timesteps": 10,
        "duration_seconds": 100.0
    }
    save_forecast_metadata(test_forecast_id, test_metadata)
    
    # Create test GeoJSON
    test_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"timestep": 0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                }
            }
        ]
    }
    geojson_path = DATA_DIR / f"{test_forecast_id}.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(test_geojson, f)
    
    yield
    
    # Cleanup
    metadata_path = DATA_DIR / f"{test_forecast_id}_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    if geojson_path.exists():
        geojson_path.unlink()


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "version" in data


def test_get_forecast_without_auth():
    """Test forecast endpoint without authentication"""
    response = client.get("/forecast/test_forecast_001")
    assert response.status_code == 401


def test_get_forecast_with_auth():
    """Test forecast endpoint with authentication"""
    response = client.get(
        "/forecast/test_forecast_001",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["forecast"]["id"] == "test_forecast_001"
    assert "geojson_url" in data


def test_get_forecast_not_found():
    """Test forecast endpoint with non-existent forecast"""
    response = client.get(
        "/forecast/nonexistent",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 404


def test_get_forecast_geojson():
    """Test forecast GeoJSON endpoint"""
    response = client.get(
        "/forecast/test_forecast_001/geojson",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


def test_get_latest_forecast():
    """Test latest forecast endpoint"""
    response = client.get(
        "/latest",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["forecast"] is not None
    assert "geojson_url" in data


def test_list_forecasts():
    """Test forecast listing endpoint"""
    response = client.get(
        "/forecasts?page=1&per_page=10",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "forecasts" in data
    assert data["page"] == 1
    assert data["per_page"] == 10


def test_list_forecasts_pagination():
    """Test forecast listing pagination"""
    response = client.get(
        "/forecasts?page=2&per_page=5",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 2
    assert data["per_page"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
