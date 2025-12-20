#!/usr/bin/env python3
"""
Integration test for GUIRA Core API + Frontend

Tests the complete workflow:
1. API serves forecast data
2. Frontend assets are available
3. GeoJSON can be fetched with authentication
4. All endpoints respond correctly
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Import app
sys.path.insert(0, str(Path(__file__).parent))
from app import app, DATA_DIR, STATIC_DIR

# Test client
client = TestClient(app)
API_KEY = "dev-key-change-in-production"


def test_static_files():
    """Test that frontend static files exist"""
    print("Testing static files...")
    
    index_path = STATIC_DIR / "index.html"
    assert index_path.exists(), "index.html not found in static directory"
    
    assets_dir = STATIC_DIR / "assets"
    assert assets_dir.exists(), "assets directory not found"
    
    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    
    assert len(js_files) > 0, "No JavaScript files found"
    assert len(css_files) > 0, "No CSS files found"
    
    print(f"  ✓ Found {len(js_files)} JS files and {len(css_files)} CSS files")


def test_api_health():
    """Test API health check"""
    print("Testing API health...")
    
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    
    print(f"  ✓ API health: {data['status']}")


def test_forecast_data():
    """Test forecast data availability"""
    print("Testing forecast data...")
    
    # List forecasts
    response = client.get("/forecasts", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["forecasts"]) > 0, "No forecasts available"
    
    forecast_id = data["forecasts"][0]["id"]
    print(f"  ✓ Found {len(data['forecasts'])} forecast(s)")
    print(f"    Latest: {forecast_id}")
    
    # Get specific forecast
    response = client.get(f"/forecast/{forecast_id}", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    
    forecast_data = response.json()
    assert forecast_data["status"] == "ok"
    print(f"    Model: {forecast_data['forecast']['model']}")
    print(f"    Region: {forecast_data['forecast']['region']}")
    print(f"    Timesteps: {forecast_data['forecast']['timesteps']}")
    
    # Get GeoJSON
    geojson_url = forecast_data["geojson_url"]
    response = client.get(geojson_url, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0
    
    print(f"  ✓ GeoJSON has {len(geojson['features'])} features")


def test_authentication():
    """Test API authentication"""
    print("Testing authentication...")
    
    # Without API key
    response = client.get("/latest")
    assert response.status_code == 401
    print("  ✓ Unauthorized access blocked")
    
    # With API key
    response = client.get("/latest", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    print("  ✓ Authenticated access allowed")


def test_frontend_serving():
    """Test that frontend index.html is served at root"""
    print("Testing frontend serving...")
    
    response = client.get("/")
    assert response.status_code == 200
    
    content = response.text
    assert "GUIRA" in content or "root" in content
    print("  ✓ Frontend served at root URL")


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("GUIRA Core Integration Test")
    print("=" * 60)
    print()
    
    tests = [
        test_static_files,
        test_api_health,
        test_forecast_data,
        test_authentication,
        test_frontend_serving,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✅ All integration tests passed!")
    print("\nTo test manually:")
    print("  1. Start API: python app.py")
    print("  2. Open browser: http://localhost:8000")
    print("  3. Map should load with forecast overlay")


if __name__ == "__main__":
    main()
