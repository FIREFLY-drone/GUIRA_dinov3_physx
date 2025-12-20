"""
GUIRA Core Orchestrator API

FastAPI application providing endpoints for fire forecasts, alerts, and detection data.

Endpoints:
- GET /forecast/{id} - Get forecast by ID
- GET /latest - Get latest forecast
- GET /forecasts - List all forecasts (paginated)
- GET /health - Health check

Security:
- API key authentication via X-API-Key header
- Only authorized GeoJSON resolution for public views
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.environ.get("GUIRA_API_KEY", "dev-key-change-in-production")
DATA_DIR = Path(os.environ.get("GUIRA_DATA_DIR", "./data/forecasts"))
STATIC_DIR = Path(__file__).parent / "static"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="GUIRA Core API",
    description="Fire Prevention and Detection API",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Models
class ForecastMetadata(BaseModel):
    """Forecast metadata"""
    id: str = Field(..., description="Forecast ID")
    created_at: str = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    region: str = Field(..., description="Geographic region")
    timesteps: int = Field(..., description="Number of timesteps")
    duration_seconds: float = Field(..., description="Simulation duration")


class ForecastResponse(BaseModel):
    """Forecast response"""
    status: str = Field(..., description="Response status")
    forecast: ForecastMetadata
    geojson_url: str = Field(..., description="GeoJSON data URL")


class LatestForecastResponse(BaseModel):
    """Latest forecast response"""
    status: str
    forecast: Optional[ForecastMetadata] = None
    geojson_url: Optional[str] = None


class ForecastListResponse(BaseModel):
    """Forecast list response"""
    status: str
    forecasts: List[ForecastMetadata]
    total: int
    page: int
    per_page: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str


# Authentication dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# Utility functions
def load_forecast_metadata(forecast_id: str) -> Optional[Dict[str, Any]]:
    """Load forecast metadata from disk"""
    metadata_path = DATA_DIR / f"{forecast_id}_metadata.json"
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading forecast metadata {forecast_id}: {e}")
        return None


def save_forecast_metadata(forecast_id: str, metadata: Dict[str, Any]) -> bool:
    """Save forecast metadata to disk"""
    metadata_path = DATA_DIR / f"{forecast_id}_metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving forecast metadata {forecast_id}: {e}")
        return False


def list_all_forecasts() -> List[Dict[str, Any]]:
    """List all available forecasts"""
    forecasts = []
    for metadata_file in DATA_DIR.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                forecasts.append(metadata)
        except Exception as e:
            logger.error(f"Error loading {metadata_file}: {e}")
    
    # Sort by created_at descending
    forecasts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return forecasts


# API Routes
@app.get("/", response_class=FileResponse)
async def root():
    """Serve frontend index.html"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        content={
            "message": "GUIRA Core API",
            "docs": "/docs",
            "health": "/health"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/forecast/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get forecast by ID
    
    Returns forecast metadata and URL to GeoJSON data.
    Requires authentication via X-API-Key header.
    """
    logger.info(f"Fetching forecast: {forecast_id}")
    
    metadata = load_forecast_metadata(forecast_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Forecast {forecast_id} not found")
    
    # Check if GeoJSON file exists
    geojson_path = DATA_DIR / f"{forecast_id}.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast data not found for {forecast_id}")
    
    return ForecastResponse(
        status="ok",
        forecast=ForecastMetadata(**metadata),
        geojson_url=f"/forecast/{forecast_id}/geojson"
    )


@app.get("/forecast/{forecast_id}/geojson")
async def get_forecast_geojson(
    forecast_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get forecast GeoJSON data
    
    Returns raw GeoJSON for map visualization.
    Requires authentication via X-API-Key header.
    """
    geojson_path = DATA_DIR / f"{forecast_id}.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast data not found for {forecast_id}")
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        return JSONResponse(content=geojson_data)
    except Exception as e:
        logger.error(f"Error loading GeoJSON for {forecast_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading forecast data")


@app.get("/latest", response_model=LatestForecastResponse)
async def get_latest_forecast(authenticated: bool = Depends(verify_api_key)):
    """
    Get latest forecast
    
    Returns the most recent forecast metadata and GeoJSON URL.
    Requires authentication via X-API-Key header.
    """
    logger.info("Fetching latest forecast")
    
    forecasts = list_all_forecasts()
    if not forecasts:
        return LatestForecastResponse(
            status="ok",
            forecast=None,
            geojson_url=None
        )
    
    latest = forecasts[0]
    return LatestForecastResponse(
        status="ok",
        forecast=ForecastMetadata(**latest),
        geojson_url=f"/forecast/{latest['id']}/geojson"
    )


@app.get("/forecasts", response_model=ForecastListResponse)
async def list_forecasts(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    authenticated: bool = Depends(verify_api_key)
):
    """
    List all forecasts (paginated)
    
    Returns paginated list of forecast metadata.
    Requires authentication via X-API-Key header.
    """
    logger.info(f"Listing forecasts: page={page}, per_page={per_page}")
    
    all_forecasts = list_all_forecasts()
    total = len(all_forecasts)
    
    # Pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_forecasts = all_forecasts[start_idx:end_idx]
    
    return ForecastListResponse(
        status="ok",
        forecasts=[ForecastMetadata(**f) for f in page_forecasts],
        total=total,
        page=page,
        per_page=per_page
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting GUIRA Core API on port {port}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Static directory: {STATIC_DIR}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
