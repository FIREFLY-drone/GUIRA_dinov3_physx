# GUIRA Core Orchestrator

Complete implementation of API backend and frontend dashboard for fire forecast visualization.

## Overview

This implementation provides:

1. **FastAPI Backend** (`api/`): REST API serving fire forecast data with authentication
2. **React Frontend** (`../frontend/`): Interactive map dashboard with MapLibre GL
3. **Integration**: Frontend builds directly to backend static folder

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- FastAPI dependencies: `pip install fastapi uvicorn pydantic`

### Setup and Run

```bash
# 1. Initialize sample data
cd api
python init_sample_data.py

# 2. Build frontend (one-time or after changes)
cd ../frontend
npm install
./build_to_backend.sh

# 3. Start API server
cd ../orchestrator/api
python app.py

# 4. Open browser
# Visit: http://localhost:8000
```

The dashboard will display:
- Interactive map with fire spread forecast overlay
- Timeline slider to navigate through timesteps
- Color-coded visualization of fire intensity

## Architecture

```
orchestrator/
├── api/                          # FastAPI backend
│   ├── app.py                   # Main API application
│   ├── init_sample_data.py      # Sample data creation
│   ├── test_integration.py      # Integration tests
│   ├── static/                  # Built frontend (auto-generated)
│   ├── data/                    # Forecast data storage
│   │   └── forecasts/           # GeoJSON + metadata
│   └── tests/                   # Unit tests
│       └── test_api.py
│
└── frontend/ (../frontend/)     # React + TypeScript frontend
    ├── src/
    │   ├── components/
    │   │   ├── MapView.tsx      # Map with GeoJSON overlay
    │   │   └── ForecastTimeline.tsx  # Timeline control
    │   ├── App.tsx              # Main application
    │   └── main.tsx             # Entry point
    ├── vite.config.ts           # Build config (outDir: ../orchestrator/api/static)
    └── build_to_backend.sh      # Build script
```

## API Endpoints

All forecast endpoints require authentication via `X-API-Key` header.

### Public Endpoints

- `GET /` - Serve frontend dashboard
- `GET /health` - Health check

### Authenticated Endpoints

- `GET /forecast/{id}` - Get forecast by ID
- `GET /forecast/{id}/geojson` - Get GeoJSON data
- `GET /latest` - Get latest forecast
- `GET /forecasts?page=1&per_page=10` - List all forecasts (paginated)

### Example

```bash
# Get latest forecast
curl -H "X-API-Key: dev-key-change-in-production" \
  http://localhost:8000/latest

# Get GeoJSON
curl -H "X-API-Key: dev-key-change-in-production" \
  http://localhost:8000/forecast/forecast_20240101_120000/geojson
```

## Frontend Features

### MapView Component

- **Base Map**: OpenStreetMap tiles via MapLibre GL
- **Forecast Overlay**: GeoJSON polygons showing fire spread
- **Color Gradient**: From yellow (early) to red (peak intensity)
- **Interactive**: Click features to see details (timestep, time, cells, perimeter)
- **Controls**: Zoom, pan, scale indicator
- **Auto-fit**: Automatically centers on forecast extent

### ForecastTimeline Component

- **Slider**: Navigate through timesteps
- **Time Display**: Current time vs total duration
- **Step Counter**: Current step vs total steps
- **Legend**: Color-coded intensity levels
- **Placeholder**: Play/pause button (for future implementation)

## Development

### Backend Development

```bash
cd api

# Run tests
pytest tests/test_api.py -v

# Run integration tests
python test_integration.py

# Start dev server with auto-reload
uvicorn app:app --reload
```

### Frontend Development

```bash
cd ../frontend

# Install dependencies
npm install

# Start dev server (with hot reload)
npm run dev
# Visit: http://localhost:5173

# Build for production
npm run build
# Output: ../orchestrator/api/static/
```

### Making Changes

1. Edit frontend code in `frontend/src/`
2. Test with dev server: `npm run dev`
3. Build: `npm run build`
4. Restart API server
5. Test: http://localhost:8000

## Configuration

### API Configuration

Set via environment variables:

```bash
export GUIRA_API_KEY=your-secure-key
export GUIRA_DATA_DIR=./data/forecasts
export PORT=8000
```

### Frontend Configuration

Copy `.env.example` to `.env` in `frontend/`:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_API_KEY=dev-key-change-in-production
```

## Security

### Authentication

- **Method**: API key via `X-API-Key` header
- **Default Key**: `dev-key-change-in-production` (change in production!)
- **Scope**: All forecast endpoints require authentication

### Production Considerations

1. Use a strong, unique API key
2. Enable HTTPS only (no HTTP)
3. Implement rate limiting
4. Consider OAuth2 or JWT tokens
5. Restrict CORS origins
6. Serve GeoJSON at appropriate resolution

### GeoJSON Resolution

The API should provide generalized GeoJSON to avoid revealing sensitive locations. High-resolution data should require additional authorization.

## Data Format

### Forecast Metadata

```json
{
  "id": "forecast_20240101_120000",
  "created_at": "2024-01-01T12:00:00",
  "model": "physx_prototype",
  "region": "north_forest_test",
  "timesteps": 20,
  "duration_seconds": 95.0,
  "description": "Fire spread simulation"
}
```

### GeoJSON Structure

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "simulation_type": "physx_prototype",
    "total_timesteps": 20
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestep": 0,
        "time_seconds": 0.0,
        "num_cells": 7,
        "perimeter_length_m": 16.0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      }
    }
  ]
}
```

## Testing

### Unit Tests

```bash
cd api
pytest tests/test_api.py -v --cov=.
```

### Integration Tests

```bash
cd api
python test_integration.py
```

Tests verify:
- API health check
- Authentication (with/without key)
- Forecast data availability
- GeoJSON loading
- Frontend static file serving

## Deployment

### Production Deployment

```bash
# Build frontend
cd frontend
npm run build

# Start API with multiple workers
cd ../orchestrator/api
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment (Future)

```dockerfile
# Example Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY api/ ./api/
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### "Module not found" errors

Install dependencies:
```bash
pip install fastapi uvicorn pydantic pytest httpx
```

### Frontend build errors

Check Node version (18+) and run:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Map not loading

1. Check browser console for errors
2. Verify API is running: `curl http://localhost:8000/health`
3. Check API key in browser network tab
4. Verify internet connection (for OSM tiles)

### API key errors

Ensure `X-API-Key` header matches `GUIRA_API_KEY` environment variable (default: `dev-key-change-in-production`)

## Performance

### Backend

- Forecast metadata: < 100ms response time
- GeoJSON loading: < 2s for typical forecasts
- Static files: Served efficiently by FastAPI

### Frontend

- Bundle size: ~510 KB gzipped
- Initial load: < 2s on fast connection
- Map rendering: < 1s after GeoJSON load

### Optimization

- Frontend uses code splitting via Vite
- GeoJSON can be pre-simplified for lower resolutions
- Consider CDN for static assets in production

## Model Metadata

**MODEL:** FastAPI + React dashboard for fire forecast visualization

**DATA:**
- Input: Fire forecast GeoJSON (FeatureCollection with Polygons)
- Sample: `integrations/guira_core/samples/physx/prototype_output.geojson`
- Format: Timestep-based fire perimeter evolution

**BUILD RECIPE:**
```bash
# Backend: No build required
pip install fastapi uvicorn pydantic

# Frontend:
cd frontend
npm install
npm run build  # Output: ../orchestrator/api/static/
```

**EVAL & ACCEPTANCE:**
- ✅ All API unit tests pass (8/8)
- ✅ All integration tests pass (5/5)
- ✅ Frontend builds without errors
- ✅ Map displays GeoJSON overlay correctly
- ✅ Timeline controls function
- ✅ Authentication required for forecast endpoints
- ✅ Response time: < 500ms for metadata, < 2s for GeoJSON
- ✅ Bundle size: < 1 MB total

## License

Part of the GUIRA (Fire Prevention) system. See main project LICENSE.
