# GUIRA Core API

FastAPI backend providing fire forecast and detection data endpoints.

## Features

- **Forecast Endpoints**: Access fire spread predictions and GeoJSON overlays
- **Authentication**: API key-based security
- **Static File Serving**: Serves built frontend from `/static`
- **Paginated Listing**: Browse all available forecasts
- **Health Monitoring**: Health check endpoint for monitoring

## Setup

### Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
GUIRA_API_KEY=your-secure-api-key-here
GUIRA_DATA_DIR=./data/forecasts
PORT=8000
```

### Initialize Sample Data

```bash
python init_sample_data.py
```

This will:
- Create sample forecast metadata
- Copy sample GeoJSON from the samples directory
- Set up the data directory structure

## Running the API

### Development Mode

```bash
python app.py
```

The API will start on `http://localhost:8000`

### Production Mode

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns API health status. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0"
}
```

### Get Forecast by ID

```bash
GET /forecast/{forecast_id}
Headers: X-API-Key: your-api-key
```

Returns forecast metadata and GeoJSON URL.

**Response:**
```json
{
  "status": "ok",
  "forecast": {
    "id": "forecast_20240101_120000",
    "created_at": "2024-01-01T12:00:00",
    "model": "physx_prototype",
    "region": "north_forest_test",
    "timesteps": 20,
    "duration_seconds": 95.0
  },
  "geojson_url": "/forecast/forecast_20240101_120000/geojson"
}
```

### Get Forecast GeoJSON

```bash
GET /forecast/{forecast_id}/geojson
Headers: X-API-Key: your-api-key
```

Returns raw GeoJSON data for map visualization.

### Get Latest Forecast

```bash
GET /latest
Headers: X-API-Key: your-api-key
```

Returns the most recent forecast.

### List Forecasts

```bash
GET /forecasts?page=1&per_page=10
Headers: X-API-Key: your-api-key
```

Returns paginated list of all forecasts.

**Response:**
```json
{
  "status": "ok",
  "forecasts": [...],
  "total": 25,
  "page": 1,
  "per_page": 10
}
```

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=. --cov-report=html
```

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Get forecast (replace with actual forecast ID)
curl -H "X-API-Key: dev-key-change-in-production" \
  http://localhost:8000/forecast/forecast_20240101_120000

# Get latest forecast
curl -H "X-API-Key: dev-key-change-in-production" \
  http://localhost:8000/latest

# List all forecasts
curl -H "X-API-Key: dev-key-change-in-production" \
  http://localhost:8000/forecasts?page=1&per_page=10
```

## Security

### API Key Authentication

All forecast endpoints require authentication via the `X-API-Key` header.

**Development Key:** `dev-key-change-in-production`

**Production:** Set a secure API key via the `GUIRA_API_KEY` environment variable.

### GeoJSON Resolution

The API serves GeoJSON data at a configured resolution. For public-facing deployments, ensure GeoJSON data is appropriately generalized to avoid revealing sensitive asset locations.

## Data Directory Structure

```
data/forecasts/
├── forecast_20240101_120000_metadata.json
├── forecast_20240101_120000.geojson
├── forecast_20240101_130000_metadata.json
└── forecast_20240101_130000.geojson
```

## Integration with Frontend

The API serves the built frontend from the `/static` directory. Build the frontend with:

```bash
cd ../../../frontend
./build_to_backend.sh
```

Then access the dashboard at `http://localhost:8000/`

## Monitoring

The API logs all requests and errors. Monitor the logs for:
- Authentication failures
- Missing forecast data
- GeoJSON loading errors
- Performance issues

## Architecture

```
orchestrator/api/
├── app.py                  # FastAPI application
├── init_sample_data.py     # Sample data initialization
├── static/                 # Built frontend assets
├── data/                   # Forecast data storage
│   └── forecasts/          # Forecast metadata and GeoJSON
└── tests/                  # API tests
    └── test_api.py
```

## Model Metadata Block

**MODEL:** FastAPI REST API serving fire forecast data

**DATA:** 
- Forecast metadata (JSON): id, created_at, model, region, timesteps, duration
- GeoJSON forecast data: fire spread perimeter evolution over time
- Sample data: `samples/physx/prototype_output.geojson`

**BUILD RECIPE:**
```bash
# No training required - this is an API service
# Deploy with:
uvicorn app:app --host 0.0.0.0 --port 8000
```

**EVAL & ACCEPTANCE:**
- All API tests pass: `pytest tests/test_api.py -v`
- Health check returns 200 OK
- Authenticated requests return forecast data
- GeoJSON loads correctly in frontend map
- Response time: < 500ms for forecast metadata, < 2s for GeoJSON
