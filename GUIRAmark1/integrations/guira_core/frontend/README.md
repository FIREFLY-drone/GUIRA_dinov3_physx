# GUIRA Frontend

React + TypeScript frontend with MapLibre GL for visualizing fire forecast data.

## Features

- **Interactive Map**: MapLibre GL map displaying GeoJSON fire spread overlays
- **Timeline Control**: Slider to navigate through forecast timesteps
- **Color-Coded Visualization**: Fire intensity displayed with gradient colors
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Fetches latest forecast data from API

## Technology Stack

- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **MapLibre GL**: Open-source mapping library
- **OpenStreetMap**: Base map tiles

## Setup

### Install Dependencies

```bash
npm install
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_API_KEY=dev-key-change-in-production
```

## Development

### Start Dev Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Hot Reload

Vite provides instant hot module replacement. Edit any file and see changes immediately.

## Building

### Build for Production

```bash
npm run build
```

This will:
1. Compile TypeScript
2. Bundle and minify assets
3. Output to `../orchestrator/api/static/`

### Build Script

Use the convenience script:

```bash
./build_to_backend.sh
```

This script:
- Checks for dependencies
- Runs the build
- Outputs to the backend static folder
- Provides test instructions

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── MapView.tsx          # Map component with GeoJSON overlay
│   │   ├── MapView.css
│   │   ├── ForecastTimeline.tsx # Timeline slider control
│   │   └── ForecastTimeline.css
│   ├── types/                    # TypeScript type definitions
│   ├── App.tsx                   # Main application component
│   ├── App.css
│   ├── main.tsx                  # Application entry point
│   └── index.css                 # Global styles
├── index.html                    # HTML template
├── vite.config.ts                # Vite configuration
├── tsconfig.json                 # TypeScript configuration
├── package.json                  # Dependencies
├── build_to_backend.sh           # Build script
└── README.md
```

## Components

### MapView

The `MapView` component displays the fire forecast on an interactive map.

**Props:**
- `forecastUrl`: URL to fetch GeoJSON data
- `apiKey`: API key for authentication

**Features:**
- Loads GeoJSON from API
- Color-codes fire spread by timestep
- Clickable features show details in popup
- Auto-fits bounds to forecast extent
- Navigation controls (zoom, pan)
- Scale indicator

**Map Layers:**
- Base: OpenStreetMap raster tiles
- Forecast Fill: Semi-transparent polygons
- Forecast Line: Red outline of fire perimeter

### ForecastTimeline

The `ForecastTimeline` component provides timeline controls.

**Props:**
- `timesteps`: Total number of timesteps
- `duration`: Total simulation duration in seconds

**Features:**
- Slider to select timestep
- Time display (current/total)
- Step counter
- Color legend
- Play/pause button (placeholder)

## Styling

The app uses a clean, minimal design with:
- Blue/purple gradient header
- White content areas
- Grey neutral backgrounds
- Red/orange fire color scheme
- Responsive layout

Colors match the fire intensity gradient:
- Early: `#ffeda0` (yellow)
- Moderate: `#feb24c` (light orange)
- Active: `#fd8d3c` (orange)
- Intense: `#fc4e2a` (red-orange)
- Peak: `#e31a1c` (red)

## API Integration

### Authentication

All API calls include the `X-API-Key` header from environment variables.

### Endpoints Used

- `GET /latest`: Fetch most recent forecast
- `GET /forecast/{id}/geojson`: Fetch GeoJSON data

### Error Handling

The app handles:
- Network errors
- Authentication failures
- Missing data
- Invalid responses

Errors are displayed in the UI with appropriate messages.

## Testing the Build

After building, test with the backend:

```bash
# Build frontend
npm run build

# Start API server
cd ../orchestrator/api
python init_sample_data.py
python app.py

# Open browser
open http://localhost:8000
```

## Customization

### Changing the Base Map

Edit `MapView.tsx` and modify the map style:

```typescript
style: {
  version: 8,
  sources: {
    'custom': {
      type: 'raster',
      tiles: ['https://your-tile-server/{z}/{x}/{y}.png'],
      tileSize: 256
    }
  },
  layers: [...]
}
```

### Adjusting Colors

Modify the color gradient in `MapView.tsx`:

```typescript
'fill-color': [
  'interpolate',
  ['linear'],
  ['get', 'timestep'],
  0, '#yourcolor1',
  50, '#yourcolor2',
  100, '#yourcolor3'
]
```

## Security Considerations

### API Key

The API key is stored in environment variables and not committed to version control.

For production:
1. Use a strong, unique API key
2. Rotate keys regularly
3. Consider OAuth2 or JWT tokens
4. Enable HTTPS only

### GeoJSON Resolution

The API should provide appropriately generalized GeoJSON to avoid revealing sensitive locations. High-resolution data should be restricted to authenticated users with proper authorization.

## Performance

### Build Size

Typical production build:
- JavaScript: ~300 KB (gzipped)
- CSS: ~10 KB (gzipped)
- Map library: ~200 KB (gzipped)
- Total: ~510 KB

### Loading Performance

- Initial load: < 2s on fast connection
- Map initialization: < 1s
- GeoJSON loading: depends on data size

### Optimization

The Vite build automatically:
- Minifies JavaScript and CSS
- Tree-shakes unused code
- Optimizes assets
- Generates source maps

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Troubleshooting

### Build Errors

**Issue:** `npm run build` fails with TypeScript errors

**Solution:** Check TypeScript configuration and fix type errors

### Map Not Loading

**Issue:** Map appears grey or doesn't load

**Solution:** 
- Check console for errors
- Verify internet connection (OpenStreetMap tiles)
- Check browser WebGL support

### API Connection Failed

**Issue:** "Failed to load forecast" error

**Solution:**
- Verify API is running
- Check API URL in `.env`
- Verify API key is correct
- Check browser console for CORS errors

## Model Metadata Block

**MODEL:** React + Vite frontend with MapLibre GL

**DATA:**
- Input: GeoJSON from API endpoints
- Sample: `samples/physx/prototype_output.geojson`
- Format: GeoJSON FeatureCollection with Polygon features

**BUILD RECIPE:**
```bash
npm install
npm run build
# Output: ../orchestrator/api/static/
```

**EVAL & ACCEPTANCE:**
- Build completes without errors
- All TypeScript type checks pass
- Map loads and displays forecast
- Timeline controls function
- Responsive on mobile devices
- Build output < 1 MB total size
- Initial load < 3s on 3G connection
