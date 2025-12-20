import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './MapView.css';

interface MapViewProps {
  forecastUrl: string;
  apiKey: string;
}

export default function MapView({ forecastUrl, apiKey }: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize map
    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            'osm': {
              type: 'raster',
              tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
              tileSize: 256,
              attribution: '© OpenStreetMap contributors'
            }
          },
          layers: [
            {
              id: 'osm',
              type: 'raster',
              source: 'osm',
              minzoom: 0,
              maxzoom: 19
            }
          ]
        },
        center: [0, 0],
        zoom: 5
      });

      map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
      map.current.addControl(new maplibregl.ScaleControl({}), 'bottom-left');

      map.current.on('load', () => {
        setMapLoaded(true);
      });

      map.current.on('error', (e) => {
        console.error('Map error:', e);
        setError('Map initialization error');
      });
    } catch (err) {
      console.error('Error initializing map:', err);
      setError('Failed to initialize map');
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!map.current || !mapLoaded || !forecastUrl) return;

    const loadForecast = async () => {
      try {
        setError('');
        
        const response = await fetch(forecastUrl, {
          headers: {
            'X-API-Key': apiKey,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to load forecast: ${response.status}`);
        }

        const geojson = await response.json();

        // Remove existing forecast layers and sources
        if (map.current!.getLayer('forecast-fill')) {
          map.current!.removeLayer('forecast-fill');
        }
        if (map.current!.getLayer('forecast-line')) {
          map.current!.removeLayer('forecast-line');
        }
        if (map.current!.getSource('forecast')) {
          map.current!.removeSource('forecast');
        }

        // Add forecast source
        map.current!.addSource('forecast', {
          type: 'geojson',
          data: geojson
        });

        // Add fill layer with color based on timestep
        map.current!.addLayer({
          id: 'forecast-fill',
          type: 'fill',
          source: 'forecast',
          paint: {
            'fill-color': [
              'interpolate',
              ['linear'],
              ['get', 'timestep'],
              0, '#ffeda0',
              25, '#feb24c',
              50, '#fd8d3c',
              75, '#fc4e2a',
              100, '#e31a1c'
            ],
            'fill-opacity': 0.4
          }
        });

        // Add outline layer
        map.current!.addLayer({
          id: 'forecast-line',
          type: 'line',
          source: 'forecast',
          paint: {
            'line-color': '#e31a1c',
            'line-width': 2,
            'line-opacity': 0.8
          }
        });

        // Fit bounds to forecast data
        if (geojson.features && geojson.features.length > 0) {
          const bounds = new maplibregl.LngLatBounds();
          
          geojson.features.forEach((feature: any) => {
            if (feature.geometry.type === 'Polygon') {
              feature.geometry.coordinates[0].forEach((coord: [number, number]) => {
                bounds.extend(coord);
              });
            }
          });

          map.current!.fitBounds(bounds, {
            padding: 50,
            duration: 1000
          });
        }

        // Add popup on click
        map.current!.on('click', 'forecast-fill', (e) => {
          if (!e.features || e.features.length === 0) return;
          
          const feature = e.features[0];
          const props = feature.properties;
          
          new maplibregl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`
              <div style="padding: 8px;">
                <strong>Timestep:</strong> ${props?.timestep || 'N/A'}<br/>
                <strong>Time:</strong> ${props?.time_seconds || 0}s<br/>
                <strong>Cells:</strong> ${props?.num_cells || 'N/A'}<br/>
                <strong>Perimeter:</strong> ${props?.perimeter_length_m || 'N/A'}m
              </div>
            `)
            .addTo(map.current!);
        });

        // Change cursor on hover
        map.current!.on('mouseenter', 'forecast-fill', () => {
          map.current!.getCanvas().style.cursor = 'pointer';
        });

        map.current!.on('mouseleave', 'forecast-fill', () => {
          map.current!.getCanvas().style.cursor = '';
        });

      } catch (err) {
        console.error('Error loading forecast:', err);
        setError(err instanceof Error ? err.message : 'Failed to load forecast data');
      }
    };

    loadForecast();
  }, [mapLoaded, forecastUrl, apiKey]);

  return (
    <div className="map-view">
      <div ref={mapContainer} className="map" />
      {error && <div className="map-error">{error}</div>}
      {!mapLoaded && <div className="map-loading">Loading map...</div>}
    </div>
  );
}
