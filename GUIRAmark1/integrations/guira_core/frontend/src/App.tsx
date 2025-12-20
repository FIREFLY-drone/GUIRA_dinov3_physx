import { useState, useEffect } from 'react'
import MapView from './components/MapView'
import ForecastTimeline from './components/ForecastTimeline'
import './App.css'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_KEY = import.meta.env.VITE_API_KEY || 'dev-key-change-in-production';

interface ForecastMetadata {
  id: string;
  created_at: string;
  model: string;
  region: string;
  timesteps: number;
  duration_seconds: number;
}

function App() {
  const [forecast, setForecast] = useState<ForecastMetadata | null>(null);
  const [forecastUrl, setForecastUrl] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchLatestForecast();
  }, []);

  const fetchLatestForecast = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/latest`, {
        headers: {
          'X-API-Key': API_KEY,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.forecast) {
        setForecast(data.forecast);
        setForecastUrl(`${API_BASE_URL}${data.geojson_url}`);
      } else {
        setError('No forecasts available');
      }
    } catch (err) {
      console.error('Error fetching forecast:', err);
      setError(err instanceof Error ? err.message : 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔥 GUIRA - Fire Prevention Dashboard</h1>
        <div className="header-info">
          {loading && <span>Loading...</span>}
          {error && <span className="error">{error}</span>}
          {forecast && (
            <span>
              {forecast.model} | {forecast.region} | {forecast.timesteps} steps
            </span>
          )}
        </div>
      </header>

      <main className="main-content">
        <div className="map-container">
          {forecastUrl ? (
            <MapView forecastUrl={forecastUrl} apiKey={API_KEY} />
          ) : (
            <div className="placeholder">
              {loading ? 'Loading map...' : 'No forecast data available'}
            </div>
          )}
        </div>

        {forecast && (
          <div className="timeline-container">
            <ForecastTimeline
              timesteps={forecast.timesteps}
              duration={forecast.duration_seconds}
            />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
