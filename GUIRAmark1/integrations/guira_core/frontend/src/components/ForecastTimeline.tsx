import { useState } from 'react';
import './ForecastTimeline.css';

interface ForecastTimelineProps {
  timesteps: number;
  duration: number;
}

export default function ForecastTimeline({ timesteps, duration }: ForecastTimelineProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentStep(parseInt(e.target.value));
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const currentTime = (currentStep / timesteps) * duration;

  return (
    <div className="forecast-timeline">
      <div className="timeline-header">
        <h3>Fire Spread Timeline</h3>
        <div className="timeline-info">
          <span className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
          <span className="step-display">
            Step {currentStep} / {timesteps}
          </span>
        </div>
      </div>

      <div className="timeline-controls">
        <button
          className="play-button"
          onClick={() => setPlaying(!playing)}
          disabled={true}
          title="Playback not implemented in this version"
        >
          {playing ? '⏸' : '▶'}
        </button>

        <div className="slider-container">
          <input
            type="range"
            min="0"
            max={timesteps}
            value={currentStep}
            onChange={handleSliderChange}
            className="timeline-slider"
          />
          <div className="slider-ticks">
            {Array.from({ length: 5 }, (_, i) => {
              const step = Math.floor((timesteps / 4) * i);
              const time = (step / timesteps) * duration;
              return (
                <span key={i} className="tick-label">
                  {formatTime(time)}
                </span>
              );
            })}
          </div>
        </div>
      </div>

      <div className="timeline-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#ffeda0' }}></span>
          <span>Early spread</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#feb24c' }}></span>
          <span>Moderate</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#fd8d3c' }}></span>
          <span>Active</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#fc4e2a' }}></span>
          <span>Intense</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#e31a1c' }}></span>
          <span>Peak</span>
        </div>
      </div>
    </div>
  );
}
