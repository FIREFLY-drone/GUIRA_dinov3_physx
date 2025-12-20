-- init_postgis.sql
-- PostGIS schema for GUIRA fire prevention system
-- Stores detections, forecasts, and geospatial data

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Detections table: stores all detection events from various models
CREATE TABLE IF NOT EXISTS detections (
  id TEXT PRIMARY KEY,
  ts TIMESTAMP NOT NULL,
  source TEXT NOT NULL,
  class TEXT NOT NULL,
  confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  geom GEOMETRY(POINT, 4326),
  embedding_uri TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial index for efficient geospatial queries
CREATE INDEX IF NOT EXISTS idx_detections_geom ON detections USING GIST(geom);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);

-- Forecasts table: stores fire spread predictions and simulation results
CREATE TABLE IF NOT EXISTS forecasts (
  request_id TEXT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW(),
  results_uri TEXT NOT NULL,
  meta JSONB DEFAULT '{}',
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for forecast queries
CREATE INDEX IF NOT EXISTS idx_forecasts_status ON forecasts(status);
CREATE INDEX IF NOT EXISTS idx_forecasts_created_at ON forecasts(created_at);

-- Sessions table: tracks live ingestion and analysis sessions
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  source TEXT NOT NULL,
  source_platform TEXT,
  start_ts TIMESTAMP DEFAULT NOW(),
  end_ts TIMESTAMP,
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_start_ts ON sessions(start_ts);

-- Embeddings table: stores vector embeddings for RAG
CREATE TABLE IF NOT EXISTS embeddings (
  id TEXT PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
  detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
  embedding_vector FLOAT[] NOT NULL,
  text_content TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_session_id ON embeddings(session_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_detection_id ON embeddings(detection_id);

-- Function to update timestamp on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_forecasts_updated_at BEFORE UPDATE ON forecasts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for production)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO guira_app;
