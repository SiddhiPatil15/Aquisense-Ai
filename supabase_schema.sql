-- ============================================================================
-- AquiSense AI — Supabase PostgreSQL Database Schema
-- Version: 2.0.0
-- Target: Supabase SQL Editor / PostgreSQL 15+
-- ============================================================================

-- Enable UUID extension if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ----------------------------------------------------------------------------
-- 1. Table: monitoring_stations
-- Stores telemetric piezometers, observation borewells, and CGWB stations.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitoring_stations (
    station_id TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    state TEXT DEFAULT 'Maharashtra',
    district TEXT DEFAULT 'Pune',
    taluka TEXT,
    aquifer_type TEXT DEFAULT 'Deccan Basalt',
    well_depth_m NUMERIC(6, 2),
    status TEXT DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'MAINTENANCE', 'DECOMMISSIONED')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE monitoring_stations IS 'Geospatial registry of telemetric and manual groundwater monitoring stations';
COMMENT ON COLUMN monitoring_stations.station_id IS 'Unique station identifier (e.g. W-01, DWLR-PUN-04)';

-- ----------------------------------------------------------------------------
-- 2. Table: groundwater_readings
-- Time-series hydrological and hydrochemical measurements.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS groundwater_readings (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL REFERENCES monitoring_stations(station_id) ON DELETE CASCADE,
    measurement_date DATE NOT NULL,
    measurement_timestamp TIMESTAMPTZ,
    groundwater_level_m NUMERIC(6, 2) NOT NULL,
    rainfall_mm NUMERIC(6, 2) DEFAULT 0.0,
    temperature_c NUMERIC(5, 2),
    soil_moisture_pct NUMERIC(5, 2),
    ph NUMERIC(4, 2),
    tds_mg_l NUMERIC(7, 2),
    fluoride_mg_l NUMERIC(5, 2),
    nitrate_mg_l NUMERIC(6, 2),
    chloride_mg_l NUMERIC(6, 2),
    data_source TEXT DEFAULT 'manual_import',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Enforce idempotency: prevent duplicate measurement for the same station on the same date
    CONSTRAINT uq_station_measurement_date UNIQUE (station_id, measurement_date)
);

COMMENT ON TABLE groundwater_readings IS 'Time-series groundwater table depths, meteorological drivers, and water quality parameters';
COMMENT ON COLUMN groundwater_readings.groundwater_level_m IS 'Depth to groundwater in metres below reference ground level (bgl)';

-- Indexes for performant time-series querying
CREATE INDEX IF NOT EXISTS idx_gw_readings_station_date 
    ON groundwater_readings(station_id, measurement_date DESC);
CREATE INDEX IF NOT EXISTS idx_gw_readings_date 
    ON groundwater_readings(measurement_date DESC);
CREATE INDEX IF NOT EXISTS idx_gw_readings_source 
    ON groundwater_readings(data_source);

-- ----------------------------------------------------------------------------
-- 3. Table: groundwater_predictions
-- AI/ML model inference history (XGBoost, Random Forest, GBDT, etc.)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS groundwater_predictions (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL REFERENCES monitoring_stations(station_id) ON DELETE CASCADE,
    prediction_date TIMESTAMPTZ DEFAULT NOW(),
    model_name TEXT NOT NULL,
    forecast_horizon_days INT NOT NULL DEFAULT 30,
    target_variable TEXT NOT NULL DEFAULT 'groundwater_level',
    predicted_level_m NUMERIC(6, 2) NOT NULL,
    risk_category TEXT NOT NULL DEFAULT 'SAFE' CHECK (risk_category IN ('SAFE', 'SEMI-CRITICAL', 'CRITICAL', 'OVER-EXPLOITED')),
    model_metrics JSONB DEFAULT '{}'::jsonb,
    forecast_series JSONB DEFAULT '[]'::jsonb,
    feature_importance JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE groundwater_predictions IS 'Archival log of genuine AI model forecasts, performance metrics, and feature importance';

CREATE INDEX IF NOT EXISTS idx_gw_preds_station_date 
    ON groundwater_predictions(station_id, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_gw_preds_model 
    ON groundwater_predictions(model_name);

-- ----------------------------------------------------------------------------
-- 4. Table: groundwater_alerts
-- Early-warning hydrological and water quality alerts.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS groundwater_alerts (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL REFERENCES monitoring_stations(station_id) ON DELETE CASCADE,
    alert_timestamp TIMESTAMPTZ DEFAULT NOW(),
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    current_level_m NUMERIC(6, 2),
    threshold_m NUMERIC(6, 2),
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'ACKNOWLEDGED', 'RESOLVED')),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE groundwater_alerts IS 'Active and historical aquifer depletion and contamination warning signals';

CREATE INDEX IF NOT EXISTS idx_gw_alerts_station 
    ON groundwater_alerts(station_id, alert_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_gw_alerts_status 
    ON groundwater_alerts(status);

-- ----------------------------------------------------------------------------
-- 5. Row Level Security (RLS) Policies
-- Ensures secure access: public read-only, while inserts/updates require 
-- service-role (backend FastAPI microservice) or authenticated token.
-- ----------------------------------------------------------------------------
ALTER TABLE monitoring_stations ENABLE ROW LEVEL SECURITY;
ALTER TABLE groundwater_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE groundwater_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE groundwater_alerts ENABLE ROW LEVEL SECURITY;

-- Read policies for public/anon access
DROP POLICY IF EXISTS "Allow public read-only access on monitoring_stations" ON monitoring_stations;
CREATE POLICY "Allow public read-only access on monitoring_stations"
    ON monitoring_stations FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read-only access on groundwater_readings" ON groundwater_readings;
CREATE POLICY "Allow public read-only access on groundwater_readings"
    ON groundwater_readings FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read-only access on groundwater_predictions" ON groundwater_predictions;
CREATE POLICY "Allow public read-only access on groundwater_predictions"
    ON groundwater_predictions FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read-only access on groundwater_alerts" ON groundwater_alerts;
CREATE POLICY "Allow public read-only access on groundwater_alerts"
    ON groundwater_alerts FOR SELECT USING (true);

-- Note: The service_role key used by FastAPI automatically bypasses RLS in Supabase,
-- guaranteeing that only your verified backend microservice can write or update records.
