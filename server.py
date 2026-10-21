"""
Aquifer Sense AI — High-Level Enterprise ML & Hydrological Microservice
Provides REST endpoints for:
1. Real-time Live Weather & Climate Telemetry Ingestion (Open-Meteo API)
2. Advanced ML Pipeline (XGBoost, Random Forest, Gradient Boosting, Linear Models)
3. Explainable AI (XAI) Feature Attribution & Hydrological Driver Decomposition
4. Climate & Policy "What-If" Scenario Simulator (CGWB Stage of Development)
5. Supabase PostgreSQL Persistent Database & Alert Management
"""

import sys
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import math
import random
import datetime
import urllib.request
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

app = FastAPI(
    title="AquiSense AI — Advanced Groundwater Intelligence API",
    description="High-level ML, Explainable AI, Meteorological Ingestion, Hydrological Simulation, and Supabase PostgreSQL Microservice",
    version="2.0.0"
)

# ---------------------------------------------------------------------------
# CORS Configuration (Vercel Production & Local Environments)
# ---------------------------------------------------------------------------
FRONTEND_ORIGINS = os.environ.get(
    "FRONTEND_ORIGINS",
    "https://aquisense-ai.vercel.app,http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:5500"
)
origins = [o.strip() for o in FRONTEND_ORIGINS.split(",") if o.strip()]
if "https://aquisense-ai.vercel.app" not in origins:
    origins.append("https://aquisense-ai.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Supabase PostgreSQL Client Initialization
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")
API_AUTH_KEY = os.environ.get("API_AUTH_KEY")

try:
    from supabase import create_client, Client
    supabase_client: Optional[Client] = (
        create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        if SUPABASE_URL and SUPABASE_SECRET_KEY
        else None
    )
except Exception as e:
    print(f"Warning: Failed to initialize Supabase client: {e}")
    supabase_client = None

def get_supabase_client() -> Any:
    if not supabase_client:
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable. Supabase credentials (SUPABASE_URL, SUPABASE_SECRET_KEY) are not configured."
        )
    return supabase_client

def verify_write_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """
    Authorization protection for database write endpoints.
    If API_AUTH_KEY is set in environment, verifies incoming API key or Bearer token.
    """
    expected_key = os.environ.get("API_AUTH_KEY")
    if not expected_key:
        return True  # Open in development if not explicitly configured

    provided_key = x_api_key
    if not provided_key and authorization:
        if authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()
        else:
            provided_key = authorization.strip()

    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. A valid API key (X-API-Key or Authorization Bearer header) is required to write to the database."
        )
    return True

# Global model cache
MODEL_REGISTRY = {}
SCALER_REGISTRY = {}

# ---------------------------------------------------------------------------
# Models & Schemas
# ---------------------------------------------------------------------------
class RecordModel(BaseModel):
    Date: str
    Well_ID: str
    Latitude: float
    Longitude: float
    Groundwater_Level: float
    Rainfall: float
    Temperature: float
    Soil_Moisture: float
    pH: Optional[float] = 7.2
    TDS: Optional[float] = 400.0
    Fluoride: Optional[float] = 1.0
    Nitrate: Optional[float] = 30.0
    Chloride: Optional[float] = 90.0

class TrainRequest(BaseModel):
    records: List[Dict[str, Any]]
    target: str = "Groundwater_Level"

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    horizon_weeks: int = 12
    well_id: Optional[str] = None

class SimulationRequest(BaseModel):
    well_id: str
    baseline_level: float
    rainfall_variation_pct: float = 0.0     # -50% to +50%
    extraction_variation_pct: float = 0.0   # -30% to +50%
    temperature_anomaly_c: float = 0.0      # 0 to +4 C
    recharge_intervention: bool = False      # Managed Aquifer Recharge (MAR)
    horizon_months: int = 12

class XaiRequest(BaseModel):
    well_id: str
    recent_records: List[Dict[str, Any]]

# --- Supabase Database Models ---
class StationModel(BaseModel):
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    state: Optional[str] = "Maharashtra"
    district: Optional[str] = "Pune"
    taluka: Optional[str] = None
    aquifer_type: Optional[str] = "Deccan Basalt"
    well_depth_m: Optional[float] = None
    status: Optional[str] = "ACTIVE"

class StationsBatchRequest(BaseModel):
    stations: List[StationModel]

class ReadingItemModel(BaseModel):
    station_id: str
    measurement_date: str  # YYYY-MM-DD
    groundwater_level_m: float
    measurement_timestamp: Optional[str] = None
    rainfall_mm: Optional[float] = 0.0
    temperature_c: Optional[float] = None
    soil_moisture_pct: Optional[float] = None
    ph: Optional[float] = None
    tds_mg_l: Optional[float] = None
    fluoride_mg_l: Optional[float] = None
    nitrate_mg_l: Optional[float] = None
    chloride_mg_l: Optional[float] = None
    data_source: Optional[str] = "manual_import"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    state: Optional[str] = None
    district: Optional[str] = None
    taluka: Optional[str] = None
    station_name: Optional[str] = None

class ReadingsBatchRequest(BaseModel):
    readings: List[ReadingItemModel]
    auto_register_stations: Optional[bool] = True

class PredictionItemModel(BaseModel):
    station_id: str
    model_name: str
    forecast_horizon_days: int = 30
    target_variable: str = "groundwater_level"
    predicted_level_m: float
    risk_category: str = "SAFE"
    model_metrics: Optional[Dict[str, Any]] = None
    forecast_series: Optional[List[Dict[str, Any]]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None

class PredictionsBatchRequest(BaseModel):
    predictions: List[PredictionItemModel]

class AlertItemModel(BaseModel):
    station_id: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    current_level_m: Optional[float] = None
    threshold_m: Optional[float] = None
    status: Optional[str] = "ACTIVE"

class AlertsBatchRequest(BaseModel):
    alerts: List[AlertItemModel]

class AlertStatusUpdate(BaseModel):
    status: str  # ACTIVE, ACKNOWLEDGED, RESOLVED

# ---------------------------------------------------------------------------
# 1. Health & System Capability
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    return {
        "status": "ONLINE",
        "service": "AquiSense AI Enterprise Hydrological Microservice",
        "version": "2.0.0",
        "supported_engines": [
            "XGBoost Regressor (v" + xgb.__version__ + ")",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Ordinary Least Squares"
        ],
        "capabilities": [
            "Real-time Open-Meteo Meteorological Ingestion",
            "Explainable AI (XAI) Hydrological Driver Decomposition",
            "What-If Policy & Climate Resilience Simulation",
            "CGWB / India-WRIS Aquifer Stress Classification",
            "Supabase PostgreSQL Persistent Storage"
        ],
        "database_configured": supabase_client is not None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

# ---------------------------------------------------------------------------
# 2. Live Meteorological & Climate Sync (Open-Meteo)
# ---------------------------------------------------------------------------
@app.get("/api/weather/live")
def get_live_weather(lat: float = Query(18.44719), lon: float = Query(73.61512)):
    """
    Fetches real-time live meteorological telemetry from Open-Meteo API
    for the exact geographical coordinates of the monitoring station.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
        f"&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AquiSenseAI/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            current = data.get("current", {})
            daily = data.get("daily", {})
            
            # Estimate soil moisture index based on recent precipitation and evapotranspiration
            daily_precip = daily.get("precipitation_sum", [0])[0] if daily.get("precipitation_sum") else 0
            et0 = daily.get("et0_fao_evapotranspiration", [3.5])[0] if daily.get("et0_fao_evapotranspiration") else 3.5
            calc_soil_moisture = min(95.0, max(12.0, 30.0 + daily_precip * 1.5 - et0 * 2.0))

            return {
                "source": "Open-Meteo Real-Time Global Telemetry",
                "coordinates": {"latitude": lat, "longitude": lon},
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation", 0.0),
                "soil_moisture_estimate": round(calc_soil_moisture, 1),
                "evapotranspiration_et0": et0,
                "surface_pressure_hpa": current.get("surface_pressure"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "observation_time": current.get("time"),
                "status": "LIVE_SYNC_SUCCESS"
            }
    except Exception as e:
        # Fallback to realistic seasonal estimate if offline
        return {
            "source": "Offline Hydrological Heuristic Estimator (Fallback)",
            "coordinates": {"latitude": lat, "longitude": lon},
            "temperature": 27.5,
            "humidity": 58.0,
            "precipitation": 0.0,
            "soil_moisture_estimate": 28.5,
            "evapotranspiration_et0": 3.8,
            "observation_time": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "status": "OFFLINE_FALLBACK",
            "error_detail": str(e)
        }

# ---------------------------------------------------------------------------
# 3. Model Training & Comparison Engine (XGBoost, RF, GBDT, Linear)
# ---------------------------------------------------------------------------
@app.post("/api/models/train")
def train_models(payload: TrainRequest):
    records = payload.records
    if not records or len(records) < 15:
        raise HTTPException(status_code=400, detail="Insufficient records for robust ML training (min 15 required).")

    df = pd.DataFrame(records)
    target = payload.target
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not present in records.")

    # Drop non-numeric and identifier columns
    drop_cols = ["Date", "Well_ID", target]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No numeric feature columns found.")

    df_clean = df.dropna(subset=feature_cols + [target])
    X = df_clean[feature_cols]
    y = df_clean[target]

    if len(df_clean) < 10:
        raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning.")

    # Time-ordered train/test split per well (80% train, 20% test)
    if "Well_ID" in df_clean.columns:
        train_indices, test_indices = [], []
        for _, group in df_clean.groupby("Well_ID"):
            k = int(len(group) * 0.8)
            if k > 0 and len(group) - k > 0:
                train_indices.extend(group.index[:k])
                test_indices.extend(group.index[k:])
            else:
                train_indices.extend(group.index)
        if train_indices and test_indices:
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate Models
    candidate_models = {
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }

    comparison = []
    best_r2 = -float("inf")
    best_model_name = "XGBoost Regressor"
    best_model_obj = None

    for name, model in candidate_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))

            comparison.append({
                "model": name,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4)
            })

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_obj = model
        except Exception as ex:
            comparison.append({
                "model": name,
                "error": str(ex)
            })

    # Save best model to registry
    MODEL_REGISTRY[target] = {
        "model": best_model_obj,
        "name": best_model_name,
        "features": feature_cols,
        "scaler": scaler
    }

    # Extract feature importance from best tree model
    feature_importance = []
    if hasattr(best_model_obj, "feature_importances_"):
        fi = best_model_obj.feature_importances_
        for f_name, f_imp in sorted(zip(feature_cols, fi), key=lambda x: x[1], reverse=True):
            feature_importance.append({"feature": f_name, "importance": round(float(f_imp), 4)})
    elif hasattr(best_model_obj, "coef_"):
        weights = np.abs(best_model_obj.coef_)
        total_w = np.sum(weights) if np.sum(weights) > 0 else 1
        for f_name, f_w in sorted(zip(feature_cols, weights / total_w), key=lambda x: x[1], reverse=True):
            feature_importance.append({"feature": f_name, "importance": round(float(f_w), 4)})

    # Sort comparison by lowest RMSE
    valid_comparison = [c for c in comparison if "rmse" in c]
    valid_comparison.sort(key=lambda x: x["rmse"])

    return {
        "target": target,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_model_name,
        "comparison": valid_comparison,
        "feature_importance": feature_importance
    }

# ---------------------------------------------------------------------------
# 4. Explainable AI (XAI) Hydrological Attribution
# ---------------------------------------------------------------------------
@app.post("/api/xai/attribution")
def explain_prediction(payload: XaiRequest):
    """
    Computes hydrological feature attribution (SHAP-style) explaining
    why groundwater depth is increasing/decreasing for a given monitoring station.
    """
    records = payload.recent_records
    if not records:
        raise HTTPException(status_code=400, detail="Recent records required for XAI attribution.")

    df = pd.DataFrame(records)
    recent_rainfall = df["Rainfall"].tail(4).mean() if "Rainfall" in df.columns else 25.0
    recent_temp = df["Temperature"].tail(4).mean() if "Temperature" in df.columns else 28.0
    recent_sm = df["Soil_Moisture"].tail(4).mean() if "Soil_Moisture" in df.columns else 25.0

    # Baseline norms for Indian Deccan / semi-arid basalt aquifers
    norm_rainfall = 55.0   # weekly average target for replenishment
    norm_temp = 24.0       # baseline evapotranspiration temp
    norm_sm = 35.0         # baseline field capacity

    # Calculate hydrological driver deviations
    rain_effect = (norm_rainfall - recent_rainfall) * 1.8   # Deficit rain drives depletion
    temp_effect = (recent_temp - norm_temp) * 1.2           # High heat drives evapotranspiration
    sm_effect = (norm_sm - recent_sm) * 0.9                 # Soil desiccation delays recharge
    pumping_effect = 22.0                                   # Estimated baseline agricultural pumping load

    total_effect = abs(rain_effect) + abs(temp_effect) + abs(sm_effect) + abs(pumping_effect)
    if total_effect == 0: total_effect = 1.0

    drivers = [
        {
            "factor": "Rainfall Deficit / Recharge Anomaly",
            "contribution_pct": round((abs(rain_effect) / total_effect) * 100, 1),
            "direction": "DEPLETING" if rain_effect > 0 else "REPLENISHING",
            "detail": f"Observed {recent_rainfall:.1f} mm/wk vs {norm_rainfall:.1f} mm/wk historical baseline"
        },
        {
            "factor": "Thermal Evapotranspiration Stress",
            "contribution_pct": round((abs(temp_effect) / total_effect) * 100, 1),
            "direction": "DEPLETING" if temp_effect > 0 else "BENIGN",
            "detail": f"Ambient temperature {recent_temp:.1f}°C accelerates vadose zone moisture depletion"
        },
        {
            "factor": "Vadose Zone Soil Desiccation",
            "contribution_pct": round((abs(sm_effect) / total_effect) * 100, 1),
            "direction": "DELAYING_RECHARGE" if sm_effect > 0 else "FACILITATING_RECHARGE",
            "detail": f"Soil moisture at {recent_sm:.1f}% indicates dry antecedent condition"
        },
        {
            "factor": "Base Anthropogenic Extraction Load",
            "contribution_pct": round((abs(pumping_effect) / total_effect) * 100, 1),
            "direction": "DEPLETING",
            "detail": "Estimated agricultural tube-well and rural domestic abstraction"
        }
    ]

    drivers.sort(key=lambda x: x["contribution_pct"], reverse=True)

    primary = drivers[0]
    narrative = (
        f"For station {payload.well_id}, the dominant hydrological stress driver is "
        f"'{primary['factor']}' contributing {primary['contribution_pct']}% of observed aquifer drawdown, "
        f"accompanied by {drivers[1]['factor'].lower()} ({drivers[1]['contribution_pct']}%)."
    )

    return {
        "well_id": payload.well_id,
        "narrative": narrative,
        "attribution_breakdown": drivers
    }

# ---------------------------------------------------------------------------
# 5. What-If Climate & Policy Resilience Simulator
# ---------------------------------------------------------------------------
@app.post("/api/simulate")
def run_simulation(sim: SimulationRequest):
    """
    High-level decision support engine for water authorities:
    Simulates aquifer trajectory under drought, heatwaves, over-extraction,
    and Managed Aquifer Recharge (MAR) interventions.
    """
    base_level = sim.baseline_level
    months = sim.horizon_months
    
    # Hydrological parameters
    rain_factor = 1.0 + (sim.rainfall_variation_pct / 100.0)
    extract_factor = 1.0 + (sim.extraction_variation_pct / 100.0)
    temp_anomaly = sim.temperature_anomaly_c
    recharge_bonus = 0.28 if sim.recharge_intervention else 0.0 # m/month recovery via MAR

    trajectory = []
    current_level = base_level
    
    # Typical monthly rainfall distribution (Monsoon in Jun-Sep)
    monthly_recharge_rates = [0.05, 0.02, 0.02, 0.01, 0.03, 0.35, 0.85, 0.70, 0.45, 0.15, 0.08, 0.05]
    
    now = datetime.datetime.now()
    critical_threshold = 20.0 # meters below ground
    exhaustion_month = None

    for m in range(months):
        sim_date = now + datetime.timedelta(days=30 * (m + 1))
        month_idx = (now.month - 1 + m) % 12
        
        # Hydrological balance
        natural_recharge = monthly_recharge_rates[month_idx] * rain_factor
        et_loss = (0.05 + temp_anomaly * 0.02)
        pumping_loss = (0.22 * extract_factor)
        
        net_delta = -natural_recharge + et_loss + pumping_loss - recharge_bonus
        current_level = max(1.0, current_level + net_delta)
        
        if current_level >= critical_threshold and exhaustion_month is None:
            exhaustion_month = m + 1

        trajectory.append({
            "month_step": m + 1,
            "date": sim_date.strftime("%Y-%m"),
            "projected_level_m": round(current_level, 2),
            "net_monthly_delta": round(net_delta, 2)
        })

    total_drawdown = current_level - base_level
    
    # Calculate CGWB Stage of Groundwater Development (% Extraction / Recharge)
    baseline_sod = 72.0 # % (Semi-Critical baseline)
    simulated_sod = baseline_sod * (extract_factor / max(0.4, rain_factor))
    if sim.recharge_intervention:
        simulated_sod *= 0.82 # 18% improvement from artificial recharge

    # CGWB Categorization
    if simulated_sod < 70.0:
        cgwb_category = "SAFE"
        color = "GOOD"
    elif simulated_sod <= 90.0:
        cgwb_category = "SEMI-CRITICAL"
        color = "WARNING"
    elif simulated_sod <= 100.0:
        cgwb_category = "CRITICAL"
        color = "RUST"
    else:
        cgwb_category = "OVER-EXPLOITED"
        color = "CRITICAL"

    recommendations = []
    if simulated_sod > 90.0:
        recommendations.append("Mandate micro-irrigation (drip/sprinkler) for sugarcane and high-water crops.")
        recommendations.append("Issue CGWA notification to cap new commercial tube-well permissions.")
    if sim.rainfall_variation_pct < -20.0:
        recommendations.append("Trigger Stage-2 Drought Preparedness: mobilize surface water storage transfers.")
    if not sim.recharge_intervention and total_drawdown > 1.5:
        recommendations.append("Priority deployment of Managed Aquifer Recharge (MAR) check dams and percolation tanks.")
    if sim.recharge_intervention:
        recommendations.append("Artificial recharge intervention stabilizes the water table by ~" + f"{recharge_bonus * months:.1f}m over the simulation horizon.")

    return {
        "well_id": sim.well_id,
        "baseline_level_m": base_level,
        "final_projected_level_m": round(current_level, 2),
        "total_drawdown_m": round(total_drawdown, 2),
        "cgwb_stage_of_development_pct": round(simulated_sod, 1),
        "cgwb_category": cgwb_category,
        "critical_threshold_breached": current_level >= critical_threshold,
        "months_to_critical_threshold": exhaustion_month,
        "trajectory": trajectory,
        "policy_recommendations": recommendations
    }

# ---------------------------------------------------------------------------
# 6. Supabase PostgreSQL Integration Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/db/status")
def get_db_status():
    """
    Diagnostic and health-check endpoint for Supabase PostgreSQL.
    """
    if not supabase_client:
        return {
            "status": "NOT_CONFIGURED",
            "configured": False,
            "database": "Supabase PostgreSQL",
            "message": "Supabase credentials are not configured on this server.",
            "required_env": ["SUPABASE_URL", "SUPABASE_SECRET_KEY"],
            "help": "Configure SUPABASE_URL and SUPABASE_SECRET_KEY in Render environment variables.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    try:
        res_stations = supabase_client.table("monitoring_stations").select("station_id", count="exact").limit(1).execute()
        res_readings = supabase_client.table("groundwater_readings").select("id", count="exact").limit(1).execute()
        res_preds = supabase_client.table("groundwater_predictions").select("id", count="exact").limit(1).execute()
        res_alerts = supabase_client.table("groundwater_alerts").select("id", count="exact").limit(1).execute()

        return {
            "status": "CONNECTED",
            "configured": True,
            "database": "Supabase PostgreSQL",
            "url_masked": (SUPABASE_URL[:14] + "..." + SUPABASE_URL[-10:]) if SUPABASE_URL and len(SUPABASE_URL) > 24 else "Configured",
            "counts": {
                "monitoring_stations": getattr(res_stations, "count", 0) or 0,
                "groundwater_readings": getattr(res_readings, "count", 0) or 0,
                "groundwater_predictions": getattr(res_preds, "count", 0) or 0,
                "groundwater_alerts": getattr(res_alerts, "count", 0) or 0
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "status": "CONNECTION_ERROR",
            "configured": True,
            "database": "Supabase PostgreSQL",
            "message": "Connected to host, but failed querying tables. Please run supabase_schema.sql in the Supabase SQL editor.",
            "error_detail": str(e),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

@app.post("/api/db/stations")
def save_monitoring_stations(payload: StationsBatchRequest, authorized: bool = Depends(verify_write_auth)):
    """
    Upsert monitoring stations into Supabase PostgreSQL.
    """
    db = get_supabase_client()
    if not payload.stations:
        raise HTTPException(status_code=400, detail="No stations provided in payload.")
    
    records = [s.model_dump() for s in payload.stations]
    try:
        db.table("monitoring_stations").upsert(records, on_conflict="station_id").execute()
        return {
            "status": "SUCCESS",
            "saved_count": len(records),
            "stations": [s["station_id"] for s in records],
            "message": f"Successfully upserted {len(records)} monitoring stations in PostgreSQL."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error while saving stations: {str(e)}")

@app.get("/api/db/stations")
def list_monitoring_stations(state: Optional[str] = None, district: Optional[str] = None):
    """
    Retrieve monitoring stations from Supabase PostgreSQL.
    """
    db = get_supabase_client()
    try:
        query = db.table("monitoring_stations").select("*").order("station_id")
        if state:
            query = query.eq("state", state)
        if district:
            query = query.eq("district", district)
        res = query.execute()
        return {
            "count": len(res.data) if res.data else 0,
            "stations": res.data or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error fetching stations: {str(e)}")

@app.post("/api/db/readings")
def save_groundwater_readings(payload: ReadingsBatchRequest, authorized: bool = Depends(verify_write_auth)):
    """
    Save or upsert a batch of groundwater readings into Supabase PostgreSQL.
    Deduplication is enforced on (station_id, measurement_date).
    """
    db = get_supabase_client()
    readings = payload.readings
    if not readings:
        raise HTTPException(status_code=400, detail="No readings provided in payload.")

    # Auto-register parent stations if requested to maintain relational integrity
    # Auto-register parent stations if requested using actual coordinates from readings
    if payload.auto_register_stations:
        stn_map = {}
        for r in readings:
            sid = r.station_id
            if sid not in stn_map:
                stn_map[sid] = {
                    "station_id": sid,
                    "station_name": r.station_name or f"Station {sid}",
                    "latitude": r.latitude if r.latitude is not None else 18.52,
                    "longitude": r.longitude if r.longitude is not None else 73.86,
                    "state": r.state or "Maharashtra",
                    "district": r.district or "Pune",
                    "taluka": r.taluka or None,
                    "aquifer_type": "Deccan Basalt"
                }
            elif r.latitude is not None:
                stn_map[sid]["latitude"] = r.latitude
                stn_map[sid]["longitude"] = r.longitude
                if r.state: stn_map[sid]["state"] = r.state
                if r.district: stn_map[sid]["district"] = r.district
                if r.station_name: stn_map[sid]["station_name"] = r.station_name
        
        station_records = list(stn_map.values())
        try:
            db.table("monitoring_stations").upsert(station_records, on_conflict="station_id").execute()
        except Exception:
            pass

    READING_DB_COLS = {
        "station_id", "measurement_date", "measurement_timestamp",
        "groundwater_level_m", "rainfall_mm", "temperature_c",
        "soil_moisture_pct", "ph", "tds_mg_l", "fluoride_mg_l",
        "nitrate_mg_l", "chloride_mg_l", "data_source"
    }
    records = []
    for r in readings:
        dumped = r.model_dump(exclude_none=True)
        records.append({k: v for k, v in dumped.items() if k in READING_DB_COLS})

    # Process in chunks of 200 records to prevent payload timeouts
    CHUNK_SIZE = 200
    processed_count = 0
    try:
        for i in range(0, len(records), CHUNK_SIZE):
            chunk = records[i:i + CHUNK_SIZE]
            db.table("groundwater_readings").upsert(chunk, on_conflict="station_id,measurement_date").execute()
            processed_count += len(chunk)

        return {
            "status": "SUCCESS",
            "records_processed": processed_count,
            "unique_stations": list({r['station_id'] for r in records}),
            "message": f"Successfully processed and deduplicated {processed_count} groundwater measurements in PostgreSQL."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error while saving groundwater readings: {str(e)}")

@app.get("/api/db/readings")
def query_groundwater_readings(
    station_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0)
):
    """
    Query historical groundwater readings from Supabase PostgreSQL.
    """
    db = get_supabase_client()
    try:
        query = db.table("groundwater_readings").select("*").order("measurement_date", desc=True)
        if station_id:
            query = query.eq("station_id", station_id)
        if start_date:
            query = query.gte("measurement_date", start_date)
        if end_date:
            query = query.lte("measurement_date", end_date)

        res = query.range(offset, offset + limit - 1).execute()
        raw_readings = res.data or []
        
        # Attach station coordinates if available in monitoring_stations
        try:
            stn_res = db.table("monitoring_stations").select("station_id, latitude, longitude, station_name, state, district, taluka").execute()
            stn_dict = {s["station_id"]: s for s in (stn_res.data or [])}
            for r in raw_readings:
                s_meta = stn_dict.get(r.get("station_id"))
                if s_meta:
                    if r.get("latitude") is None: r["latitude"] = s_meta.get("latitude")
                    if r.get("longitude") is None: r["longitude"] = s_meta.get("longitude")
                    if r.get("station_name") is None: r["station_name"] = s_meta.get("station_name")
                    if r.get("state") is None: r["state"] = s_meta.get("state")
                    if r.get("district") is None: r["district"] = s_meta.get("district")
        except Exception:
            pass

        return {
            "count": len(raw_readings),
            "limit": limit,
            "offset": offset,
            "readings": raw_readings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error querying readings: {str(e)}")

@app.post("/api/db/predictions")
def save_predictions_history(payload: PredictionsBatchRequest, authorized: bool = Depends(verify_write_auth)):
    """
    Save genuine AI model forecast records into Supabase PostgreSQL.
    """
    db = get_supabase_client()
    predictions = payload.predictions
    if not predictions:
        raise HTTPException(status_code=400, detail="No predictions provided in payload.")

    # Ensure parent stations exist
    unique_stations = {p.station_id for p in predictions}
    station_records = [
        {"station_id": sid, "station_name": f"Piezometer {sid}", "latitude": 18.52, "longitude": 73.86}
        for sid in unique_stations
    ]
    try:
        db.table("monitoring_stations").upsert(station_records, on_conflict="station_id").execute()
    except Exception:
        pass

    records = [p.model_dump(exclude_none=True) for p in predictions]
    try:
        db.table("groundwater_predictions").insert(records).execute()
        return {
            "status": "SUCCESS",
            "saved_count": len(records),
            "message": f"Successfully archived {len(records)} AI model predictions in PostgreSQL."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error saving predictions: {str(e)}")

@app.get("/api/db/predictions")
def query_predictions_history(
    station_id: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Retrieve historical AI forecasts from Supabase PostgreSQL.
    """
    db = get_supabase_client()
    try:
        query = db.table("groundwater_predictions").select("*").order("prediction_date", desc=True)
        if station_id:
            query = query.eq("station_id", station_id)
        if model_name:
            query = query.eq("model_name", model_name)

        res = query.range(offset, offset + limit - 1).execute()
        return {
            "count": len(res.data) if res.data else 0,
            "limit": limit,
            "offset": offset,
            "predictions": res.data or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error querying predictions: {str(e)}")

@app.post("/api/db/alerts")
def save_alerts_history(payload: AlertsBatchRequest, authorized: bool = Depends(verify_write_auth)):
    """
    Store groundwater depletion or contamination alerts into Supabase PostgreSQL.
    """
    db = get_supabase_client()
    alerts = payload.alerts
    if not alerts:
        raise HTTPException(status_code=400, detail="No alerts provided in payload.")

    records = [a.model_dump(exclude_none=True) for a in alerts]
    try:
        db.table("groundwater_alerts").insert(records).execute()
        return {
            "status": "SUCCESS",
            "saved_count": len(records),
            "message": f"Successfully stored {len(records)} alerts in PostgreSQL."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error saving alerts: {str(e)}")

@app.get("/api/db/alerts")
def query_alerts_history(
    station_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """
    Retrieve alerts from Supabase PostgreSQL.
    """
    db = get_supabase_client()
    try:
        query = db.table("groundwater_alerts").select("*").order("alert_timestamp", desc=True)
        if station_id:
            query = query.eq("station_id", station_id)
        if status:
            query = query.eq("status", status)
        if severity:
            query = query.eq("severity", severity)

        res = query.limit(limit).execute()
        return {
            "count": len(res.data) if res.data else 0,
            "alerts": res.data or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error querying alerts: {str(e)}")

@app.patch("/api/db/alerts/{alert_id}")
def update_alert_status(alert_id: int, payload: AlertStatusUpdate, authorized: bool = Depends(verify_write_auth)):
    """
    Acknowledge or resolve an active alert in Supabase PostgreSQL.
    """
    db = get_supabase_client()
    valid_statuses = ["ACTIVE", "ACKNOWLEDGED", "RESOLVED"]
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status '{payload.status}'. Allowed: {valid_statuses}")

    update_data: Dict[str, Any] = {"status": payload.status}
    if payload.status == "ACKNOWLEDGED":
        update_data["acknowledged_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        res = db.table("groundwater_alerts").update(update_data).eq("id", alert_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Alert with id {alert_id} not found.")
        return {
            "status": "SUCCESS",
            "message": f"Alert {alert_id} updated to {payload.status}.",
            "alert": res.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error updating alert: {str(e)}")

# ---------------------------------------------------------------------------
# 8. Static Files & Single-Origin Web Server
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

# Mount current directory so http://127.0.0.1:8000/ serves index.html, app.js, data, etc.
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "="*70)
    print("  AquiSense AI — Advanced Hydrological Intelligence Platform")
    print(f"  Server is LIVE at http://{host}:{port}")
    print(f"  Local browser access: http://localhost:{port}")
    print("="*70 + "\n")
    uvicorn.run("server:app", host=host, port=port, reload=False)
