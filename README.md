# AquiSense AI — Advanced Groundwater Intelligence & Early Warning Platform

An enterprise-grade, high-level **Hydrological Decision Support & Early Warning Platform** designed for water resources departments, environmental authorities (CGWB / CGWA / India-WRIS), and watershed management planners.

- **Live Frontend (Vercel)**: [https://aquisense-ai.vercel.app/](https://aquisense-ai.vercel.app/)
- **Backend Service (Render)**: Python FastAPI Microservice with XGBoost, Open-Meteo REST API, and Supabase PostgreSQL.
- **GitHub Repository**: [https://github.com/SiddhiPatil15/Aquisense-Ai](https://github.com/SiddhiPatil15/Aquisense-Ai)

AquiSense AI combines **machine learning (XGBoost & Random Forest ensembles)**, **Explainable AI (XAI)**, **real-time meteorological telemetry synchronization (Open-Meteo REST API)**, an interactive **Policy & Climate Resilience Simulator**, and **persistent cloud storage powered by Supabase PostgreSQL**.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME DATA INGESTION LAYER                       │
│  • Open-Meteo REST API (Live Precipitation, Temp, Evapotranspiration) │
│  • CGWB / India-WRIS Standard Water Quality & Piezometer Networks       │
│  • Field Observational Groundwater CSV Ingestion & Bulk Imports         │
│  • Supabase PostgreSQL Relational Storage (Time-Series Measurements)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                AQUISENSE AI MACHINE LEARNING MICROSERVICE              │
│                          (FastAPI + Python)                            │
│  • Multi-Model Leaderboard: XGBoost, Random Forest, GBDT, Ridge        │
│  • Time-Series Lag Engineering (4-Week Rolling Precip, Antecedent SM)   │
│  • Explainable AI (XAI): Hydrological Feature Attribution (SHAP-style)  │
│  • Policy & Climate "What-If" Simulation Engine (CGWB Stress Testing)  │
│  • Relational Persistence Gateway: Deduplication & Archival via Supabase│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  DECISION SUPPORT & ACTION LAYER                       │
│  • Dynamic Aquifer Drawdown Hydrographs (12-Month Projections)         │
│  • CGWB Stage of Groundwater Development (% Extraction / Recharge)     │
│  • Automated Early Warning Dispatch (Safe / Semi-Critical / Critical)  │
│  • Managed Aquifer Recharge (MAR) Site & Intervention Modeling         │
│  • Cloud Database & Audit History (Load Past Readings on Refresh)      │
│  • Executive Hydrological Audit PDF Reports                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Supabase PostgreSQL Database Integration

AquiSense AI utilizes a relational PostgreSQL database on **Supabase** for persistent storage, historical comparison, and audit logging.

### Relational Schema Tables:
1. `monitoring_stations`: Registry of telemetric piezometers and borewells with coordinates, aquifer classifications, depths, and operational statuses.
2. `groundwater_readings`: Time-series observations storing depth to water table ($m$ bgl), rainfall ($mm$), ambient temperature, soil moisture, and chemical parameters ($pH$, $TDS$, Fluoride, Nitrate, Chloride).
   - **Duplicate Prevention**: Enforces a strict unique constraint `UNIQUE(station_id, measurement_date)` to detect and prevent duplicate readings idempotently.
3. `groundwater_predictions`: Archival log of genuine AI model forecasts, model names (e.g. XGBoost Regressor), forecast horizons (days), predicted levels, risk categories, and performance metrics ($R^2$, $RMSE$).
4. `groundwater_alerts`: Active and acknowledged early warning signals, severity rankings (Low to Critical), depth thresholds, and resolution audit trails.

---

## ⚙️ Setup & Deployment Guide

### Step 1: Create Supabase Project & Run SQL Schema
1. Sign in to [supabase.com](https://supabase.com/) and create a new project.
2. In your project dashboard, navigate to the **SQL Editor** on the left menu.
3. Open the file `supabase_schema.sql` from this repository, copy its entire contents, paste it into the SQL Editor, and click **Run**.
4. The tables `monitoring_stations`, `groundwater_readings`, `groundwater_predictions`, and `groundwater_alerts` will be created with primary keys, indexes, and Row Level Security (RLS) policies.

### Step 2: Retrieve Supabase API Credentials
1. In Supabase, go to **Project Settings** > **API**.
2. Note down:
   - **Project URL**: `https://<your-project-id>.supabase.co`
   - **service_role Key** (Secret): Secret key for server-side FastAPI backend access.

### Step 3: Configure Environment Variables in Render
1. Open your [Render Dashboard](https://dashboard.render.com/) and select the `Aquisense-Ai` web service.
2. Go to **Environment** tab and add:
   - `SUPABASE_URL`: `https://<your-project-id>.supabase.co`
   - `SUPABASE_SECRET_KEY`: `<your-supabase-service-role-secret-key>`
   - `API_AUTH_KEY`: `<your-optional-write-protection-secret-key>` *(optional)*
   - `FRONTEND_ORIGINS`: `https://aquisense-ai.vercel.app,http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000`
3. Click **Save Changes**. Render will automatically redeploy the service.

### Step 4: Configure Frontend API Base URL (Vercel)
1. Open your live application at [https://aquisense-ai.vercel.app/](https://aquisense-ai.vercel.app/).
2. Click the **Database & History** tab on the left sidebar.
3. Under **Cloud Connection & API Endpoint Configuration**:
   - Verify the **Backend API Base URL** matches your Render URL (e.g., `https://aquisense-ai.onrender.com`).
   - If an `API_AUTH_KEY` was configured, enter it in the **API Authorization Key** field.
4. Click **Save Configuration & Test**.
5. The top status pill will turn green: `🗄 Supabase: Connected (X readings)`.

---

## 🧪 Testing & Verification

An automated integration and regression test suite is included in `test_supabase_integration.py`.

Run the test suite with:
```bash
python -m unittest test_supabase_integration.py -v
```

### Verified Test Cases:
- ✅ `test_01_api_status_capability`: Verifies `/api/status` exposes Supabase capability and configuration state.
- ✅ `test_02_db_status_unconfigured`: Verifies clean diagnostic response when credentials are not configured.
- ✅ `test_03_db_endpoints_503_when_unconfigured`: Verifies HTTP 503 Service Unavailable when calling read/write endpoints without database configuration.
- ✅ `test_04_write_authorization_protection`: Verifies HTTP 401 Unauthorized when write endpoints are accessed without proper `X-API-Key` or Bearer token.
- ✅ `test_05_validation_empty_payloads`: Verifies HTTP 400 Bad Request on empty or malformed batches.
- ✅ `test_06_mocked_supabase_readings_flow`: Tests batch upsert, deduplication on `(station_id, measurement_date)`, and query filtering.
- ✅ `test_07_mocked_supabase_predictions_flow`: Tests saving genuine AI model forecasts, metrics, and horizon series.
- ✅ `test_08_mocked_supabase_alerts_management`: Tests alert generation and status patching (e.g., acknowledging an alert).
- ✅ `test_09_regression_existing_endpoints`: Verifies zero regression across existing ML training, Open-Meteo live sync, and What-If simulators.

---

## 🚀 Running Locally

### Option A: Full-Stack Mode (FastAPI + Supabase + Web UI)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Supabase credentials:
   ```bash
   copy .env.example .env
   ```
3. Start the backend:
   ```bash
   python server.py
   # Or on Windows, double-click run_backend.bat
   ```
4. Access the platform at `http://127.0.0.1:8000`.

### Option B: Standalone Web Mode
- Simply open `index.html` in any modern browser.
- All client-side fallback engines (heuristic models and local analytics) remain operational even when offline.

---

## 📁 Repository Structure

```
Aquisense-Ai-main/
├── index.html                     # Enterprise application shell with system status top bar
├── app.js                         # UI router, ML pipelines, and Supabase client methods
├── server.py                      # FastAPI microservice with XGBoost, XAI, and Supabase endpoints
├── supabase_schema.sql            # PostgreSQL DDL migration script for Supabase
├── requirements.txt               # Pinned Python dependencies including official supabase client
├── .env.example                   # Clean template for environment variables
├── test_supabase_integration.py   # Automated unit and integration test suite
├── render.yaml                    # Render cloud deployment blueprint
├── Procfile                       # PaaS process file for Render and Railway
├── Dockerfile                     # Container definition for Docker & Hugging Face
├── .dockerignore                  # Docker exclusion rules
├── .gitignore                     # Git exclusion rules (.env, __pycache__, etc.)
├── run_backend.bat                # Windows quick launcher
└── README.md                      # Comprehensive technical documentation
```
