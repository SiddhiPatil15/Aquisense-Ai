# Aqui Sense AI — Groundwater Prediction & Early Warning Prototype

A working, browser-based prototype of an AI-powered groundwater prediction and
early-warning system. Every figure the app shows — validation counts, model
metrics, WQI, risk, alerts — is computed live from whichever dataset is
loaded. Nothing is hard-coded or faked.

## What this is (and isn't)

The original brief asked for a full production stack: React/TypeScript
frontend, FastAPI backend, PostgreSQL database, and a Python ML service
(scikit-learn / XGBoost) behind REST APIs. That stack can't be stood up and
hosted persistently from this delivery format, so this package takes a
different, honest path: **a single-page app that runs the entire pipeline
client-side**, in plain HTML/CSS/JavaScript, with no build step and no
server required.

That means:
- Data upload, validation, feature engineering, model training/evaluation,
  WQI, risk scoring, alerts, mapping, and reporting are all real,
  computed operations — not mocked screens.
- It is **single-user and in-memory**: nothing persists after you close the
  tab, and there's no multi-user database. Section 3/26/27 of the original
  spec describe the natural next step (FastAPI + PostgreSQL) if this needs
  to become a shared, persisted, multi-user system — see "Path to the full
  stack" below.
- Random Forest / XGBoost (which need a Python runtime) are represented
  honestly by a real, from-scratch Decision Tree regressor and a
  persistence baseline, alongside a real multiple linear regression. All
  three are genuinely trained and evaluated on a held-out, time-ordered
  test split — MAE / RMSE / R² are computed, never entered by hand.

## Running it

No install, no build, no server. Just open `index.html` in a modern browser
(Chrome, Edge, Firefox, Safari). It loads Chart.js, Leaflet, PapaParse, and
jsPDF from a CDN, so an internet connection is needed for those libraries
the first time.

If your browser blocks `fetch`/module behavior on `file://` URLs, serve the
folder locally instead:

```bash
cd aquifer-sense-prototype
python3 -m http.server 8000
# then open http://localhost:8000
```

## Using the prototype

1. **Home** → "Create Project".
2. **Projects** → fill in the project form (name, location, number of
   wells) and create it.
3. **Data Upload** → either:
   - drag in a CSV with the columns in `data/templates/groundwater_sample_template.csv`, or
   - click **Load demo project** to use the bundled, clearly-labeled
     synthetic dataset (`data/sample/synthetic_groundwater_sample.csv`,
     also generated live in-browser), or
   - click **Download sample template** for a starter file.
4. **Validation** → see real counts for missing values, duplicates,
   invalid dates, bad coordinates, range flags, and IQR-based outliers,
   plus a per-record quality flag (Valid / Review / Invalid).
5. **Prediction** → compares Linear Regression, Decision Tree, and a
   persistence baseline on a time-ordered train/test split, picks the
   model with the lowest RMSE, and forecasts each well forward.
6. **Water Quality** → WQI (weighted arithmetic index) per well, with the
   contributing parameters shown.
7. **Risk & Alerts** → a rule-based SAFE / WARNING / CRITICAL classification
   per well, dynamically generated alerts, and plain-language
   recommendations.
8. **Map** → wells plotted with Leaflet using the coordinates in your data,
   colored by risk; click a marker for well detail.
9. **Dashboard** → the top-level KPI view.
10. **Reports** → a text summary of the whole pipeline, downloadable as PDF.

## Important limitations (read before using this for real decisions)

- **Reference values are configurable defaults, not a cited regulatory
  standard.** The range-check thresholds and WQI standard values in
  `app.js` (`DEFAULT_THRESHOLDS`, `DEFAULT_WQI_STANDARDS`) are placeholders
  so the pipeline has something to score against. Replace them with the
  standard that applies in your jurisdiction before treating any output as
  a safety determination.
- **Risk levels and alerts are a prototype heuristic**, not a formal
  hydrological or government classification.
- **The ML models are simple and client-side.** They're real (genuinely
  trained, genuinely evaluated) but a browser is not the place to run
  Random Forest / XGBoost / LSTM at scale — see below for how to extend
  this into a proper backend.
- **No persistence.** Refreshing the page clears all projects and data.
- Recommendations are decision-support suggestions, not a substitute for
  professional hydrological assessment or laboratory testing.

## Path to the full stack described in the original brief

If/when this needs to be a real multi-user product, the natural next step
(matching sections 3, 26, 27 of the original spec) is:

- **Backend:** FastAPI service exposing the endpoints already sketched in
  the spec (`/projects`, `/datasets/{id}/validate`, `/projects/{id}/train`,
  `/projects/{id}/predict`, `/projects/{id}/wqi`, `/projects/{id}/risk`,
  `/projects/{id}/report`, etc.). Most of the JavaScript in `app.js`
  (validation rules, feature engineering, WQI formula, risk rules) is
  already organized as small, pure functions — porting the logic to Python
  is mostly a translation exercise, not a redesign.
- **ML:** move `trainAndEvaluate` to a Python service using
  scikit-learn (Linear Regression, Random Forest, SVM) and XGBoost, with
  the same time-aware train/test split and MAE/RMSE/R² evaluation this
  prototype already does.
- **Database:** PostgreSQL with the tables listed in the spec (`projects`,
  `monitoring_wells`, `groundwater_records`, `water_quality_records`,
  `validation_results`, `predictions`, `wqi_results`, `risk_assessments`,
  `alerts`, `recommendations`, `reports`).
- **Frontend:** the current HTML/CSS/JS views map fairly directly onto
  React components/pages if a React rewrite is wanted, since the app is
  already organized as one render function per screen.

## File structure

```
aquifer-sense-prototype/
├── index.html                 # app shell
├── app.js                     # entire app: state, pipeline, UI
├── data/
│   ├── sample/synthetic_groundwater_sample.csv   # labeled synthetic dataset
│   └── templates/groundwater_sample_template.csv # blank starter template
└── README.md
```
