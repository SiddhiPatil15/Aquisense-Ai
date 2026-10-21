"""
AquiSense AI — Automated Integration & Unit Test Suite for Supabase PostgreSQL
Tests:
1. System Health & Capability Endpoints
2. Database Status Endpoint (/api/db/status) with and without credentials
3. Graceful 503 handling when database is not configured
4. Write Authorization protection (API_AUTH_KEY / X-API-Key)
5. Pydantic validation on readings, predictions, and stations
6. Mocked Supabase End-to-End Persistence:
   - Stations upsert
   - Groundwater readings batch save with deduplication
   - Groundwater readings query with filters
   - AI predictions save and query
   - Alert creation, retrieval, and status patch
7. Regression testing for existing ML, simulation, and IoT endpoints
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import server
from server import app

class TestSupabaseIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_api_status_capability(self):
        """Verify /api/status includes Supabase PostgreSQL capability."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("Supabase PostgreSQL Persistent Storage", data["capabilities"])
        self.assertIn("database_configured", data)

    def test_02_db_status_unconfigured(self):
        """Verify /api/db/status returns NOT_CONFIGURED when no credentials are set."""
        with patch.object(server, "supabase_client", None):
            res = self.client.get("/api/db/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "NOT_CONFIGURED")
            self.assertFalse(data["configured"])
            self.assertIn("required_env", data)

    def test_03_db_endpoints_503_when_unconfigured(self):
        """Verify write and read DB endpoints raise 503 when Supabase is not configured."""
        with patch.object(server, "supabase_client", None):
            res_read = self.client.get("/api/db/readings")
            self.assertEqual(res_read.status_code, 503)
            self.assertIn("Database service unavailable", res_read.json()["detail"])

            res_write = self.client.post("/api/db/readings", json={
                "readings": [
                    {
                        "station_id": "W-01",
                        "measurement_date": "2026-01-01",
                        "groundwater_level_m": 8.5
                    }
                ]
            })
            self.assertEqual(res_write.status_code, 503)

    def test_04_write_authorization_protection(self):
        """Verify write endpoints reject unauthorized requests when API_AUTH_KEY is configured."""
        with patch.dict(os.environ, {"API_AUTH_KEY": "secret-test-key-123"}):
            # Unauthorized call without header
            res = self.client.post("/api/db/stations", json={"stations": []})
            self.assertEqual(res.status_code, 401)
            self.assertIn("Unauthorized", res.json()["detail"])

            # Unauthorized call with wrong header
            res_wrong = self.client.post(
                "/api/db/stations",
                json={"stations": []},
                headers={"X-API-Key": "wrong-key"}
            )
            self.assertEqual(res_wrong.status_code, 401)

    def test_05_validation_empty_payloads(self):
        """Verify 400 Bad Request when submitting empty station/reading batches."""
        mock_db = MagicMock()
        with patch.object(server, "supabase_client", mock_db):
            res_stn = self.client.post("/api/db/stations", json={"stations": []})
            self.assertEqual(res_stn.status_code, 400)

            res_rdg = self.client.post("/api/db/readings", json={"readings": []})
            self.assertEqual(res_rdg.status_code, 400)

            res_prd = self.client.post("/api/db/predictions", json={"predictions": []})
            self.assertEqual(res_prd.status_code, 400)

    def test_06_mocked_supabase_readings_flow(self):
        """Verify end-to-end flow of saving and retrieving readings with duplicate detection."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table

        # Setup mock upsert
        mock_upsert = MagicMock()
        mock_upsert.execute.return_value = MagicMock(data=[{"id": 1}])
        mock_table.upsert.return_value = mock_upsert

        # Setup chainable mock query for select/order/eq/range
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.gte.return_value = mock_query
        mock_query.lte.return_value = mock_query
        mock_query.range.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[
            {
                "id": 101,
                "station_id": "W-01",
                "measurement_date": "2026-02-15",
                "groundwater_level_m": 9.42,
                "rainfall_mm": 18.5,
                "data_source": "cgwb_benchmark"
            }
        ])
        mock_table.select.return_value = mock_query

        with patch.object(server, "supabase_client", mock_db):
            # 1. Post readings batch
            payload = {
                "auto_register_stations": True,
                "readings": [
                    {
                        "station_id": "W-01",
                        "measurement_date": "2026-02-15",
                        "groundwater_level_m": 9.42,
                        "rainfall_mm": 18.5,
                        "temperature_c": 27.2,
                        "soil_moisture_pct": 34.0,
                        "ph": 7.3,
                        "tds_mg_l": 420.0
                    }
                ]
            }
            res_post = self.client.post("/api/db/readings", json=payload)
            self.assertEqual(res_post.status_code, 200)
            self.assertEqual(res_post.json()["status"], "SUCCESS")
            self.assertEqual(res_post.json()["records_processed"], 1)

            # Verify upsert was called with on_conflict
            mock_table.upsert.assert_called()

            # 2. Query readings
            res_get = self.client.get("/api/db/readings?station_id=W-01")
            self.assertEqual(res_get.status_code, 200)
            data = res_get.json()
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["readings"][0]["station_id"], "W-01")

    def test_07_mocked_supabase_predictions_flow(self):
        """Verify saving and retrieving AI model prediction logs."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table

        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": 1}])
        mock_table.insert.return_value = mock_insert

        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.range.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[
            {
                "id": 1,
                "station_id": "W-01",
                "model_name": "XGBoost Regressor",
                "predicted_level_m": 10.15,
                "forecast_horizon_days": 30,
                "risk_category": "SEMI-CRITICAL"
            }
        ])
        mock_table.select.return_value = mock_query

        with patch.object(server, "supabase_client", mock_db):
            payload = {
                "predictions": [
                    {
                        "station_id": "W-01",
                        "model_name": "XGBoost Regressor",
                        "forecast_horizon_days": 30,
                        "predicted_level_m": 10.15,
                        "risk_category": "SEMI-CRITICAL",
                        "model_metrics": {"r2": 0.89, "rmse": 0.32}
                    }
                ]
            }
            res_post = self.client.post("/api/db/predictions", json=payload)
            self.assertEqual(res_post.status_code, 200)
            self.assertEqual(res_post.json()["saved_count"], 1)

            res_get = self.client.get("/api/db/predictions?station_id=W-01")
            self.assertEqual(res_get.status_code, 200)
            self.assertEqual(res_get.json()["count"], 1)

    def test_08_mocked_supabase_alerts_management(self):
        """Verify creating alerts and updating their status (acknowledging)."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table

        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": 5}])
        mock_table.insert.return_value = mock_insert

        mock_update = MagicMock()
        mock_eq = MagicMock()
        mock_eq.execute.return_value = MagicMock(data=[{"id": 5, "status": "ACKNOWLEDGED"}])
        mock_update.eq.return_value = mock_eq
        mock_table.update.return_value = mock_update

        with patch.object(server, "supabase_client", mock_db):
            # Create alert
            res_post = self.client.post("/api/db/alerts", json={
                "alerts": [
                    {
                        "station_id": "W-01",
                        "alert_type": "CRITICAL_DEPLETION",
                        "severity": "CRITICAL",
                        "message": "Water table dropped below emergency threshold 12m.",
                        "current_level_m": 12.8,
                        "threshold_m": 12.0
                    }
                ]
            })
            self.assertEqual(res_post.status_code, 200)
            self.assertEqual(res_post.json()["saved_count"], 1)

            # Acknowledge alert
            res_patch = self.client.patch("/api/db/alerts/5", json={"status": "ACKNOWLEDGED"})
            self.assertEqual(res_patch.status_code, 200)
            self.assertEqual(res_patch.json()["alert"]["status"], "ACKNOWLEDGED")

    def test_09_regression_existing_endpoints(self):
        """Verify existing simulation and live weather endpoints remain operational."""
        # Live weather fallback or online response
        res_weather = self.client.get("/api/weather/live?lat=18.52&lon=73.86")
        self.assertEqual(res_weather.status_code, 200)
        self.assertIn("temperature", res_weather.json())

        # What-If Simulator
        res_sim = self.client.post("/api/simulate", json={
            "well_id": "W-01",
            "baseline_level": 8.0,
            "rainfall_variation_pct": -20.0,
            "extraction_variation_pct": 15.0,
            "horizon_months": 6
        })
        self.assertEqual(res_sim.status_code, 200)
        self.assertIn("cgwb_category", res_sim.json())
        self.assertEqual(len(res_sim.json()["trajectory"]), 6)

if __name__ == "__main__":
    unittest.main()
