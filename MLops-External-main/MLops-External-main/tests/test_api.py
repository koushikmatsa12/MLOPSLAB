"""Tests for Task 4: FastAPI Model Serving and API-level Capacity Validation."""

from fastapi.testclient import TestClient

from api.main import app
from src.config import PRODUCTION_MODEL_PATH
from src.data_generation import create_and_version_datasets
from src.pipeline import run_data_pipeline
from src.train import train_and_register_models

# Ensure model artifact exists when running tests standalone in CI
if not PRODUCTION_MODEL_PATH.exists():
    create_and_version_datasets()
    run_data_pipeline()
    train_and_register_models()

client = TestClient(app)


def test_health_endpoint():
    """Verify /health confirms model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["tank_capacity_litres"] == 100000.0


def test_predict_shortage_risk_scenario():
    """High summer evening demand with low tank level should predict Shortage Risk."""
    payload = {
        "tank_level_8pm": 32000.0,
        "connected_households": 150,
        "avg_hourly_consumption": 54.0,
        "expected_pumping_duration": 0.8,
        "expected_pumping_rate": 8000.0,
        "leakage_estimate": 3500.0,
        "day_type": "Weekend",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Shortage Risk"
    assert 0.0 <= data["shortage_probability"] <= 1.0
    assert data["engineered_features"]["projected_water_buffer"] < 0


def test_predict_adequate_reserve_scenario():
    """High tank level with moderate demand and strong pumping should predict Adequate Reserve."""
    payload = {
        "tank_level_8pm": 88000.0,
        "connected_households": 95,
        "avg_hourly_consumption": 30.0,
        "expected_pumping_duration": 2.8,
        "expected_pumping_rate": 12500.0,
        "leakage_estimate": 900.0,
        "day_type": "Weekday",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Adequate Reserve"
    assert 0.0 <= data["shortage_probability"] <= 1.0
    assert data["engineered_features"]["projected_water_buffer"] > 0


def test_api_rejects_tank_level_exceeding_capacity():
    """API must reject tank_level_8pm > 100,000 litres with HTTP 422."""
    payload = {
        "tank_level_8pm": 105000.0,
        "connected_households": 120,
        "avg_hourly_consumption": 38.0,
        "expected_pumping_duration": 2.0,
        "expected_pumping_rate": 10000.0,
        "leakage_estimate": 1500.0,
        "day_type": "Weekday",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_rejects_negative_tank_level():
    """API must reject negative tank_level_8pm with HTTP 422."""
    payload = {
        "tank_level_8pm": -100.0,
        "connected_households": 120,
        "avg_hourly_consumption": 38.0,
        "expected_pumping_duration": 2.0,
        "expected_pumping_rate": 10000.0,
        "leakage_estimate": 1500.0,
        "day_type": "Weekday",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

