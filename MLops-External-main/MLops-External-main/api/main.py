"""Task 4: FastAPI Model Serving for Overnight Water Shortage Prediction.

Exposes:
- GET  /         : Service overview and quick links
- GET  /health   : Model readiness & metadata check
- POST /predict  : Predicts 'Shortage Risk' or 'Adequate Reserve' from 8:00 PM conditions
- GET  /ui       : Interactive browser dashboard for live evening scenario predictions
"""

from __future__ import annotations

import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import (
    CRITICAL_THRESHOLD_LITRES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_NUMERIC_FEATURES,
    OVERNIGHT_HOURS,
    PRODUCTION_MODEL_PATH,
    TANK_CAPACITY_LITRES,
)
from src.feature_engineering import engineer_features

app = FastAPI(
    title="Overnight Residential Water Shortage Prediction API",
    description=(
        "MLOps inference service predicting whether a 100,000-litre overhead residential "
        "water tank will fall below 15% capacity (15,000 L) by 6:00 AM based on 8:00 PM conditions."
    ),
    version="1.0.0",
)

_MODEL = None
_MODEL_META: dict = {}


def load_production_model():
    """Load the registered production model artifact lazily."""
    global _MODEL, _MODEL_META
    if _MODEL is None:
        if not PRODUCTION_MODEL_PATH.exists():
            raise RuntimeError(
                f"Production model not found at {PRODUCTION_MODEL_PATH}. "
                "Run `python -m src.train` first."
            )
        _MODEL = joblib.load(PRODUCTION_MODEL_PATH)
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                _MODEL_META = json.load(f)
    return _MODEL, _MODEL_META


class EveningTankConditions(BaseModel):
    """Input schema for 8:00 PM overhead tank and colony operating conditions."""

    tank_level_8pm: float = Field(
        ...,
        ge=0.0,
        le=TANK_CAPACITY_LITRES,
        description="Tank water level at 8:00 PM in litres (must be between 0 and 100,000 L).",
        examples=[42000.0],
    )
    connected_households: int = Field(
        ...,
        gt=0,
        le=1000,
        description="Number of active residential households connected to the tank.",
        examples=[140],
    )
    avg_hourly_consumption: float = Field(
        ...,
        gt=0.0,
        le=500.0,
        description="Average hourly consumption per household (litres/household/hour).",
        examples=[48.5],
    )
    expected_pumping_duration: float = Field(
        ...,
        ge=0.0,
        le=OVERNIGHT_HOURS,
        description="Scheduled overnight municipal pumping duration in hours (0 to 10).",
        examples=[1.2],
    )
    expected_pumping_rate: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Expected municipal pump inflow rate in litres/hour.",
        examples=[9500.0],
    )
    leakage_estimate: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Estimated overnight water loss due to leakage (litres).",
        examples=[2800.0],
    )
    day_type: Literal["Weekday", "Weekend", "Holiday"] = Field(
        ...,
        description="Day category affecting overnight household demand.",
        examples=["Weekend"],
    )


class EngineeredFeaturesOut(BaseModel):
    expected_overnight_demand: float
    expected_pump_supply: float
    projected_water_buffer: float


class PredictionResponse(BaseModel):
    prediction: Literal["Shortage Risk", "Adequate Reserve"]
    shortage_probability: float
    engineered_features: EngineeredFeaturesOut
    tank_capacity_litres: float
    critical_threshold_litres: float
    model_name: str
    model_version: str


@app.get("/")
def root() -> dict:
    """Service summary endpoint."""
    return {
        "service": "Overnight Residential Water Shortage Prediction API",
        "tank_capacity_litres": TANK_CAPACITY_LITRES,
        "critical_threshold_litres": CRITICAL_THRESHOLD_LITRES,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "swagger_docs": "/docs",
            "interactive_ui": "/ui",
        },
    }


@app.get("/health")
def health() -> dict:
    """Health check confirming the registered production model is loaded."""
    try:
        _, meta = load_production_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "healthy",
        "selected_model": meta.get("selected_model", "Unknown"),
        "mlflow_registered_model": meta.get("mlflow_registered_model", "WaterShortagePredictor"),
        "mlflow_model_version": meta.get("mlflow_model_version", "1"),
        "tank_capacity_litres": TANK_CAPACITY_LITRES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_shortage(payload: EveningTankConditions) -> PredictionResponse:
    """Predict whether the overhead tank faces an overnight Shortage Risk or Adequate Reserve."""
    try:
        model, meta = load_production_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw_df = pd.DataFrame([payload.model_dump()])
    feat_df = engineer_features(raw_df)

    model_input = feat_df[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]
    pred_class = int(model.predict(model_input)[0])
    shortage_prob = float(model.predict_proba(model_input)[0, 1])

    label: Literal["Shortage Risk", "Adequate Reserve"] = (
        "Shortage Risk" if pred_class == 1 else "Adequate Reserve"
    )

    return PredictionResponse(
        prediction=label,
        shortage_probability=round(shortage_prob, 4),
        engineered_features=EngineeredFeaturesOut(
            expected_overnight_demand=float(feat_df["expected_overnight_demand"].iloc[0]),
            expected_pump_supply=float(feat_df["expected_pump_supply"].iloc[0]),
            projected_water_buffer=float(feat_df["projected_water_buffer"].iloc[0]),
        ),
        tank_capacity_litres=TANK_CAPACITY_LITRES,
        critical_threshold_litres=CRITICAL_THRESHOLD_LITRES,
        model_name=meta.get("selected_model", "RandomForest"),
        model_version=str(meta.get("mlflow_model_version", "1")),
    )


@app.get("/ui", response_class=HTMLResponse)
def interactive_ui() -> str:
    """Clean browser UI for testing live 8:00 PM scenarios and viewing API JSON responses."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Residential Water Tank Shortage Predictor</title>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 28px; }
  .container { max-width: 960px; margin: 0 auto; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }
  h1 { margin: 0; font-size: 1.55rem; color: #38bdf8; }
  .badge { background: #1e293b; border: 1px solid #38bdf8; color: #7dd3fc; padding: 6px 12px; border-radius: 999px; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
  .card h2 { margin-top: 0; font-size: 1.1rem; color: #f8fafc; }
  label { display: block; font-size: 0.85rem; color: #94a3b8; margin-top: 10px; }
  input, select { width: 100%; box-sizing: border-box; margin-top: 4px; padding: 8px 10px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; font-size: 0.95rem; }
  .btn-row { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  button { padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.9rem; }
  .btn-primary { background: #0284c7; color: white; flex: 1; }
  .btn-danger { background: #dc2626; color: white; }
  .btn-safe { background: #059669; color: white; }
  .result-box { padding: 16px; border-radius: 10px; margin-bottom: 14px; font-size: 1.25rem; font-weight: 700; text-align: center; }
  .risk { background: rgba(220, 38, 38, 0.2); border: 2px solid #ef4444; color: #fca5a5; }
  .safe { background: rgba(5, 150, 105, 0.2); border: 2px solid #10b981; color: #6ee7b7; }
  .neutral { background: #0f172a; border: 1px dashed #475569; color: #94a3b8; }
  pre { background: #0f172a; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; color: #a5f3fc; border: 1px solid #334155; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Overnight Water Shortage Predictor (8:00 PM &rarr; 6:00 AM)</h1>
      <div style="color:#94a3b8;font-size:0.9rem;margin-top:4px;">Capacity: 100,000 L | Critical 6:00 AM Reserve Threshold: 15% (15,000 L)</div>
    </div>
    <span class="badge">FastAPI + MLflow Registry</span>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Evening Conditions (8:00 PM)</h2>
      <label>Tank Level at 8:00 PM (Litres, max 100,000)</label>
      <input id="tank_level_8pm" type="number" value="38000" step="500" />
      <label>Connected Households</label>
      <input id="connected_households" type="number" value="145" />
      <label>Average Hourly Consumption (L/household/hr)</label>
      <input id="avg_hourly_consumption" type="number" value="51.5" step="0.5" />
      <label>Expected Pumping Duration (Hours)</label>
      <input id="expected_pumping_duration" type="number" value="1.1" step="0.1" />
      <label>Expected Pumping Rate (Litres/hr)</label>
      <input id="expected_pumping_rate" type="number" value="9000" step="250" />
      <label>Leakage Estimate (Litres overnight)</label>
      <input id="leakage_estimate" type="number" value="3200" step="100" />
      <label>Day Type</label>
      <select id="day_type">
        <option value="Weekday">Weekday</option>
        <option value="Weekend" selected>Weekend</option>
        <option value="Holiday">Holiday</option>
      </select>
      <div class="btn-row">
        <button class="btn-primary" onclick="runPredict()">Predict 6:00 AM Status</button>
        <button class="btn-danger" onclick="loadSummerRisk()">Summer Risk Preset</button>
        <button class="btn-safe" onclick="loadSafePreset()">Adequate Reserve Preset</button>
      </div>
    </div>
    <div class="card">
      <h2>Prediction Output</h2>
      <div id="statusBox" class="result-box neutral">Awaiting Prediction</div>
      <h3 style="font-size:0.95rem;color:#cbd5e1;margin-bottom:6px;">FastAPI JSON Response (/predict)</h3>
      <pre id="jsonOutput">{ "status": "Click Predict to invoke POST /predict" }</pre>
    </div>
  </div>
</div>
<script>
function loadSummerRisk() {
  document.getElementById('tank_level_8pm').value = 38000;
  document.getElementById('connected_households').value = 145;
  document.getElementById('avg_hourly_consumption').value = 51.5;
  document.getElementById('expected_pumping_duration').value = 1.1;
  document.getElementById('expected_pumping_rate').value = 9000;
  document.getElementById('leakage_estimate').value = 3200;
  document.getElementById('day_type').value = 'Weekend';
  runPredict();
}
function loadSafePreset() {
  document.getElementById('tank_level_8pm').value = 84000;
  document.getElementById('connected_households').value = 105;
  document.getElementById('avg_hourly_consumption').value = 33.0;
  document.getElementById('expected_pumping_duration').value = 2.6;
  document.getElementById('expected_pumping_rate').value = 12000;
  document.getElementById('leakage_estimate').value = 1100;
  document.getElementById('day_type').value = 'Weekday';
  runPredict();
}
async function runPredict() {
  const payload = {
    tank_level_8pm: parseFloat(document.getElementById('tank_level_8pm').value),
    connected_households: parseInt(document.getElementById('connected_households').value, 10),
    avg_hourly_consumption: parseFloat(document.getElementById('avg_hourly_consumption').value),
    expected_pumping_duration: parseFloat(document.getElementById('expected_pumping_duration').value),
    expected_pumping_rate: parseFloat(document.getElementById('expected_pumping_rate').value),
    leakage_estimate: parseFloat(document.getElementById('leakage_estimate').value),
    day_type: document.getElementById('day_type').value
  };
  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
  const box = document.getElementById('statusBox');
  if (!resp.ok) {
    box.className = 'result-box risk';
    box.textContent = 'Validation Error (HTTP ' + resp.status + ')';
  } else if (data.prediction === 'Shortage Risk') {
    box.className = 'result-box risk';
    box.textContent = 'SHORTAGE RISK (Prob: ' + (data.shortage_probability * 100).toFixed(1) + '%)';
  } else {
    box.className = 'result-box safe';
    box.textContent = 'ADEQUATE RESERVE (Shortage Prob: ' + (data.shortage_probability * 100).toFixed(1) + '%)';
  }
}
</script>
</body>
</html>"""

