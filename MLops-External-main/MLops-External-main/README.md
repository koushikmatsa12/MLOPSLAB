# Predicting Overnight Water Shortages in a Residential Water Tank (MLOps System)

End-to-end MLOps pipeline predicting whether a **100,000-litre** residential overhead storage tank will fall below **15% capacity (15,000 litres)** by **6:00 AM**, given operating conditions recorded at **8:00 PM**.

## Project Architecture & Tasks Covered

1. **Dataset Creation & Versioning (`src/data_generation.py`)**
   - Generates **250 synthetic nightly records** for **Version 1 (`water_tank_v1.csv`)** under normal demand conditions.
   - Generates **250 synthetic nightly records** for **Version 2 (`water_tank_v2.csv`)** simulating **summer conditions (~35% higher household demand)**.
   - Tracks version metadata, SHA-256 checksums, demand shift statistics, and shortage rates in `data/dataset_versions.json`.

2. **Automated Data Pipeline & Reusable Feature Store (`src/pipeline.py`, `src/feature_engineering.py`)**
   - Validates physical invariants (e.g., `0 <= tank_level_8pm <= 100,000 L`, valid `day_type`, non-negative consumption/pumping/leakage).
   - Imputes missing sensor/log values automatically (median for numeric, mode for categorical).
   - Engineers and persists reusable features in `feature_store/`:
     - `expected_overnight_demand = connected_households * avg_hourly_consumption * 10 + leakage_estimate`
     - `expected_pump_supply = expected_pumping_duration * expected_pumping_rate`
     - `projected_water_buffer = tank_level_8pm + expected_pump_supply - expected_overnight_demand - 15000`

3. **Model Training, Comparison & MLflow Model Registry (`src/train.py`)**
   - Trains **Decision Tree** and **Random Forest** classifiers using Feature Store tables.
   - Compares **Precision, Recall, F1-score, Accuracy, ROC-AUC**, and **Confusion Matrices**.
   - Selects the model that better identifies possible water shortages (prioritizing **Recall** on the shortage class to minimize missed shortages / False Negatives) and registers it in **MLflow Model Registry** (`WaterShortagePredictor`, alias `@production`).

4. **CI/CD & FastAPI Model Serving (`api/main.py`, `tests/`, `.github/workflows/ci.yml`)**
   - Automated **GitHub Actions** pipeline runs data generation, preprocessing, model training, and `pytest` validation suite on every push / PR.
   - Strict validation tests enforce that **tank levels cannot exceed 100,000 L capacity** in both the data pipeline and the FastAPI endpoint (`HTTP 422`).
   - **FastAPI** serves predictions at `POST /predict`, returning **`"Shortage Risk"`** or **`"Adequate Reserve"`**, plus an interactive dashboard at `/ui` and Swagger UI at `/docs`.

5. **Documentation & Report (`docs/Water_Shortage_MLOps_Report.pdf`)**
   - Complete PDF report with system architecture, dataset version management, preprocessing, feature engineering, model comparison, MLflow UI screenshots, GitHub Actions/pytest evidence, and live FastAPI prediction screenshots.

---

## Quickstart (Run Locally)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Task 1: Generate versioned datasets (v1 Normal, v2 Summer +35% demand)
python -m src.data_generation

# 3. Task 2: Run automated preprocessing & materialize Feature Store
python -m src.pipeline

# 4. Task 3: Train Decision Tree & Random Forest, compare & register winner in MLflow
python -m src.train

# 5. Task 4: Run automated tests (including tank capacity validation)
pytest -v

# 6. Launch MLflow Tracking & Model Registry UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 7. Launch FastAPI Model Serving API (in another terminal)
uvicorn api.main:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/docs (Swagger) or http://127.0.0.1:8000/ui (Interactive UI)
```

## Example API Request (`POST /predict`)

```json
{
  "tank_level_8pm": 38000,
  "connected_households": 145,
  "avg_hourly_consumption": 51.5,
  "expected_pumping_duration": 1.1,
  "expected_pumping_rate": 9000,
  "leakage_estimate": 3200,
  "day_type": "Weekend"
}
```

Example response:
```json
{
  "prediction": "Shortage Risk",
  "shortage_probability": 0.96,
  "engineered_features": {
    "expected_overnight_demand": 77875.0,
    "expected_pump_supply": 9900.0,
    "projected_water_buffer": -44975.0
  },
  "tank_capacity_litres": 100000.0,
  "critical_threshold_litres": 15000.0,
  "model_name": "RandomForest",
  "model_version": "1"
}
```

