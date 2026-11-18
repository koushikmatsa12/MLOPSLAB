"""Central configuration constants for the Water Tank Shortage MLOps system."""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_VERSIONS_FILE = DATA_DIR / "dataset_versions.json"

FEATURE_STORE_DIR = ROOT_DIR / "feature_store"
FEATURE_METADATA_FILE = FEATURE_STORE_DIR / "feature_metadata.json"

MODELS_DIR = ROOT_DIR / "models"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
REPORT_PDF_PATH = DOCS_DIR / "Water_Shortage_MLOps_Report.pdf"

# Domain constants
TANK_CAPACITY_LITRES = 100_000.0
CRITICAL_THRESHOLD_RATIO = 0.15
CRITICAL_THRESHOLD_LITRES = TANK_CAPACITY_LITRES * CRITICAL_THRESHOLD_RATIO  # 15,000 L
OVERNIGHT_HOURS = 10.0  # 8:00 PM to 6:00 AM

VALID_DAY_TYPES = ("Weekday", "Weekend", "Holiday")

# Dataset generation settings
RECORDS_PER_VERSION = 250
SUMMER_DEMAND_MULTIPLIER = 1.35  # ~35% higher household demand in Version 2
RANDOM_SEED_V1 = 42
RANDOM_SEED_V2 = 108

# MLflow settings
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT_NAME = "Overnight_Water_Shortage_Prediction"
MLFLOW_MODEL_NAME = "WaterShortagePredictor"

# Feature columns
RAW_FEATURE_COLUMNS = [
    "tank_level_8pm",
    "connected_households",
    "avg_hourly_consumption",
    "expected_pumping_duration",
    "expected_pumping_rate",
    "leakage_estimate",
    "day_type",
]

ENGINEERED_FEATURE_COLUMNS = [
    "expected_overnight_demand",
    "expected_pump_supply",
    "projected_water_buffer",
]

MODEL_NUMERIC_FEATURES = [
    "tank_level_8pm",
    "connected_households",
    "avg_hourly_consumption",
    "expected_pumping_duration",
    "expected_pumping_rate",
    "leakage_estimate",
    "expected_overnight_demand",
    "expected_pump_supply",
    "projected_water_buffer",
]

MODEL_CATEGORICAL_FEATURES = ["day_type"]
TARGET_COLUMN = "morning_water_shortage"

