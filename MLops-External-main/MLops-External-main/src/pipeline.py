"""Task 2: Automated Data Preprocessing Pipeline and Reusable Feature Store.

Reads raw versioned datasets (v1, v2), validates physical constraints, imputes missing values,
engineers domain features (expected_overnight_demand, expected_pump_supply, projected_water_buffer),
and persists engineered feature tables + schema metadata into the Feature Store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CRITICAL_THRESHOLD_LITRES,
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_METADATA_FILE,
    FEATURE_STORE_DIR,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    OVERNIGHT_HOURS,
    RAW_DATA_DIR,
    TANK_CAPACITY_LITRES,
    TARGET_COLUMN,
)
from src.feature_engineering import engineer_features, validate_tank_records


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run automated validation, missing-value imputation, type coercion, and feature engineering."""
    validate_tank_records(df)

    cleaned = df.copy()

    # Numeric imputation using column medians
    numeric_raw_cols = [
        "tank_level_8pm",
        "connected_households",
        "avg_hourly_consumption",
        "expected_pumping_duration",
        "expected_pumping_rate",
        "leakage_estimate",
    ]
    for col in numeric_raw_cols:
        if cleaned[col].isna().any():
            median_val = float(cleaned[col].median())
            cleaned[col] = cleaned[col].fillna(median_val)

    # Categorical imputation using column mode
    if cleaned["day_type"].isna().any():
        mode_val = str(cleaned["day_type"].mode().iloc[0])
        cleaned["day_type"] = cleaned["day_type"].fillna(mode_val)

    cleaned["connected_households"] = cleaned["connected_households"].round().astype(int)
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    # Engineer domain features
    engineered = engineer_features(cleaned)

    # Re-validate after preprocessing
    validate_tank_records(engineered)
    return engineered


def save_to_feature_store(df: pd.DataFrame, version: str) -> Path:
    """Persist engineered features for a given dataset version inside the Feature Store."""
    version_dir = FEATURE_STORE_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    out_path = version_dir / "engineered_features.csv"
    df.to_csv(out_path, index=False)
    return out_path


def load_from_feature_store(version: str = "combined") -> pd.DataFrame:
    """Load reusable engineered features from the Feature Store."""
    feature_path = FEATURE_STORE_DIR / version / "engineered_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature store table for version '{version}' not found at {feature_path}. "
            "Run `python -m src.pipeline` first."
        )
    return pd.read_csv(feature_path)


def run_data_pipeline() -> dict:
    """Execute end-to-end preprocessing and materialize v1, v2, and combined feature store tables."""
    v1_raw_path = RAW_DATA_DIR / "water_tank_v1.csv"
    v2_raw_path = RAW_DATA_DIR / "water_tank_v2.csv"

    if not v1_raw_path.exists() or not v2_raw_path.exists():
        raise FileNotFoundError(
            "Raw versioned datasets not found. Run `python -m src.data_generation` first."
        )

    v1_raw = pd.read_csv(v1_raw_path)
    v2_raw = pd.read_csv(v2_raw_path)

    v1_feat = preprocess_dataframe(v1_raw)
    v1_feat["dataset_version"] = "v1"

    v2_feat = preprocess_dataframe(v2_raw)
    v2_feat["dataset_version"] = "v2"

    combined_feat = pd.concat([v1_feat, v2_feat], ignore_index=True)

    p_v1 = save_to_feature_store(v1_feat, "v1")
    p_v2 = save_to_feature_store(v2_feat, "v2")
    p_comb = save_to_feature_store(combined_feat, "combined")

    metadata = {
        "feature_store_version": "1.0",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tank_capacity_litres": TANK_CAPACITY_LITRES,
        "critical_threshold_litres": CRITICAL_THRESHOLD_LITRES,
        "overnight_hours": OVERNIGHT_HOURS,
        "engineered_features": {
            "expected_overnight_demand": (
                "connected_households * avg_hourly_consumption * 10 + leakage_estimate (litres)"
            ),
            "expected_pump_supply": (
                "expected_pumping_duration * expected_pumping_rate (litres)"
            ),
            "projected_water_buffer": (
                "tank_level_8pm + expected_pump_supply - expected_overnight_demand - 15000 (litres)"
            ),
        },
        "model_features": {
            "numeric": MODEL_NUMERIC_FEATURES,
            "categorical": MODEL_CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
        },
        "tables": {
            "v1": {
                "path": str(p_v1.relative_to(FEATURE_STORE_DIR.parent).as_posix()),
                "rows": int(len(v1_feat)),
                "missing_values_after_cleaning": int(v1_feat.isna().sum().sum()),
                "mean_expected_overnight_demand": round(
                    float(v1_feat["expected_overnight_demand"].mean()), 2
                ),
                "mean_projected_water_buffer": round(
                    float(v1_feat["projected_water_buffer"].mean()), 2
                ),
                "shortage_rate": round(float(v1_feat[TARGET_COLUMN].mean()), 4),
            },
            "v2": {
                "path": str(p_v2.relative_to(FEATURE_STORE_DIR.parent).as_posix()),
                "rows": int(len(v2_feat)),
                "missing_values_after_cleaning": int(v2_feat.isna().sum().sum()),
                "mean_expected_overnight_demand": round(
                    float(v2_feat["expected_overnight_demand"].mean()), 2
                ),
                "mean_projected_water_buffer": round(
                    float(v2_feat["projected_water_buffer"].mean()), 2
                ),
                "shortage_rate": round(float(v2_feat[TARGET_COLUMN].mean()), 4),
            },
            "combined": {
                "path": str(p_comb.relative_to(FEATURE_STORE_DIR.parent).as_posix()),
                "rows": int(len(combined_feat)),
                "missing_values_after_cleaning": int(combined_feat.isna().sum().sum()),
                "mean_expected_overnight_demand": round(
                    float(combined_feat["expected_overnight_demand"].mean()), 2
                ),
                "mean_projected_water_buffer": round(
                    float(combined_feat["projected_water_buffer"].mean()), 2
                ),
                "shortage_rate": round(float(combined_feat[TARGET_COLUMN].mean()), 4),
            },
        },
    }

    FEATURE_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    meta = run_data_pipeline()
    print("Data pipeline and Feature Store materialization complete:")
    print(json.dumps(meta, indent=2))

