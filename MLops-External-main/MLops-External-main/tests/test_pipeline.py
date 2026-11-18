"""Tests for Task 1 (Dataset Versioning) and Task 2 (Preprocessing & Feature Store)."""

import json

import pandas as pd

from src.config import (
    CRITICAL_THRESHOLD_LITRES,
    DATASET_VERSIONS_FILE,
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_METADATA_FILE,
    RAW_DATA_DIR,
    RECORDS_PER_VERSION,
    TANK_CAPACITY_LITRES,
)
from src.data_generation import create_and_version_datasets
from src.feature_engineering import engineer_features
from src.pipeline import load_from_feature_store, run_data_pipeline


def test_dataset_v1_and_v2_generation_and_summer_demand_shift():
    """Verify v1 and v2 each have 250 records, valid tank levels, and ~35% higher demand in v2."""
    manifest = create_and_version_datasets()

    assert DATASET_VERSIONS_FILE.exists()
    v1_df = pd.read_csv(RAW_DATA_DIR / "water_tank_v1.csv")
    v2_df = pd.read_csv(RAW_DATA_DIR / "water_tank_v2.csv")

    assert len(v1_df) == RECORDS_PER_VERSION == 250
    assert len(v2_df) == RECORDS_PER_VERSION == 250

    # Strict capacity invariant on all generated records
    assert (v1_df["tank_level_8pm"] <= TANK_CAPACITY_LITRES).all()
    assert (v2_df["tank_level_8pm"] <= TANK_CAPACITY_LITRES).all()
    assert (v1_df["tank_level_8pm"] >= 0).all()
    assert (v2_df["tank_level_8pm"] >= 0).all()

    # Version 2 simulates summer with ~35% higher household demand (allow reasonable sampling tolerance)
    pct_increase = manifest["version_comparison"]["v1_to_v2_demand_increase_pct"]
    assert 28.0 <= pct_increase <= 42.0, f"Expected ~35% demand increase, got {pct_increase}%"


def test_engineered_feature_formulas():
    """Verify exact mathematical definitions of the three engineered features."""
    df = pd.DataFrame(
        [
            {
                "tank_level_8pm": 60000.0,
                "connected_households": 100,
                "avg_hourly_consumption": 40.0,
                "expected_pumping_duration": 2.0,
                "expected_pumping_rate": 10000.0,
                "leakage_estimate": 2000.0,
                "day_type": "Weekday",
            }
        ]
    )
    out = engineer_features(df)

    # 100 * 40 * 10 + 2000 = 42,000
    assert out["expected_overnight_demand"].iloc[0] == 42000.0
    # 2.0 * 10000 = 20,000
    assert out["expected_pump_supply"].iloc[0] == 20000.0
    # (60000 + 20000 - 42000) - 15000 = 23,000
    assert out["projected_water_buffer"].iloc[0] == (60000.0 + 20000.0 - 42000.0 - CRITICAL_THRESHOLD_LITRES)


def test_pipeline_and_feature_store_persistence():
    """Verify automated pipeline cleans missing values and saves reusable features."""
    if not (RAW_DATA_DIR / "water_tank_v1.csv").exists():
        create_and_version_datasets()

    meta = run_data_pipeline()
    assert FEATURE_METADATA_FILE.exists()

    with open(FEATURE_METADATA_FILE, "r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert saved_meta["tables"]["v1"]["rows"] == 250
    assert saved_meta["tables"]["v2"]["rows"] == 250
    assert saved_meta["tables"]["combined"]["rows"] == 500
    assert meta["tables"]["combined"]["missing_values_after_cleaning"] == 0

    for version in ("v1", "v2", "combined"):
        fs_df = load_from_feature_store(version)
        for col in ENGINEERED_FEATURE_COLUMNS:
            assert col in fs_df.columns
        assert fs_df.isna().sum().sum() == 0

