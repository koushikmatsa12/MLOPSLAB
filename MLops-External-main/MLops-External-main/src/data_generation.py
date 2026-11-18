"""Task 1: Synthetic Dataset Creation and Version Management.

Generates 250 synthetic nightly records for:
- Version 1 (v1): Normal residential water demand conditions.
- Version 2 (v2): Summer conditions with ~35% higher household water demand.

Maintains a SHA-256 dataset version manifest in data/dataset_versions.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CRITICAL_THRESHOLD_LITRES,
    DATASET_VERSIONS_FILE,
    OVERNIGHT_HOURS,
    RANDOM_SEED_V1,
    RANDOM_SEED_V2,
    RAW_DATA_DIR,
    RECORDS_PER_VERSION,
    SUMMER_DEMAND_MULTIPLIER,
    TANK_CAPACITY_LITRES,
    VALID_DAY_TYPES,
)


def compute_sha256(file_path: Path) -> str:
    """Return hex SHA-256 digest of a file for reproducible dataset versioning."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_nightly_dataset(
    n_records: int = RECORDS_PER_VERSION,
    demand_multiplier: float = 1.0,
    seed: int = RANDOM_SEED_V1,
    version_label: str = "v1",
    start_date: str = "2025-01-01",
    inject_missing: bool = True,
) -> pd.DataFrame:
    """Generate realistic nightly overhead tank records at 8:00 PM and 6:00 AM shortage label.

    Parameters
    ----------
    n_records : int
        Number of nightly records to synthesize (default 250).
    demand_multiplier : float
        Multiplier applied to base household hourly consumption (1.0 for v1 normal,
        1.35 for v2 summer conditions).
    seed : int
        Random seed for reproducibility.
    version_label : str
        Dataset version tag ('v1' or 'v2').
    start_date : str
        Starting date (YYYY-MM-DD) for the synthetic nightly series.
    inject_missing : bool
        If True, introduces a small (~2%) realistic missingness rate in non-target fields
        so the automated preprocessing pipeline demonstrates imputation.
    """
    rng = np.random.default_rng(seed)

    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [(base_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_records)]

    # Day type distribution (~68% Weekday, ~24% Weekend, ~8% Holiday)
    day_types = rng.choice(VALID_DAY_TYPES, size=n_records, p=[0.68, 0.24, 0.08])

    # Overhead tank level at 8:00 PM (litres), strictly within [0, 100,000]
    tank_level_8pm = np.round(
        np.clip(rng.normal(loc=56_000, scale=17_500, size=n_records), 18_000, 96_000),
        1,
    )

    # Number of active connected households in the residential colony
    connected_households = rng.integers(90, 161, size=n_records)

    # Base hourly consumption per household (litres/household/hour)
    # Weekend and Holiday nights typically see higher evening/early-morning usage
    day_factor_map = {"Weekday": 1.00, "Weekend": 1.12, "Holiday": 1.18}
    day_multipliers = np.array([day_factor_map[d] for d in day_types])

    base_hourly = rng.normal(loc=36.0, scale=5.5, size=n_records)
    base_hourly = np.clip(base_hourly, 22.0, 55.0)
    avg_hourly_consumption = np.round(base_hourly * day_multipliers * demand_multiplier, 2)

    # Scheduled municipal pumping duration (hours) and pumping rate (litres/hour)
    expected_pumping_duration = np.round(rng.uniform(0.5, 3.2, size=n_records), 2)
    expected_pumping_rate = np.round(rng.uniform(7_000, 14_500, size=n_records), 1)

    # Estimated overnight leakage (litres over the 10-hour period)
    leakage_estimate = np.round(rng.uniform(600, 5_200, size=n_records), 1)

    # Simulate realistic overnight stochastic dynamics between 8:00 PM and 6:00 AM:
    # - Unexpected household demand surges (±14% variation)
    # - Occasional delayed pumping (actual delivered pump volume 72%-103% of expected)
    # - Actual leakage fluctuation (±18%)
    actual_demand_variation = rng.normal(loc=1.02, scale=0.13, size=n_records)
    actual_demand_variation = np.clip(actual_demand_variation, 0.72, 1.45)

    # ~18% of nights experience a notable pumping delay (reduced effective pumping)
    pump_delay_mask = rng.random(n_records) < 0.18
    pump_efficiency = np.where(
        pump_delay_mask,
        rng.uniform(0.45, 0.80, size=n_records),
        rng.uniform(0.92, 1.04, size=n_records),
    )

    actual_leakage = leakage_estimate * np.clip(rng.normal(1.04, 0.16, size=n_records), 0.70, 1.50)

    actual_overnight_consumption = (
        connected_households * avg_hourly_consumption * OVERNIGHT_HOURS * actual_demand_variation
    )
    actual_pump_supply = expected_pumping_duration * expected_pumping_rate * pump_efficiency

    # Remaining water at 6:00 AM cannot exceed tank capacity (overflow) or go below 0
    actual_tank_level_6am = np.clip(
        tank_level_8pm + actual_pump_supply - actual_overnight_consumption - actual_leakage,
        0.0,
        TANK_CAPACITY_LITRES,
    )

    # Target: 1 if 6:00 AM tank level falls below 15% of capacity (15,000 L), else 0
    morning_water_shortage = (actual_tank_level_6am < CRITICAL_THRESHOLD_LITRES).astype(int)

    df = pd.DataFrame(
        {
            "record_id": [f"{version_label.upper()}-{i + 1:04d}" for i in range(n_records)],
            "date": dates,
            "tank_level_8pm": tank_level_8pm,
            "connected_households": connected_households,
            "avg_hourly_consumption": avg_hourly_consumption,
            "expected_pumping_duration": expected_pumping_duration,
            "expected_pumping_rate": expected_pumping_rate,
            "leakage_estimate": leakage_estimate,
            "day_type": day_types,
            "morning_water_shortage": morning_water_shortage,
        }
    )

    # Introduce a small realistic fraction (~2%) of missing sensor/log entries in raw data
    # so the automated preprocessing pipeline actively cleans and imputes them.
    if inject_missing:
        missing_cols = ["avg_hourly_consumption", "leakage_estimate", "day_type"]
        for col in missing_cols:
            miss_idx = rng.choice(n_records, size=max(1, int(0.02 * n_records)), replace=False)
            df.loc[miss_idx, col] = np.nan

    return df


def create_and_version_datasets() -> dict:
    """Generate v1 (normal) and v2 (summer, +35% demand) datasets and write version manifest."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    v1_df = generate_nightly_dataset(
        n_records=RECORDS_PER_VERSION,
        demand_multiplier=1.0,
        seed=RANDOM_SEED_V1,
        version_label="v1",
        start_date="2025-01-01",
        inject_missing=True,
    )
    v2_df = generate_nightly_dataset(
        n_records=RECORDS_PER_VERSION,
        demand_multiplier=SUMMER_DEMAND_MULTIPLIER,
        seed=RANDOM_SEED_V2,
        version_label="v2",
        start_date="2025-04-15",
        inject_missing=True,
    )

    v1_path = RAW_DATA_DIR / "water_tank_v1.csv"
    v2_path = RAW_DATA_DIR / "water_tank_v2.csv"

    v1_df.to_csv(v1_path, index=False)
    v2_df.to_csv(v2_path, index=False)

    v1_mean_cons = float(v1_df["avg_hourly_consumption"].mean())
    v2_mean_cons = float(v2_df["avg_hourly_consumption"].mean())
    demand_increase_pct = ((v2_mean_cons - v1_mean_cons) / v1_mean_cons) * 100.0

    manifest = {
        "tank_capacity_litres": TANK_CAPACITY_LITRES,
        "critical_threshold_litres": CRITICAL_THRESHOLD_LITRES,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "v1": {
                "file": str(v1_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix()),
                "description": "Baseline normal-season residential overnight water records",
                "record_count": int(len(v1_df)),
                "sha256": compute_sha256(v1_path),
                "mean_avg_hourly_consumption": round(v1_mean_cons, 2),
                "mean_tank_level_8pm": round(float(v1_df["tank_level_8pm"].mean()), 2),
                "shortage_rate": round(float(v1_df["morning_water_shortage"].mean()), 4),
                "missing_values_total": int(v1_df.isna().sum().sum()),
            },
            "v2": {
                "file": str(v2_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix()),
                "description": "Summer-season simulation with ~35% higher household water demand",
                "record_count": int(len(v2_df)),
                "sha256": compute_sha256(v2_path),
                "mean_avg_hourly_consumption": round(v2_mean_cons, 2),
                "mean_tank_level_8pm": round(float(v2_df["tank_level_8pm"].mean()), 2),
                "shortage_rate": round(float(v2_df["morning_water_shortage"].mean()), 4),
                "missing_values_total": int(v2_df.isna().sum().sum()),
            },
        },
        "version_comparison": {
            "v1_to_v2_demand_increase_pct": round(demand_increase_pct, 2),
            "v1_shortage_count": int(v1_df["morning_water_shortage"].sum()),
            "v2_shortage_count": int(v2_df["morning_water_shortage"].sum()),
        },
    }

    DATASET_VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_VERSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    summary = create_and_version_datasets()
    print("Dataset generation and versioning complete:")
    print(json.dumps(summary, indent=2))

