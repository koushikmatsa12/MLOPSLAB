"""Domain validation and reusable feature engineering logic.

Shared across offline training pipeline and online FastAPI serving to eliminate
training-serving skew.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    CRITICAL_THRESHOLD_LITRES,
    OVERNIGHT_HOURS,
    TANK_CAPACITY_LITRES,
    VALID_DAY_TYPES,
)


class TankDataValidationError(ValueError):
    """Raised when input water tank records violate physical or schema constraints."""


def validate_tank_records(df: pd.DataFrame) -> None:
    """Validate physical and schema invariants on tank records.

    Raises
    ------
    TankDataValidationError
        If any record violates physical constraints (e.g., tank_level_8pm > 100,000 L).
    """
    required_cols = {
        "tank_level_8pm",
        "connected_households",
        "avg_hourly_consumption",
        "expected_pumping_duration",
        "expected_pumping_rate",
        "leakage_estimate",
        "day_type",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise TankDataValidationError(f"Missing required columns: {sorted(missing_cols)}")

    # Strict capacity validation: tank level cannot exceed 100,000 L or be negative
    tank_levels = df["tank_level_8pm"].dropna()
    if (tank_levels > TANK_CAPACITY_LITRES).any():
        max_val = float(tank_levels.max())
        raise TankDataValidationError(
            f"tank_level_8pm cannot exceed tank capacity of {TANK_CAPACITY_LITRES:,.0f} litres "
            f"(found {max_val:,.2f} litres)."
        )
    if (tank_levels < 0).any():
        min_val = float(tank_levels.min())
        raise TankDataValidationError(
            f"tank_level_8pm cannot be negative (found {min_val:,.2f} litres)."
        )

    # Households must be strictly positive
    households = df["connected_households"].dropna()
    if (households <= 0).any():
        raise TankDataValidationError("connected_households must be greater than 0.")

    # Consumption, pumping duration, pumping rate, and leakage must be non-negative
    for col in [
        "avg_hourly_consumption",
        "expected_pumping_duration",
        "expected_pumping_rate",
        "leakage_estimate",
    ]:
        vals = df[col].dropna()
        if (vals < 0).any():
            raise TankDataValidationError(f"{col} cannot be negative.")

    # Pumping duration cannot exceed the 10-hour overnight window
    pump_dur = df["expected_pumping_duration"].dropna()
    if (pump_dur > OVERNIGHT_HOURS).any():
        raise TankDataValidationError(
            f"expected_pumping_duration cannot exceed {OVERNIGHT_HOURS:.0f} overnight hours."
        )

    # Day type check (ignoring NaN before imputation)
    invalid_days = set(df["day_type"].dropna().unique()) - set(VALID_DAY_TYPES)
    if invalid_days:
        raise TankDataValidationError(
            f"Invalid day_type values: {sorted(invalid_days)}. Expected one of {VALID_DAY_TYPES}."
        )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three domain-engineered features for overnight water shortage prediction.

    Engineered Features:
    - expected_overnight_demand:
        Total expected water loss over the 10-hour night (8:00 PM to 6:00 AM):
        connected_households * avg_hourly_consumption * 10 + leakage_estimate
    - expected_pump_supply:
        Total expected water replenished by scheduled pumping:
        expected_pumping_duration * expected_pumping_rate
    - projected_water_buffer:
        Projected surplus (or deficit if negative) above the 15% critical reserve (15,000 L):
        (tank_level_8pm + expected_pump_supply - expected_overnight_demand) - 15,000
    """
    out = df.copy()
    out["expected_overnight_demand"] = (
        out["connected_households"] * out["avg_hourly_consumption"] * OVERNIGHT_HOURS
        + out["leakage_estimate"]
    ).round(2)

    out["expected_pump_supply"] = (
        out["expected_pumping_duration"] * out["expected_pumping_rate"]
    ).round(2)

    out["projected_water_buffer"] = (
        out["tank_level_8pm"]
        + out["expected_pump_supply"]
        - out["expected_overnight_demand"]
        - CRITICAL_THRESHOLD_LITRES
    ).round(2)

    return out

