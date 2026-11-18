"""Validation tests including strict tank capacity constraints (<= 100,000 litres)."""

import pandas as pd
import pytest

from src.config import TANK_CAPACITY_LITRES
from src.feature_engineering import TankDataValidationError, validate_tank_records


def _sample_valid_record() -> dict:
    return {
        "tank_level_8pm": 65000.0,
        "connected_households": 120,
        "avg_hourly_consumption": 38.0,
        "expected_pumping_duration": 2.0,
        "expected_pumping_rate": 10000.0,
        "leakage_estimate": 1500.0,
        "day_type": "Weekday",
    }


def test_tank_level_cannot_exceed_capacity():
    """Verify tank_level_8pm above 100,000 L raises TankDataValidationError."""
    record = _sample_valid_record()
    record["tank_level_8pm"] = TANK_CAPACITY_LITRES + 1.0  # 100,001 L
    df = pd.DataFrame([record])

    with pytest.raises(TankDataValidationError, match="cannot exceed tank capacity"):
        validate_tank_records(df)


def test_tank_level_exact_capacity_is_allowed():
    """Verify tank_level_8pm equal to exact capacity (100,000 L) is valid."""
    record = _sample_valid_record()
    record["tank_level_8pm"] = TANK_CAPACITY_LITRES
    df = pd.DataFrame([record])
    validate_tank_records(df)


def test_tank_level_cannot_be_negative():
    """Verify negative tank_level_8pm is rejected."""
    record = _sample_valid_record()
    record["tank_level_8pm"] = -500.0
    df = pd.DataFrame([record])

    with pytest.raises(TankDataValidationError, match="cannot be negative"):
        validate_tank_records(df)


def test_invalid_day_type_rejected():
    """Verify unexpected day_type categories are rejected."""
    record = _sample_valid_record()
    record["day_type"] = "MonsoonDay"
    df = pd.DataFrame([record])

    with pytest.raises(TankDataValidationError, match="Invalid day_type"):
        validate_tank_records(df)


def test_pumping_duration_cannot_exceed_overnight_window():
    """Verify pumping duration > 10 overnight hours is rejected."""
    record = _sample_valid_record()
    record["expected_pumping_duration"] = 12.0
    df = pd.DataFrame([record])

    with pytest.raises(TankDataValidationError, match="expected_pumping_duration"):
        validate_tank_records(df)

