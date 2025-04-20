# infrastructure/external_services/data_quality.py
"""
Data quality assessment module for F1 data.

This module provides functionality to assess the quality of F1 data from different sources.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("f1_prediction.external_services.data_quality")

@dataclass
class F1DataQuality:
    """Data structure containing quality metrics for F1 data."""
    complete: bool
    reliability_score: float  # 0-10 scale
    driver_coverage: float    # Percentage of expected drivers
    missing_features: List[str]
    consistency_issues: List[str]

def assess_data_quality(race_data: pd.DataFrame, quali_data: pd.DataFrame) -> F1DataQuality:
    """
    Assess the quality of F1 data.
    
    Args:
        race_data: Race data to assess
        quali_data: Qualifying data to assess
        
    Returns:
        F1DataQuality object with quality metrics
    """
    quality = F1DataQuality(
        complete=True,
        reliability_score=10.0,  # Start with perfect score
        driver_coverage=0.0,
        missing_features=[],
        consistency_issues=[]
    )
    
    # Check if data exists
    if race_data.empty:
        quality.complete = False
        quality.reliability_score -= 5.0
        quality.missing_features.append("race_data")
    
    if quali_data.empty:
        quality.complete = False
        quality.reliability_score -= 2.5
        quality.missing_features.append("quali_data")
    
    # If both are missing, return minimum quality
    if race_data.empty and quali_data.empty:
        quality.reliability_score = 0.0
        return quality
    
    # Check essential race data columns
    essential_race_columns = ['DriverNumber', 'Driver', 'TeamName', 'RacePosition']
    for col in essential_race_columns:
        if col not in race_data.columns:
            quality.complete = False
            quality.missing_features.append(col)
            quality.reliability_score -= 2.0
    
    # Check essential qualifying data columns
    essential_quali_columns = ['DriverNumber', 'Driver', 'TeamName', 'GridPosition']
    for col in essential_quali_columns:
        if col not in quali_data.columns:
            quality.complete = False
            quality.missing_features.append(col)
            quality.reliability_score -= 1.0
    
    # Check for telemetry-based features
    telemetry_features = ['FastestLap_sec', 'AverageLap_sec', 'Interval_sec']
    for col in telemetry_features:
        if col not in race_data.columns or race_data[col].isna().all():
            quality.missing_features.append(col)
            quality.reliability_score -= 0.5
    
    # Check driver coverage (standard F1 grid = 20 drivers)
    expected_drivers = 20
    actual_drivers = len(race_data) if not race_data.empty else 0
    quality.driver_coverage = (actual_drivers / expected_drivers) * 100
    
    # Penalize for low driver coverage
    if quality.driver_coverage < 80:
        quality.reliability_score -= (80 - quality.driver_coverage) / 10
    
    # Check for consistency between qualifying and race data
    if not race_data.empty and not quali_data.empty:
        # Check if all drivers in qualifying are in race data
        quali_drivers = set(quali_data['DriverNumber'].astype(str))
        race_drivers = set(race_data['DriverNumber'].astype(str))
        
        missing_drivers = quali_drivers - race_drivers
        if missing_drivers:
            quality.consistency_issues.append(f"Missing {len(missing_drivers)} drivers in race data")
            quality.reliability_score -= len(missing_drivers) * 0.2
    
    # Clamp reliability score to 0-10 range
    quality.reliability_score = max(0.0, min(10.0, quality.reliability_score))
    
    return quality