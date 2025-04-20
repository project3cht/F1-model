"""
Data processing module for F1 race predictions.

This module provides functions to process and prepare F1 data for modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_driver_stats(historical_data):
    """
    Calculate driver statistics from historical data.
    
    Args:
        historical_data (DataFrame): Historical race results
        
    Returns:
        DataFrame: Driver statistics
    """
    if historical_data.empty:
        return pd.DataFrame()
    
    driver_stats = []
    
    # Group by driver
    for driver, group in historical_data.groupby('Driver'):
        # Calculate statistics
        avg_finish = group['Position'].mean()
        avg_grid = group['GridPosition'].mean()
        avg_positions_gained = (group['GridPosition'] - group['Position']).mean()
        
        # Calculate finishing rate
        races = len(group)
        dnfs = sum('DNF' in str(interval) for interval in group['Interval'])
        finishing_rate = (races - dnfs) / races
        
        # Add to driver stats
        driver_stats.append({
            'Driver': driver,
            'AvgFinish': avg_finish,
            'AvgGrid': avg_grid,
            'AvgPositionsGained': avg_positions_gained,
            'FinishingRate': finishing_rate,
            'Races': races
        })
    
    # Create DataFrame
    driver_stats_df = pd.DataFrame(driver_stats)
    
    return driver_stats_df

def calculate_team_stats(historical_data):
    """
    Calculate team statistics from historical data.
    
    Args:
        historical_data (DataFrame): Historical race results
        
    Returns:
        DataFrame: Team statistics
    """
    if historical_data.empty:
        return pd.DataFrame()
    
    team_stats = []
    
    # Group by team
    for team, group in historical_data.groupby('Team'):
        # Calculate statistics
        avg_finish = group['Position'].mean()
        avg_grid = group['GridPosition'].mean()
        
        # Calculate points
        if 'Points' in group.columns:
            avg_points = group['Points'].mean()
            total_points = group['Points'].sum()
        else:
            # Calculate points from positions if not provided
            points = [calculate_points(pos) for pos in group['Position']]
            avg_points = sum(points) / len(points)
            total_points = sum(points)
        
        # Add to team stats
        team_stats.append({
            'Team': team,
            'AvgFinish': avg_finish,
            'AvgGrid': avg_grid,
            'AvgPoints': avg_points,
            'TotalPoints': total_points,
            'Races': len(group)
        })
    
    # Create DataFrame
    team_stats_df = pd.DataFrame(team_stats)
    
    return team_stats_df

def calculate_points(position):
    """
    Calculate F1 points for a position.
    
    Args:
        position (int): Finishing position
        
    Returns:
        int: Points scored
    """
    # Standard F1 points system 
    points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    
    return points_system.get(position, 0)

def prepare_prediction_features(quali_data, driver_stats=None, team_stats=None):
    """
    Prepare features for prediction model.
    
    Args:
        quali_data (DataFrame): Qualifying data
        driver_stats (DataFrame): Driver statistics
        team_stats (DataFrame): Team statistics
        
    Returns:
        DataFrame: Features for prediction
    """
    # Create a copy of qualifying data
    features = quali_data.copy()
    
    # Add driver stats if available
    if driver_stats is not None and not driver_stats.empty:
        # Merge driver stats
        features = pd.merge(features, driver_stats, on='Driver', how='left')
    
    # Add team stats if available
    if team_stats is not None and not team_stats.empty:
        # Merge team stats
        features = pd.merge(features, team_stats, on='Team', how='left')
    
    # Calculate grid position ranking
    features['GridRank'] = features['GridPosition'].rank(method='min')
    
    # Handle missing values
    # For drivers without historical data, use average values
    for col in ['AvgFinish', 'AvgGrid', 'AvgPositionsGained', 'FinishingRate']:
        if col in features.columns and features[col].isna().any():
            mean_val = features[col].mean()
            features[col].fillna(mean_val, inplace=True)
    
    # For teams without historical data, use average values
    for col in ['AvgFinish', 'AvgGrid', 'AvgPoints']:
        if col in features.columns and features[col].isna().any():
            mean_val = features[col].mean()
            features[col].fillna(mean_val, inplace=True)
    
    return features