# data/fetching.py
"""
Data fetching module for F1 race predictions.

This module provides functions to load and fetch F1 race data.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

# Add parent directory to path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.helpers import DRIVERS
except ImportError:
    from utils.constants import DRIVERS

# Set up logging
logger = logging.getLogger('f1_prediction.data_fetching')

def load_sample_data(data_type='qualifying'):
    """
    Load sample data for development and testing.
    
    Args:
        data_type (str): Type of data ('qualifying' or 'race')
        
    Returns:
        DataFrame: Sample F1 data
    """
    # Use drivers and teams from the DRIVERS dictionary
    drivers = list(DRIVERS.keys())
    teams = [DRIVERS[driver] for driver in drivers]
    
    if data_type == 'qualifying':
        # Generate qualifying data
        # For simplicity, we'll use a base time and add some variation
        base_time = 80.0  # Base lap time in seconds
        
        # Create a DataFrame with qualifying results
        data = []
        
        for i, (driver, team) in enumerate(zip(drivers, teams)):
            # Generate grid position (roughly aligned with driver ability but with randomness)
            # We ensure positions are unique by adding the index
            grid_position = i + 1
            
            # Generate qualifying time - better drivers are generally faster
            # but we add some randomness to make it realistic
            quali_time = base_time + (grid_position * 0.1) + np.random.normal(0, 0.2)
            
            # Add to data
            data.append({
                'Driver': driver,
                'Team': team,
                'GridPosition': grid_position, 
                'Q1': quali_time + np.random.normal(0, 0.3),
                'Q2': quali_time + np.random.normal(0, 0.2) if grid_position <= 15 else None,
                'Q3': quali_time + np.random.normal(0, 0.1) if grid_position <= 10 else None,
                'QualifyingTime': quali_time
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate gap to pole
        pole_time = df['QualifyingTime'].min()
        df['GapToPole'] = df['QualifyingTime'] - pole_time
        
        # Add Grid column for compatibility
        df['Grid'] = df['GridPosition']
        
        return df
        
    elif data_type == 'race':
        # Generate race data
        # Create random grid (qualifying) and race positions
        grid_positions = list(range(1, len(drivers) + 1))
        
        # Shuffle positions slightly to simulate race performance differences
        race_positions = grid_positions.copy()
        for i in range(len(race_positions)):
            # 70% chance of position change
            if np.random.random() < 0.7:
                # Move up or down 1-3 positions
                change = np.random.choice([-3, -2, -1, 1, 2, 3])
                new_pos = race_positions[i] + change
                
                # Ensure position is valid (1 to 20)
                new_pos = max(1, min(20, new_pos))
                
                # Find current driver in that position and swap
                if new_pos != race_positions[i]:
                    idx_to_swap = race_positions.index(new_pos)
                    race_positions[i], race_positions[idx_to_swap] = new_pos, race_positions[i]
        
        # Create data with intervals
        data = []
        for i, (driver, team) in enumerate(zip(drivers, teams)):
            position = race_positions[i]
            grid = grid_positions[i]
            
            # Calculate interval from winner
            if position == 1:
                interval = 0.0
                interval_str = "WINNER"
            else:
                # Generate a plausible interval - increases with position
                interval = (position - 1) * 2.5 + np.random.normal(0, 0.5)
                interval_str = f"+{interval:.3f}s"
            
            data.append({
                'Driver': driver,
                'Team': team,
                'GridPosition': grid,
                'Grid': grid,  # Add Grid column for compatibility
                'Position': position,
                'Interval': interval_str,
                'IntervalSeconds': interval,
                'Interval (s)': interval,  # Add Interval (s) for compatibility with visualizations
                'Points': calculate_points(position)
            })
        
        # Create DataFrame and sort by position
        df = pd.DataFrame(data)
        df = df.sort_values('Position').reset_index(drop=True)
        
        return df
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")

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

def fetch_race_data(year=None, round_num=None, manual_data=None):
    """
    Fetch race data from a specific year and round, or use provided data.
    
    Args:
        year (int): Year of race
        round_num (int): Round number
        manual_data (dict): Manually provided data
        
    Returns:
        DataFrame: Race data
    """
    if manual_data is not None:
        # Use manually provided data
        # Expected format: list of dicts with driver, team, and grid position
        
        # Create DataFrame
        df = pd.DataFrame(manual_data)
        
        # Make sure required columns exist
        if 'Driver' not in df.columns:
            raise ValueError("Manual data must contain a 'Driver' column")
        
        # If Team is not provided, add it based on DRIVERS dict
        if 'Team' not in df.columns:
            df['Team'] = df['Driver'].apply(lambda driver: DRIVERS.get(driver, "Unknown Team"))
        
        # Add grid positions if not provided
        if 'GridPosition' not in df.columns and 'Grid' not in df.columns:
            df['GridPosition'] = range(1, len(df) + 1)
            df['Grid'] = df['GridPosition']
        
        return df
    
    # If no year/round specified, return sample data
    if year is None or round_num is None:
        return load_sample_data(data_type='qualifying')
    
    # In a real implementation, this would fetch data from an API
    # For this simplified version, we'll just return sample data
    logger.info(f"Fetching data for {year} round {round_num}")
    
    # Return sample data
    return load_sample_data(data_type='qualifying')