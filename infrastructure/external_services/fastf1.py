# infrastructure/external_services/fastf1_client.py
"""FastF1 API client implementation.

This module provides a robust client for the FastF1 API to fetch Formula 1 data.
"""

import logging
import os
import time
import random
import pandas as pd
import numpy as np
from typing import Optional, Union, Any, Dict
from datetime import datetime

# Try to import fastf1, handle if not available
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

from infrastructure.external_services.rate_limiter import RateLimiter
from infrastructure.persistence.data_cache import DataCache

# Constants
DEFAULT_FASTF1_CACHE_DIR = "cache/fastf1"
MAX_RETRIES = 3

# Set up logging
logger = logging.getLogger("f1_prediction.external_services.fastf1_client")

class RateLimitExceededError(Exception):
    """Exception raised when API rate limit is exceeded."""
    pass

class FastF1Client:
    """
    Client for FastF1 API with enhanced error handling.
    
    Handles data fetching, caching, and error recovery.
    """
    def __init__(
        self, 
        cache_dir: str = DEFAULT_FASTF1_CACHE_DIR,
        rate_limiter: Optional[RateLimiter] = None,
        data_cache: Optional[DataCache] = None,
        enable_cache: bool = True
    ):
        """
        Initialize FastF1 client.
        
        Args:
            cache_dir: Directory for FastF1 cache
            rate_limiter: Rate limiter for API calls
            data_cache: Data cache for API results
            enable_cache: Whether to enable FastF1 caching
        """
        self.available = FASTF1_AVAILABLE
        self.rate_limiter = rate_limiter or RateLimiter()
        self.data_cache = data_cache or DataCache()
        
        if self.available and enable_cache:
            # Setup FastF1 cache
            fastf1.Cache.enable_cache(cache_dir)
            logger.info(f"FastF1 cache enabled at {cache_dir}")
        
    def get_session(
        self, 
        year: int, 
        round_num: int, 
        session_type: str = 'Q',
        force_refresh: bool = False
    ) -> Optional[Any]:
        """
        Get F1 session data with error handling.
        
        Args:
            year: Season year
            round_num: Round number
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3')
            force_refresh: Whether to bypass cache
            
        Returns:
            FastF1 session or None if not available
        """
        if not self.available:
            logger.error("FastF1 not available")
            return None
        
        # Check our internal cache if not forcing refresh
        if not force_refresh:
            cache_key = f"session_{year}_{round_num}_{session_type}_raw"
            cached_session = self.data_cache.get(cache_key)
            if cached_session is not None:
                return cached_session
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(f"Fetching {session_type} data for {year} round {round_num}")
                
                # Get session
                session = fastf1.get_session(year, round_num, session_type)
                # Load data
                session.load()
                
                # Cache the session
                if not force_refresh:
                    self.data_cache.set(cache_key, session)
                
                return session
                
            except RateLimitExceededError as e:
                if attempt < MAX_RETRIES:
                    wait_time = (2 ** attempt) * 5 + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")
                    return None
            except Exception as e:
                logger.error(f"Error fetching session: {e}")
                if attempt < MAX_RETRIES:
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
                    logger.warning(f"Retrying in {wait_time:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {MAX_RETRIES} retries")
                    return None
    
    def get_quali_results(self, year: int, round_num: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get qualifying results with enhanced error handling.
        
        Args:
            year: Season year
            round_num: Round number
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with qualifying results
        """
        # Check our cache first
        if not force_refresh:
            cache_key = f"quali_results_{year}_{round_num}"
            cached_results = self.data_cache.get(cache_key)
            if cached_results is not None:
                return cached_results
        
        try:
            # Get session
            session = self.get_session(year, round_num, 'Q')
            if session is None:
                logger.error(f"Could not get qualifying session for {year} round {round_num}")
                return pd.DataFrame()
            
            # Get results
            quali_results = session.get_qualifying()
            
            # Process results
            results = self._process_quali_results(quali_results)
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, results)
            
            return results
        except Exception as e:
            logger.error(f"Error getting qualifying results: {e}")
            return pd.DataFrame()
    
    def get_race_results(self, year: int, round_num: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get race results with enhanced error handling.
        
        Args:
            year: Season year
            round_num: Round number
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with race results
        """
        # Check our cache first
        if not force_refresh:
            cache_key = f"race_results_{year}_{round_num}"
            cached_results = self.data_cache.get(cache_key)
            if cached_results is not None:
                return cached_results
        
        try:
            # Get session
            session = self.get_session(year, round_num, 'R')
            if session is None:
                logger.error(f"Could not get race session for {year} round {round_num}")
                return pd.DataFrame()
            
            # Get results
            race_results = session.get_race_results()
            
            # Process results
            results = self._process_race_results(race_results, session)
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, results)
            
            return results
        except Exception as e:
            logger.error(f"Error getting race results: {e}")
            return pd.DataFrame()
    
    def _process_quali_results(self, quali_results: pd.DataFrame) -> pd.DataFrame:
        """Process qualifying results for consistent format."""
        if quali_results is None or quali_results.empty:
            return pd.DataFrame()
        
        # Make a copy
        processed = quali_results.copy()
        
        # Rename columns
        column_mapping = {
            'DriverNumber': 'DriverNumber',
            'Abbreviation': 'DriverCode',
            'FullName': 'Driver',
            'TeamName': 'TeamName',
            'Position': 'GridPosition'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in processed.columns:
                processed[new_col] = processed[old_col]
        
        # Process qualifying times
        for q_col in ['Q1', 'Q2', 'Q3']:
            if q_col in processed.columns:
                processed[f'{q_col}_sec'] = self._convert_time_to_seconds(processed[q_col])
        
        # Calculate best qualifying time
        if all(f'{q_col}_sec' in processed.columns for q_col in ['Q1', 'Q2', 'Q3']):
            processed['QualifyingBestTime'] = processed[[
                'Q1_sec', 'Q2_sec', 'Q3_sec'
            ]].min(axis=1)
        
        # Calculate gap to pole
        if 'QualifyingBestTime' in processed.columns:
            pole_time = processed['QualifyingBestTime'].min()
            if not pd.isna(pole_time):
                processed['QualifyingGapToPole'] = processed['QualifyingBestTime'] - pole_time
        
        # Calculate relative position
        if 'GridPosition' in processed.columns:
            processed['RelativeGridPosition'] = processed['GridPosition'].rank(method='min')
        
        return processed
    
    def _process_race_results(self, race_results: pd.DataFrame, session: Any) -> pd.DataFrame:
        """Process race results with additional telemetry data."""
        if race_results is None or race_results.empty:
            return pd.DataFrame()
        
        # Make a copy
        processed = race_results.copy()
        
        # Rename columns
        column_mapping = {
            'DriverNumber': 'DriverNumber',
            'Abbreviation': 'DriverCode',
            'FullName': 'Driver',
            'TeamName': 'TeamName',
            'Position': 'RacePosition'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in processed.columns:
                processed[new_col] = processed[old_col]
        
        # Add fastest lap times
        try:
            laps = session.laps
            if not laps.empty:
                # Group by driver and get fastest lap
                fastest_laps = laps.groupby('DriverNumber')['LapTime'].min().reset_index()
                
                # Convert to seconds
                fastest_laps['FastestLap_sec'] = self._convert_time_to_seconds(fastest_laps['LapTime'])
                
                # Merge with race results
                processed = pd.merge(
                    processed, 
                    fastest_laps[['DriverNumber', 'FastestLap_sec']], 
                    on='DriverNumber', 
                    how='left'
                )
        except Exception as e:
            logger.warning(f"Could not process fastest laps: {e}")
        
        # Add interval to winner
        try:
            # Find the winner
            winner_idx = processed['RacePosition'].idxmin()
            
            # For each driver, calculate interval
            processed['Interval_sec'] = np.nan
            processed.loc[winner_idx, 'Interval_sec'] = 0.0
            
            if hasattr(session, 'timings') and not session.timings.empty:
                timings = session.timings
                
                # Get the final timing for each driver
                final_timings = timings.groupby('DriverNumber').last().reset_index()
                
                # Calculate intervals
                winner_time = final_timings.loc[
                    final_timings['DriverNumber'] == processed.loc[winner_idx, 'DriverNumber'], 
                    'Time'
                ].iloc[0]
                
                for idx, row in processed.iterrows():
                    if idx == winner_idx:
                        continue
                    
                    driver_time = final_timings.loc[
                        final_timings['DriverNumber'] == row['DriverNumber'], 
                        'Time'
                    ]
                    
                    if not driver_time.empty:
                        interval = driver_time.iloc[0] - winner_time
                        processed.loc[idx, 'Interval_sec'] = interval.total_seconds()
            
            # If we still have missing intervals, use synthetic ones
            if processed['Interval_sec'].isna().any():
                for idx, row in processed.iterrows():
                    if pd.isna(row['Interval_sec']):
                        # Use position as a proxy (~2 seconds per position)
                        position_diff = row['RacePosition'] - processed.loc[winner_idx, 'RacePosition']
                        processed.loc[idx, 'Interval_sec'] = position_diff * 2.0
                        
        except Exception as e:
            logger.warning(f"Could not process intervals: {e}")
            # Use synthetic intervals
            processed['Interval_sec'] = (processed['RacePosition'] - 1) * 2.0
        
        return processed
    
    def _convert_time_to_seconds(self, time_values: Union[pd.Series, Any]) -> pd.Series:
        """Convert various time formats to seconds."""
        if isinstance(time_values, pd.Series):
            return time_values.apply(self._convert_single_time_to_seconds)
        else:
            return self._convert_single_time_to_seconds(time_values)
    
    def _convert_single_time_to_seconds(self, time_value: Any) -> Optional[float]:
        """Convert a single time value to seconds."""
        if pd.isna(time_value):
            return None
        
        try:
            # Try to get total_seconds if it's a timedelta
            if hasattr(time_value, 'total_seconds'):
                return time_value.total_seconds()
            
            # If it's a string, parse it
            if isinstance(time_value, str):
                # Format like "1:30.456"
                if ':' in time_value:
                    parts = time_value.split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    # Just seconds
                    return float(time_value)
            
            # If it's a number, return as is
            if isinstance(time_value, (int, float)):
                return float(time_value)
            
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Error converting time value {time_value}: {e}")
        
        return None