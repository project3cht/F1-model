"""
Updated FastF1 API client that integrates with our F1Schedule.

This module provides a robust client for the FastF1 API to fetch Formula 1 data
using our enhanced F1Schedule class for schedule information.
"""

import logging
import os
import time
import random
import pandas as pd
import numpy as np
from typing import Optional, Union, Any, Dict, List
from datetime import datetime

# Try to import fastf1, handle if not available
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

from infrastructure.external_services.rate_limiter import RateLimiter
from infrastructure.persistence.data_cache import DataCache
from utils.constants import f1_schedule, get_race_info

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
    Client for FastF1 API with enhanced error handling and integration with F1Schedule.
    
    Handles data fetching, caching, and error recovery using our central F1Schedule.
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
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3', 'S', 'SS', 'SQ')
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
        
        # Get race info from our schedule to check event format
        race_info = get_race_info(round_num, year)
        
        # Check if this session exists for this event format
        if race_info:
            event_format = race_info.get('format', 'conventional')
            
            # Verify session exists for this format
            if session_type in ['SS', 'Sprint Shootout'] and 'sprint_shootout' not in event_format:
                logger.warning(f"Sprint Shootout not available for {event_format} format in round {round_num}")
                return None
            
            if session_type in ['SQ', 'Sprint Qualifying'] and 'sprint_qualifying' not in event_format:
                logger.warning(f"Sprint Qualifying not available for {event_format} format in round {round_num}")
                return None
            
            if session_type in ['S', 'Sprint'] and 'sprint' not in event_format:
                logger.warning(f"Sprint not available for {event_format} format in round {round_num}")
                return None
            
            if session_type == 'FP3' and event_format != 'conventional':
                logger.warning(f"FP3 not available for {event_format} format in round {round_num}")
                return None
        
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
    
    def get_sprint_results(self, year: int, round_num: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get sprint race results with enhanced error handling.
        
        Args:
            year: Season year
            round_num: Round number
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with sprint results
        """
        # Check our cache first
        if not force_refresh:
            cache_key = f"sprint_results_{year}_{round_num}"
            cached_results = self.data_cache.get(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Get race info from our schedule to check event format
        race_info = get_race_info(round_num, year)
        
        # Check if sprint exists for this event
        if race_info:
            event_format = race_info.get('format', 'conventional')
            if 'sprint' not in event_format:
                logger.warning(f"Sprint not available for {event_format} format in round {round_num}")
                return pd.DataFrame()
        
        try:
            # Get session
            session = self.get_session(year, round_num, 'S')
            if session is None:
                logger.error(f"Could not get sprint session for {year} round {round_num}")
                return pd.DataFrame()
            
            # Get results
            sprint_results = session.get_race_results()
            
            # Process results
            results = self._process_race_results(sprint_results, session)
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, results)
            
            return results
        except Exception as e:
            logger.error(f"Error getting sprint results: {e}")
            return pd.DataFrame()
    
    def get_driver_info(self, year: int = 2025) -> pd.DataFrame:
        """
        Get driver information for a specific year.
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with driver information
        """
        # Check cache
        cache_key = f"driver_info_{year}"
        cached_results = self.data_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        if not self.available:
            logger.error("FastF1 not available")
            return pd.DataFrame()
        
        try:
            # Apply rate limiting
            self.rate_limiter.wait()
            
            # Get driver information
            if year == 2025:
                # For 2025, use our constants as FastF1 may not have this data yet
                from utils.constants import DRIVERS
                
                # Create DataFrame
                driver_data = []
                for driver, team in DRIVERS.items():
                    # Split driver name
                    name_parts = driver.split(' ')
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])
                    
                    driver_data.append({
                        'FullName': driver,
                        'FirstName': first_name,
                        'LastName': last_name,
                        'TeamName': team
                    })
                
                drivers_df = pd.DataFrame(driver_data)
            else:
                # Get schedule to extract a race for driver data
                schedule = f1_schedule.get_event_schedule(year)
                
                # Get first race
                if 'RoundNumber' in schedule.columns:
                    round_num = schedule['RoundNumber'].iloc[0]
                else:
                    round_num = schedule['round'].iloc[0]
                
                # Get session
                session = self.get_session(year, round_num, 'R')
                if session is None:
                    logger.error(f"Could not get session for {year} round {round_num}")
                    return pd.DataFrame()
                
                # Get driver information
                drivers_df = session.get_driver_info()
            
            # Cache results
            self.data_cache.set(cache_key, drivers_df)
            
            return drivers_df
        except Exception as e:
            logger.error(f"Error getting driver information: {e}")
            return pd.DataFrame()
    
    def get_circuit_info(self, year: int, round_num: int) -> Dict:
        """
        Get circuit information for a specific round.
        
        Args:
            year: Season year
            round_num: Round number
            
        Returns:
            Dictionary with circuit information
        """
        # Check cache
        cache_key = f"circuit_info_{year}_{round_num}"
        cached_results = self.data_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Get race info from our schedule
        race_info = get_race_info(round_num, year)
        
        if not race_info:
            logger.error(f"Could not find race info for round {round_num}")
            return {}
        
        # Extract track information
        circuit_info = {
            'name': race_info['track'],
            'country': race_info['country'],
            'lap_count': None,  # FastF1 would provide this
            'lap_distance': None  # FastF1 would provide this
        }
        
        # If FastF1 is available, try to get additional information
        if self.available:
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                # Get session
                session = self.get_session(year, round_num, 'R')
                if session is not None:
                    # Extract additional information
                    if hasattr(session, 'event'):
                        if hasattr(session.event, 'laps'):
                            circuit_info['lap_count'] = session.event.laps
                        
                        if hasattr(session.event, 'distance'):
                            circuit_info['lap_distance'] = session.event.distance / session.event.laps
            except Exception as e:
                logger.warning(f"Could not get additional circuit information from FastF1: {e}")
        
        # Add track characteristics
        track_key = f1_schedule._find_matching_track(race_info['track'])
        from utils.constants import TRACK_CHARACTERISTICS
        
        if track_key in TRACK_CHARACTERISTICS:
            circuit_info['characteristics'] = TRACK_CHARACTERISTICS[track_key]
        
        # Cache results
        self.data_cache.set(cache_key, circuit_info)
        
        return circuit_info
    
    def get_weather_data(self, year: int, round_num: int, session_type: str = 'R') -> pd.DataFrame:
        """
        Get weather data for a specific session.
        
        Args:
            year: Season year
            round_num: Round number
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3', 'S')
            
        Returns:
            DataFrame with weather data
        """
        if not self.available:
            logger.error("FastF1 not available")
            return pd.DataFrame()
        
        # Check cache
        cache_key = f"weather_{year}_{round_num}_{session_type}"
        cached_results = self.data_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        try:
            # Get session
            session = self.get_session(year, round_num, session_type)
            if session is None:
                logger.error(f"Could not get session for {year} round {round_num}")
                return pd.DataFrame()
            
            # Get weather data
            weather = session.weather_data
            
            # Cache results
            self.data_cache.set(cache_key, weather)
            
            return weather
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
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
            'TeamName': 'Team',
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
            'TeamName': 'Team',
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
        
        # Add intervals as formatted string
        processed['Interval'] = processed['Interval_sec'].apply(
            lambda x: "WINNER" if x == 0 else f"+{x:.3f}s"
        )
        
        # Add grid position if not already present
        if 'GridPosition' not in processed.columns and 'Grid' in processed.columns:
            processed['GridPosition'] = processed['Grid']
        
        # Calculate points based on position
        if 'Points' not in processed.columns:
            processed['Points'] = processed['RacePosition'].apply(self._calculate_points)
        
        return processed
    
    def _calculate_points(self, position: int) -> float:
        """
        Calculate F1 points for a position.
        
        Args:
            position: Race position
            
        Returns:
            float: Points awarded
        """
        # Standard F1 points system
        points_system = {
            1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0,
            6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0
        }
        
        return points_system.get(position, 0.0)
    
    def _convert_time_to_seconds(self, time_values: Union[pd.Series, Any]) -> Union[pd.Series, float]:
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
    
    def get_lap_data(self, year: int, round_num: int, driver: Optional[str] = None,
                    session_type: str = 'R', force_refresh: bool = False) -> pd.DataFrame:
        """
        Get lap data for a specific session with optional driver filter.
        
        Args:
            year: Season year
            round_num: Round number
            driver: Optional driver filter
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3', 'S')
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with lap data
        """
        # Create cache key with optional driver
        driver_suffix = f"_{driver.replace(' ', '_')}" if driver else ""
        cache_key = f"laps_{year}_{round_num}_{session_type}{driver_suffix}"
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        if not self.available:
            logger.error("FastF1 not available")
            return pd.DataFrame()
        
        try:
            # Get session
            session = self.get_session(year, round_num, session_type)
            if session is None:
                logger.error(f"Could not get session for {year} round {round_num}")
                return pd.DataFrame()
            
            # Get lap data
            laps = session.laps
            
            # Filter by driver if specified
            if driver and not laps.empty:
                driver_laps = laps.pick_driver(driver)
                if driver_laps.empty:
                    logger.warning(f"No lap data found for driver {driver}")
                    return pd.DataFrame()
                laps = driver_laps
            
            # Process lap data
            processed_laps = self._process_lap_data(laps)
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, processed_laps)
            
            return processed_laps
        except Exception as e:
            logger.error(f"Error getting lap data: {e}")
            return pd.DataFrame()
    
    def _process_lap_data(self, laps: pd.DataFrame) -> pd.DataFrame:
        """Process lap data for consistent format."""
        if laps is None or laps.empty:
            return pd.DataFrame()
        
        # Make a copy
        processed = laps.copy()
        
        # Rename columns for consistency
        column_mapping = {
            'DriverNumber': 'DriverNumber',
            'Driver': 'DriverCode',
            'TeamName': 'Team',
            'LapNumber': 'LapNumber',
            'LapTime': 'LapTime',
            'Sector1Time': 'Sector1Time',
            'Sector2Time': 'Sector2Time',
            'Sector3Time': 'Sector3Time',
            'Compound': 'TireCompound',
            'TyreLife': 'TireAge',
            'FreshTyre': 'FreshTire',
            'IsPersonalBest': 'IsPersonalBest',
            'PitInTime': 'PitInTime',
            'PitOutTime': 'PitOutTime',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in processed.columns:
                processed[new_col] = processed[old_col]
        
        # Convert time columns to seconds
        time_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'PitInTime', 'PitOutTime']
        for col in time_columns:
            if col in processed.columns:
                processed[f'{col}_sec'] = self._convert_time_to_seconds(processed[col])
        
        # Add PitIn/PitOut flags if not present
        if 'PitIn' not in processed.columns and 'PitInTime' in processed.columns:
            processed['PitIn'] = ~processed['PitInTime'].isna()
        
        if 'PitOut' not in processed.columns and 'PitOutTime' in processed.columns:
            processed['PitOut'] = ~processed['PitOutTime'].isna()
        
        return processed
    
    def get_telemetry_data(self, year: int, round_num: int, driver: str, lap: int,
                         session_type: str = 'R', force_refresh: bool = False) -> pd.DataFrame:
        """
        Get telemetry data for a specific lap.
        
        Args:
            year: Season year
            round_num: Round number
            driver: Driver name or abbreviation
            lap: Lap number
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3', 'S')
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with telemetry data
        """
        # Create cache key
        cache_key = f"telemetry_{year}_{round_num}_{session_type}_{driver.replace(' ', '_')}_lap{lap}"
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        if not self.available:
            logger.error("FastF1 not available")
            return pd.DataFrame()
        
        try:
            # Get lap data first
            lap_data = self.get_lap_data(year, round_num, driver, session_type)
            if lap_data.empty:
                logger.error(f"No lap data found for {driver}")
                return pd.DataFrame()
            
            # Find the requested lap
            lap_row = lap_data[lap_data['LapNumber'] == lap]
            if lap_row.empty:
                logger.error(f"Lap {lap} not found for {driver}")
                return pd.DataFrame()
            
            # Get telemetry using FastF1
            if hasattr(lap_row.iloc[0], 'get_telemetry'):
                telemetry = lap_row.iloc[0].get_telemetry()
            else:
                # Fallback: get session and extract telemetry
                session = self.get_session(year, round_num, session_type)
                laps = session.laps.pick_driver(driver)
                lap_row_ff1 = laps[laps['LapNumber'] == lap]
                if lap_row_ff1.empty:
                    logger.error(f"Lap {lap} not found for {driver} in FastF1 data")
                    return pd.DataFrame()
                
                telemetry = lap_row_ff1.iloc[0].get_telemetry()
            
            # Process telemetry data
            processed_telemetry = self._process_telemetry_data(telemetry)
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, processed_telemetry)
            
            return processed_telemetry
        except Exception as e:
            logger.error(f"Error getting telemetry data: {e}")
            return pd.DataFrame()
    
    def _process_telemetry_data(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Process telemetry data for consistent format."""
        if telemetry is None or telemetry.empty:
            return pd.DataFrame()
        
        # Make a copy
        processed = telemetry.copy()
        
        # Rename columns for consistency
        column_mapping = {
            'SessionTime': 'SessionTime',
            'Speed': 'Speed',
            'Distance': 'Distance',
            'Throttle': 'Throttle',
            'Brake': 'Brake',
            'nGear': 'Gear',
            'RPM': 'RPM',
            'DRS': 'DRS'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in processed.columns:
                processed[new_col] = processed[old_col]
        
        # Convert SessionTime to seconds if needed
        if 'SessionTime' in processed.columns:
            processed['SessionTime_sec'] = self._convert_time_to_seconds(processed['SessionTime'])
        
        return processed
    
    def compare_drivers_pace(self, year: int, round_num: int, drivers: List[str],
                           session_type: str = 'R', force_refresh: bool = False) -> Dict:
        """
        Compare race pace between multiple drivers.
        
        Args:
            year: Season year
            round_num: Round number
            drivers: List of driver names to compare
            session_type: Session type ('Q', 'R', 'FP1', 'FP2', 'FP3', 'S')
            force_refresh: Whether to bypass cache
            
        Returns:
            Dictionary with pace comparison data
        """
        # Create cache key
        drivers_key = "_".join(sorted([d.replace(' ', '_') for d in drivers]))
        cache_key = f"pace_comparison_{year}_{round_num}_{session_type}_{drivers_key}"
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Get lap data for each driver
        driver_laps = {}
        for driver in drivers:
            laps = self.get_lap_data(year, round_num, driver, session_type)
            if not laps.empty:
                driver_laps[driver] = laps
        
        if not driver_laps:
            logger.error("No lap data found for any driver")
            return {}
        
        # Calculate pace statistics
        pace_comparison = {
            'drivers': drivers,
            'session': session_type,
            'round': round_num,
            'year': year,
            'metrics': {}
        }
        
        for driver, laps in driver_laps.items():
            # Filter valid laps (exclude in/out laps, safety car, etc.)
            valid_laps = laps[
                ~laps['PitIn'].fillna(False) & 
                ~laps['PitOut'].fillna(False)
            ]
            
            if 'LapTime_sec' in valid_laps.columns:
                lap_times = valid_laps['LapTime_sec'].dropna()
                
                if len(lap_times) > 0:
                    # Calculate statistics
                    pace_comparison['metrics'][driver] = {
                        'median_lap': lap_times.median(),
                        'mean_lap': lap_times.mean(),
                        'fastest_lap': lap_times.min(),
                        'lap_count': len(lap_times),
                        'std_dev': lap_times.std()
                    }
                    
                    # Add percentiles
                    for percentile in [25, 75, 90]:
                        pace_comparison['metrics'][driver][f'percentile_{percentile}'] = lap_times.quantile(percentile / 100)
                    
                    # Analyze stint consistency
                    stints = []
                    current_stint = []
                    
                    for i, row in valid_laps.sort_values('LapNumber').reset_index(drop=True).iterrows():
                        # Check if this is a new stint
                        if i > 0 and row['TireAge'] == 1:
                            if current_stint:
                                stints.append(current_stint)
                                current_stint = []
                        
                        # Add lap to current stint
                        if 'LapTime_sec' in row and not pd.isna(row['LapTime_sec']):
                            current_stint.append({
                                'lap': row['LapNumber'],
                                'time': row['LapTime_sec'],
                                'tire_age': row['TireAge'] if 'TireAge' in row else None,
                                'compound': row['TireCompound'] if 'TireCompound' in row else None
                            })
                    
                    # Add last stint
                    if current_stint:
                        stints.append(current_stint)
                    
                    # Calculate stint metrics
                    stint_metrics = []
                    for i, stint in enumerate(stints):
                        if len(stint) >= 3:  # Only analyze stints with 3+ laps
                            stint_times = [lap['time'] for lap in stint]
                            stint_metrics.append({
                                'stint': i + 1,
                                'laps': len(stint),
                                'tire': stint[0]['compound'] if stint[0]['compound'] else 'Unknown',
                                'median': np.median(stint_times),
                                'degradation': (stint_times[-1] - stint_times[0]) / len(stint) if len(stint) > 1 else 0
                            })
                    
                    pace_comparison['metrics'][driver]['stints'] = stint_metrics
                    
                    # Flag if good data for comparison
                    pace_comparison['metrics'][driver]['is_comparable'] = len(lap_times) >= 5
        
        # Cache results
        if not force_refresh:
            self.data_cache.set(cache_key, pace_comparison)
        
        return pace_comparison