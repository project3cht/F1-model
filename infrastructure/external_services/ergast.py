# infrastructure/external_services/ergast_client.py
"""Ergast API client implementation.

This module provides a client for the Ergast F1 API as a backup data source.
"""

import logging
import requests
from typing import Optional
from urllib.parse import urljoin
import pandas as pd
import numpy as np

from infrastructure.external_services.rate_limiter import RateLimiter
from infrastructure.persistence.data_cache import DataCache

# Constants
API_TIMEOUT = 10  # seconds

# Set up logging
logger = logging.getLogger("f1_prediction.external_services.ergast_client")

class ErgastClient:
    """
    Client for Ergast F1 API as a backup data source.
    
    Provides access to historical F1 data when FastF1 is unavailable.
    """
    def __init__(
        self, 
        base_url: str = "http://ergast.com/api/f1/",
        rate_limiter: Optional[RateLimiter] = None,
        data_cache: Optional[DataCache] = None
    ):
        """
        Initialize Ergast client.
        
        Args:
            base_url: Base URL for Ergast API
            rate_limiter: Rate limiter for API calls
            data_cache: Data cache for API results
        """
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter(calls_per_hour=150)  # More conservative for Ergast
        self.data_cache = data_cache or DataCache()
    
    def get_qualifying_results(
        self, 
        year: int, 
        round_num: int, 
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get qualifying results from Ergast API.
        
        Args:
            year: Season year
            round_num: Round number
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with qualifying results
        """
        # Check cache first
        if not force_refresh:
            cache_key = f"ergast_quali_{year}_{round_num}"
            cached_results = self.data_cache.get(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        logger.info(f"Fetching qualifying data from Ergast API for {year}, round {round_num}")
        
        # Construct the API URL
        endpoint = f"{year}/{round_num}/qualifying.json"
        url = urljoin(self.base_url, endpoint)
        
        try:
            # Make the API request
            response = requests.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            
            # Check if there is qualifying data
            race_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            if not race_data:
                logger.warning(f"No qualifying data available for {year}, round {round_num}")
                return pd.DataFrame()
            
            # Extract qualifying results
            qualifying_results = race_data[0].get('QualifyingResults', [])
            if not qualifying_results:
                logger.warning(f"No qualifying results available for {year}, round {round_num}")
                return pd.DataFrame()
            
            # Create a DataFrame from the qualifying results
            quali_data = []
            
            for result in qualifying_results:
                driver_info = result.get('Driver', {})
                constructor_info = result.get('Constructor', {})
                
                # Extract Q1, Q2, Q3 times and convert to seconds
                q1_time = self._convert_time_to_seconds(result.get('Q1', ''))
                q2_time = self._convert_time_to_seconds(result.get('Q2', ''))
                q3_time = self._convert_time_to_seconds(result.get('Q3', ''))
                
                # Determine best qualifying time
                best_time = min([t for t in [q1_time, q2_time, q3_time] if t is not None], default=None)
                
                # Create entry
                entry = {
                    'DriverNumber': result.get('number', ''),
                    'Driver': f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                    'DriverCode': driver_info.get('code', ''),
                    'TeamName': constructor_info.get('name', ''),
                    'GridPosition': int(result.get('position', 0)),
                    'Q1_sec': q1_time,
                    'Q2_sec': q2_time,
                    'Q3_sec': q3_time,
                    'QualifyingBestTime': best_time
                }
                
                quali_data.append(entry)
            
            # Convert to DataFrame
            quali_df = pd.DataFrame(quali_data)
            
            # Calculate gap to pole if we have best times
            if 'QualifyingBestTime' in quali_df.columns and not quali_df['QualifyingBestTime'].isna().all():
                # Find the pole time
                pole_time = quali_df['QualifyingBestTime'].min()
                # Calculate gaps
                quali_df['QualifyingGapToPole'] = quali_df['QualifyingBestTime'] - pole_time
            else:
                # Use grid position as a proxy
                quali_df['QualifyingGapToPole'] = (quali_df['GridPosition'] - 1) * 0.1
            
            # Calculate relative grid position
            quali_df['RelativeGridPosition'] = quali_df['GridPosition'].rank(method='min')
            
            # Add race info
            quali_df['Year'] = year
            quali_df['Round'] = round_num
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, quali_df)
            
            return quali_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Ergast API: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error processing Ergast API data: {e}")
            return pd.DataFrame()
    
    def get_race_results(
        self, 
        year: int, 
        round_num: int, 
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get race results from Ergast API.
        
        Args:
            year: Season year
            round_num: Round number
            force_refresh: Whether to bypass cache
            
        Returns:
            DataFrame with race results
        """
        # Check cache first
        if not force_refresh:
            cache_key = f"ergast_race_{year}_{round_num}"
            cached_results = self.data_cache.get(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        logger.info(f"Fetching race data from Ergast API for {year}, round {round_num}")
        
        # Construct the API URL
        endpoint = f"{year}/{round_num}/results.json"
        url = urljoin(self.base_url, endpoint)
        
        try:
            # Make the API request
            response = requests.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            
            # Check if there is race data
            race_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            if not race_data:
                logger.warning(f"No race data available for {year}, round {round_num}")
                return pd.DataFrame()
            
            # Extract race results
            race_results = race_data[0].get('Results', [])
            if not race_results:
                logger.warning(f"No race results available for {year}, round {round_num}")
                return pd.DataFrame()
            
            # Create a DataFrame from the race results
            race_data_list = []
            
            for result in race_results:
                driver_info = result.get('Driver', {})
                constructor_info = result.get('Constructor', {})
                
                # Extract timing info
                time_str = result.get('Time', {}).get('time', '') if 'Time' in result else ''
                
                # Create entry
                entry = {
                    'DriverNumber': result.get('number', ''),
                    'Driver': f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                    'DriverCode': driver_info.get('code', ''),
                    'TeamName': constructor_info.get('name', ''),
                    'RacePosition': int(result.get('position', 0)),
                    'GridPosition': int(result.get('grid', 0)),
                    'Status': result.get('status', ''),
                    'Points': float(result.get('points', 0)),
                    'TimeString': time_str
                }
                
                # Process time intervals
                if 'Time' in result and 'millis' in result['Time']:
                    entry['TimeMillis'] = int(result['Time']['millis'])
                
                # Process fastest lap if available
                if 'FastestLap' in result:
                    fastest_lap = result['FastestLap']
                    entry['FastestLapRank'] = int(fastest_lap.get('rank', 0))
                    
                    if 'Time' in fastest_lap:
                        lap_time = fastest_lap['Time'].get('time', '')
                        entry['FastestLapTimeString'] = lap_time
                        entry['FastestLap_sec'] = self._convert_time_to_seconds(lap_time)
                
                race_data_list.append(entry)
            
            # Convert to DataFrame
            race_df = pd.DataFrame(race_data_list)
            
            # Calculate intervals
            if 'TimeMillis' in race_df.columns and not race_df['TimeMillis'].isna().all():
                # Find winner's time
                winner_time = race_df.loc[race_df['RacePosition'] == 1, 'TimeMillis'].iloc[0]
                # Calculate intervals in seconds
                race_df['Interval_sec'] = (race_df['TimeMillis'] - winner_time) / 1000.0
                # Set winner's interval to 0
                race_df.loc[race_df['RacePosition'] == 1, 'Interval_sec'] = 0.0
            else:
                # Use position as a proxy for interval (~2 seconds per position)
                race_df['Interval_sec'] = (race_df['RacePosition'] - 1) * 2.0
            
            # Add race info
            race_df['Year'] = year
            race_df['Round'] = round_num
            
            # Cache results
            if not force_refresh:
                self.data_cache.set(cache_key, race_df)
            
            return race_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Ergast API: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error processing Ergast API data: {e}")
            return pd.DataFrame()
    
    def _convert_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        Convert time string to seconds.
        
        Args:
            time_str: Time string in the format 'MM:SS.sss'
            
        Returns:
            Time in seconds or None if invalid
        """
        if not time_str:
            return None
        
        try:
            # Check for MM:SS.sss format
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                # Just seconds
                return float(time_str)
        except (ValueError, IndexError):
            return None