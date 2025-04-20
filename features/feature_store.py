# features/feature_store.py
"""
Feature store for F1 race predictions.

This module provides a centralized feature store for calculating, caching,
and retrieving features for F1 race predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Set, Tuple, Any
import logging
import os
import joblib
from datetime import datetime
import hashlib
import json
from functools import lru_cache

class FeatureDefinition:
    """
    Definition of a feature or feature group in the feature store.
    """
    
    def __init__(self, 
                name: str,
                function: Callable,
                dependencies: List[str] = None,
                input_columns: List[str] = None,
                description: str = "",
                tags: List[str] = None) -> None:
        """
        Initialize a feature definition.
        
        Args:
            name: Name of the feature or feature group
            function: Function that calculates the feature(s)
            dependencies: Names of other features this depends on
            input_columns: Required columns from raw data
            description: Description of the feature
            tags: List of tags for categorizing features
        """
        self.name = name
        self.function = function
        self.dependencies = dependencies or []
        self.input_columns = input_columns or []
        self.description = description
        self.tags = tags or []
        
    def __repr__(self) -> str:
        return f"FeatureDefinition(name='{self.name}', dependencies={self.dependencies})"

class FeatureStore:
    """
    Centralized store for calculating, caching, and retrieving features.
    """
    
    def __init__(self, 
                cache_dir: str = "cache/features",
                enable_caching: bool = True,
                log_level: str = "INFO") -> None:
        """
        Initialize the feature store.
        
        Args:
            cache_dir: Directory to store cached features
            enable_caching: Whether to enable feature caching
            log_level: Logging level
        """
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        
        # Create cache directory if it doesn't exist
        if enable_caching and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Set up logging
        self.logger = logging.getLogger("f1_prediction.feature_store")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Dictionary to store feature definitions
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        
        # Dictionary to store computed features for current session
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        
        # Register built-in features
        self._register_builtin_features()
    
    def _register_builtin_features(self) -> None:
        """Register built-in feature definitions."""
        # Team and driver basic features
        self.register_feature(
            name="basic_features",
            function=self._calculate_basic_features,
            input_columns=["Driver", "GridPosition"],
            description="Basic driver and team features",
            tags=["base", "team", "driver"]
        )
        
        # Grid position features
        self.register_feature(
            name="grid_features",
            function=self._calculate_grid_features,
            dependencies=["basic_features"],
            description="Grid position derived features",
            tags=["grid", "position"]
        )
        
        # Driver statistics features
        self.register_feature(
            name="driver_stats_features",
            function=self._calculate_driver_stats_features,
            dependencies=["basic_features"],
            description="Historical driver statistics features",
            tags=["driver", "statistics", "historical"]
        )
        
        # Team statistics features
        self.register_feature(
            name="team_stats_features",
            function=self._calculate_team_stats_features,
            dependencies=["basic_features"],
            description="Historical team statistics features",
            tags=["team", "statistics", "historical"]
        )
        
        # Qualifying performance features
        self.register_feature(
            name="qualifying_features",
            function=self._calculate_qualifying_features,
            dependencies=["basic_features"],
            input_columns=["QualifyingTime", "Q1", "Q2", "Q3"],
            description="Qualifying performance features",
            tags=["qualifying", "performance"]
        )
        
        # Weather impact features
        self.register_feature(
            name="weather_features",
            function=self._calculate_weather_features,
            dependencies=["basic_features", "driver_stats_features"],
            description="Weather impact features",
            tags=["weather", "external"]
        )
        
        # Track-specific features
        self.register_feature(
            name="track_features",
            function=self._calculate_track_features,
            dependencies=["basic_features"],
            input_columns=["Track"],
            description="Track-specific features",
            tags=["track", "circuit"]
        )
    
    def register_feature(self, 
                        name: str,
                        function: Callable,
                        dependencies: List[str] = None,
                        input_columns: List[str] = None,
                        description: str = "",
                        tags: List[str] = None) -> None:
        """
        Register a new feature definition.
        
        Args:
            name: Name of the feature or feature group
            function: Function that calculates the feature(s)
            dependencies: Names of other features this depends on
            input_columns: Required columns from raw data
            description: Description of the feature
            tags: List of tags for categorizing features
        """
        if name in self.feature_definitions:
            self.logger.warning(f"Overwriting existing feature definition: {name}")
        
        feature_def = FeatureDefinition(
            name=name,
            function=function,
            dependencies=dependencies,
            input_columns=input_columns,
            description=description,
            tags=tags
        )
        
        self.feature_definitions[name] = feature_def
        self.logger.debug(f"Registered feature: {name}")
    
    def get_features(self, 
                    data: pd.DataFrame,
                    feature_names: List[str] = None,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Calculate and return requested features.
        
        Args:
            data: Input data
            feature_names: List of feature names to calculate (None for all)
            use_cache: Whether to use cached features
            
        Returns:
            DataFrame with calculated features
        """
        # If no feature names specified, calculate all
        if feature_names is None:
            feature_names = list(self.feature_definitions.keys())
        
        # Calculate dependency graph and execution order
        execution_order = self._get_execution_order(feature_names)
        
        # Initialize result with input data
        result = data.copy()
        
        # Track computed features in this run
        computed_features = set()
        
        # Calculate each feature in order
        for feature_name in execution_order:
            feature_def = self.feature_definitions[feature_name]
            
            # Check if we already have this feature cached
            cache_key = self._get_cache_key(data, feature_name)
            
            if use_cache and self.enable_caching:
                # Try memory cache first
                if feature_name in self.feature_cache:
                    self.logger.debug(f"Using memory-cached feature: {feature_name}")
                    feature_data = self.feature_cache[feature_name]
                    result = pd.merge(result, feature_data, on=self._get_merge_keys(result, feature_data))
                    computed_features.add(feature_name)
                    continue
                
                # Try disk cache
                cached_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
                if os.path.exists(cached_path):
                    try:
                        self.logger.debug(f"Loading cached feature from disk: {feature_name}")
                        feature_data = joblib.load(cached_path)
                        result = pd.merge(result, feature_data, on=self._get_merge_keys(result, feature_data))
                        # Also store in memory cache
                        self.feature_cache[feature_name] = feature_data
                        computed_features.add(feature_name)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached feature {feature_name}: {e}")
            
            # Calculate the feature
            self.logger.debug(f"Calculating feature: {feature_name}")
            
            # Check if input columns are available
            missing_columns = [col for col in feature_def.input_columns 
                              if col not in result.columns]
            if missing_columns:
                self.logger.warning(
                    f"Missing input columns for feature {feature_name}: {missing_columns}"
                )
                continue
            
            # Check if dependencies are satisfied
            missing_deps = [dep for dep in feature_def.dependencies 
                          if dep not in computed_features]
            if missing_deps:
                self.logger.error(
                    f"Dependencies not met for feature {feature_name}: {missing_deps}"
                )
                continue
            
            # Calculate feature
            try:
                feature_data = feature_def.function(result)
                
                # Ensure the result is a DataFrame
                if not isinstance(feature_data, pd.DataFrame):
                    self.logger.error(
                        f"Feature function for {feature_name} did not return a DataFrame"
                    )
                    continue
                
                # Add to result
                merge_keys = self._get_merge_keys(result, feature_data)
                if not merge_keys:
                    self.logger.error(
                        f"No common columns found for merging feature {feature_name}"
                    )
                    continue
                
                result = pd.merge(result, feature_data, on=merge_keys)
                
                # Cache the feature
                if self.enable_caching:
                    self.feature_cache[feature_name] = feature_data
                    if use_cache:
                        cache_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
                        joblib.dump(feature_data, cache_path)
                        self.logger.debug(f"Cached feature to disk: {feature_name}")
                
                computed_features.add(feature_name)
                
            except Exception as e:
                self.logger.error(f"Error calculating feature {feature_name}: {e}")
        
        return result
    
    def _get_cache_key(self, data: pd.DataFrame, feature_name: str) -> str:
        """
        Generate a cache key for a feature calculation.
        
        Args:
            data: Input data
            feature_name: Name of the feature
            
        Returns:
            Cache key string
        """
        # Get relevant columns based on feature definition
        feature_def = self.feature_definitions[feature_name]
        
        # Get input columns and dependencies
        relevant_cols = feature_def.input_columns.copy()
        
        # Add dependency input columns
        for dep in feature_def.dependencies:
            if dep in self.feature_definitions:
                relevant_cols.extend(self.feature_definitions[dep].input_columns)
        
        # Filter to columns that exist in data
        relevant_cols = [col for col in relevant_cols if col in data.columns]
        
        # If no relevant columns, use basic columns
        if not relevant_cols:
            relevant_cols = ["Driver", "Team", "GridPosition"]
            relevant_cols = [col for col in relevant_cols if col in data.columns]
        
        # Create a subset of data with only relevant columns
        if relevant_cols:
            subset = data[relevant_cols].copy()
        else:
            # Use shape and column count as fallback
            shape_info = {
                "rows": len(data),
                "columns": len(data.columns),
                "column_names": list(data.columns)
            }
            return f"{feature_name}_{hashlib.md5(json.dumps(shape_info).encode()).hexdigest()}"
        
        # Create hash of the subset
        subset_json = subset.to_json(orient="records")
        key_base = hashlib.md5(subset_json.encode()).hexdigest()
        
        return f"{feature_name}_{key_base}"
    
    def _get_merge_keys(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """
        Get common columns to use as merge keys.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            List of common columns for merging
        """
        # Priority columns for merging
        priority_keys = ["Driver", "Team", "RaceId", "CircuitId", "GridPosition"]
        
        # Find common columns that are in priority list
        common_keys = [col for col in priority_keys 
                     if col in df1.columns and col in df2.columns]
        
        # If no priority keys found, use any common columns
        if not common_keys:
            common_keys = list(set(df1.columns).intersection(set(df2.columns)))
        
        return common_keys
    
    def _get_execution_order(self, feature_names: List[str]) -> List[str]:
        """
        Determine the execution order based on dependencies.
        
        Args:
            feature_names: Names of features to calculate
            
        Returns:
            List of feature names in execution order
        """
        # Build dependency graph
        graph = {}
        for name in feature_names:
            if name not in self.feature_definitions:
                self.logger.warning(f"Unknown feature: {name}")
                continue
                
            feature_def = self.feature_definitions[name]
            graph[name] = feature_def.dependencies
        
        # Perform topological sort
        result = []
        temp_marks = set()
        perm_marks = set()
        
        def visit(node):
            if node in perm_marks:
                return
            if node in temp_marks:
                raise ValueError(f"Circular dependency detected in features: {node}")
            
            temp_marks.add(node)
            
            # Visit dependencies
            for dep in graph.get(node, []):
                if dep in graph:
                    visit(dep)
            
            temp_marks.remove(node)
            perm_marks.add(node)
            result.append(node)
        
        # Visit each node
        for node in graph:
            if node not in perm_marks:
                visit(node)
        
        return result
    
    def list_features(self, tags: List[str] = None) -> pd.DataFrame:
        """
        List available features, optionally filtered by tags.
        
        Args:
            tags: List of tags to filter by (None for all)
            
        Returns:
            DataFrame with feature information
        """
        features = []
        
        for name, feature_def in self.feature_definitions.items():
            # Filter by tags if specified
            if tags and not any(tag in feature_def.tags for tag in tags):
                continue
                
            features.append({
                "name": name,
                "description": feature_def.description,
                "dependencies": feature_def.dependencies,
                "input_columns": feature_def.input_columns,
                "tags": feature_def.tags
            })
        
        return pd.DataFrame(features)
    
    def get_feature_dependencies(self, feature_name: str) -> Dict:
        """
        Get detailed dependency information for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with dependency information
        """
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Unknown feature: {feature_name}")
            
        feature_def = self.feature_definitions[feature_name]
        
        # Build dependency tree
        def build_tree(name, visited=None):
            if visited is None:
                visited = set()
                
            if name in visited:
                return {"name": name, "circular": True}
                
            visited.add(name)
            
            if name not in self.feature_definitions:
                return {"name": name, "missing": True}
                
            deps = self.feature_definitions[name].dependencies
            return {
                "name": name,
                "dependencies": [build_tree(dep, visited.copy()) for dep in deps]
            }
        
        return build_tree(feature_name)
    
    def clear_cache(self, feature_names: List[str] = None) -> None:
        """
        Clear feature cache.
        
        Args:
            feature_names: List of feature names to clear (None for all)
        """
        # Clear memory cache
        if feature_names is None:
            self.feature_cache.clear()
            self.logger.info("Cleared all features from memory cache")
        else:
            for name in feature_names:
                if name in self.feature_cache:
                    del self.feature_cache[name]
            self.logger.info(f"Cleared specified features from memory cache: {feature_names}")
        
        # Clear disk cache if enabled
        if not self.enable_caching:
            return
            
        if feature_names is None:
            # Clear all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".joblib"):
                    os.remove(os.path.join(self.cache_dir, filename))
            self.logger.info("Cleared all features from disk cache")
        else:
            # Clear only specified features
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".joblib"):
                    for name in feature_names:
                        if filename.startswith(f"{name}_"):
                            os.remove(os.path.join(self.cache_dir, filename))
            self.logger.info(f"Cleared specified features from disk cache: {feature_names}")
    
    # Built-in feature calculation functions
    def _calculate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic driver and team features."""
        from utils.helpers import DRIVERS, TEAM_PERFORMANCE, DRIVER_PERFORMANCE, ROOKIES
        
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        # Add team if not present
        if 'Team' not in data.columns:
            result['Team'] = data['Driver'].map(lambda driver: DRIVERS.get(driver, "Unknown Team"))
        else:
            result['Team'] = data['Team']
        
        # Add team and driver performance factors
        result['TeamPerformanceFactor'] = result['Team'].map(
            lambda team: TEAM_PERFORMANCE.get(team, 1.0)
        )
        
        result['DriverPerformanceFactor'] = result['Driver'].map(
            lambda driver: DRIVER_PERFORMANCE.get(driver, 1.0)
        )
        
        # Add rookie flag
        result['IsRookie'] = result['Driver'].apply(
            lambda driver: 1 if driver in ROOKIES else 0
        )
        
        return result
    
    def _calculate_grid_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate grid position derived features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        if 'GridPosition' in data.columns:
            grid_pos = data['GridPosition']
            
            # Front row advantage
            result['FrontRowStart'] = (grid_pos <= 2).astype(int)
            
            # Dirty side disadvantage (even grid positions)
            result['DirtySideStart'] = (grid_pos % 2 == 0).astype(int)
            
            # Back of grid
            result['BackOfGrid'] = (grid_pos >= 16).astype(int)
            
            # Grid position squared (for nonlinear effects)
            result['GridPositionSquared'] = grid_pos ** 2
            
            # Log grid position
            result['GridPositionLog'] = np.log1p(grid_pos)
            
            # Grid position ranking
            result['GridRank'] = grid_pos.rank(method='min')
            
            # Grid position quintile (1-5)
            max_grid = grid_pos.max()
            result['GridQuintile'] = pd.qcut(
                grid_pos, 
                q=min(5, max_grid), 
                labels=False, 
                duplicates='drop'
            ) + 1
        
        return result
    
    def _calculate_driver_stats_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate driver statistics features."""
        # In a real implementation, this would load from a database
        # For this example, we'll assume we have some historical data
        
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        # Placeholder values for demo - in real implementation,
        # these would be calculated from historical data
        for driver in result['Driver']:
            # Random stats for demo purposes
            result.loc[result['Driver'] == driver, 'DriverAvgFinish'] = np.random.uniform(5, 15)
            result.loc[result['Driver'] == driver, 'DriverAvgStart'] = np.random.uniform(5, 15)
            result.loc[result['Driver'] == driver, 'AvgPositionsGained'] = np.random.uniform(-3, 3)
            result.loc[result['Driver'] == driver, 'DriverWinRate'] = np.random.uniform(0, 0.3)
            result.loc[result['Driver'] == driver, 'DriverPodiumRate'] = np.random.uniform(0.1, 0.5)
            result.loc[result['Driver'] == driver, 'DriverConsistencyScore'] = np.random.uniform(0.5, 1.0)
        
        return result
    
    def _calculate_team_stats_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate team statistics features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        result['Team'] = data['Team']
        
        # Placeholder values for demo - in real implementation,
        # these would be calculated from historical data
        for team in result['Team'].unique():
            # Random stats for demo purposes
            team_mask = result['Team'] == team
            result.loc[team_mask, 'TeamAvgFinish'] = np.random.uniform(5, 15)
            result.loc[team_mask, 'TeamAvgGrid'] = np.random.uniform(5, 15)
            result.loc[team_mask, 'TeamReliabilityScore'] = np.random.uniform(0.7, 1.0)
            result.loc[team_mask, 'TeamPitStopAvg'] = np.random.uniform(2.0, 4.0)
            result.loc[team_mask, 'TeamDevScore'] = np.random.uniform(0.5, 1.0)
        
        return result
    
    def _calculate_qualifying_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate qualifying performance features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        quali_cols = [col for col in ['QualifyingTime', 'Q1', 'Q2', 'Q3'] 
                     if col in data.columns]
        
        if not quali_cols:
            return result
        
        # If we have qualifying time, calculate gap to pole
        if 'QualifyingTime' in data.columns:
            pole_time = data['QualifyingTime'].min()
            result['QualifyingGapToPole'] = data['QualifyingTime'] - pole_time
            result['QualifyingGapToPolePercent'] = (result['QualifyingGapToPole'] / pole_time) * 100
        
        # Calculate Q1-Q3 improvement
        if 'Q1' in data.columns and 'Q3' in data.columns:
            q1_q3_mask = (~data['Q1'].isna()) & (~data['Q3'].isna())
            result.loc[q1_q3_mask, 'QualiImprovement'] = data.loc[q1_q3_mask, 'Q1'] - data.loc[q1_q3_mask, 'Q3']
        
        # Calculate if set improved times in each session
        if 'Q1' in data.columns and 'Q2' in data.columns:
            q1_q2_mask = (~data['Q1'].isna()) & (~data['Q2'].isna())
            result.loc[q1_q2_mask, 'ImprovedQ1Q2'] = (data.loc[q1_q2_mask, 'Q2'] < data.loc[q1_q2_mask, 'Q1']).astype(int)
        
        if 'Q2' in data.columns and 'Q3' in data.columns:
            q2_q3_mask = (~data['Q2'].isna()) & (~data['Q3'].isna())
            result.loc[q2_q3_mask, 'ImprovedQ2Q3'] = (data.loc[q2_q3_mask, 'Q3'] < data.loc[q2_q3_mask, 'Q2']).astype(int)
        
        return result
    
    def _calculate_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather impact features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        # In a real implementation, these would use actual weather data
        # and historical driver performance in different conditions
        
        # Default weather performance (neutral)
        result['WetWeatherAdvantage'] = 0.0
        
        # Apply driver-specific weather advantages
        # This is simplified - would use historical driver performance in wet conditions
        wet_weather_specialists = [
            'Lewis Hamilton', 'Max Verstappen', 'Fernando Alonso'
        ]
        
        wet_weather_strugglers = [
            'Kimi Antonelli', 'Oliver Bearman', 'Gabriel Bortoleto'
        ]
        
        # Assign advantage/disadvantage scores
        for driver in wet_weather_specialists:
            if driver in result['Driver'].values:
                result.loc[result['Driver'] == driver, 'WetWeatherAdvantage'] = np.random.uniform(0.05, 0.1)
        
        for driver in wet_weather_strugglers:
            if driver in result['Driver'].values:
                result.loc[result['Driver'] == driver, 'WetWeatherAdvantage'] = np.random.uniform(-0.1, -0.05)
        
        # Incorporate rookie disadvantage in wet conditions
        if 'IsRookie' in data.columns:
            rookie_mask = data['IsRookie'] == 1
            result.loc[rookie_mask, 'WetWeatherAdvantage'] -= 0.05
        
        return result
    
    def _calculate_track_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate track-specific features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        # If track not specified, return empty features
        if 'Track' not in data.columns:
            return result
        
        result['Track'] = data['Track']
        
        # In a real implementation, these would use historical data
        # for driver performance at specific tracks
        
        # Example track features (would be based on actual data)
        track_types = {
            'Monaco': 'street',
            'Monza': 'power',
            'Silverstone': 'balanced',
            'Singapore': 'street',
            'Spa': 'power',
            'Suzuka': 'technical'
        }
        
        # Add track type
        if 'Track' in result.columns:
            result['TrackType'] = result['Track'].map(
                lambda t: track_types.get(t, 'unknown')
            )
        
        # Add track-specific features
        for idx, row in result.iterrows():
            # Sample track-specific advantage (random for demo)
            result.loc[idx, 'TrackSpecificAdvantage'] = np.random.normal(0, 0.02)
            
            # Additional logic based on track type and driver characteristics would go here
            if 'TrackType' in result.columns:
                track_type = result.loc[idx, 'TrackType']
                
                # Example: Some drivers perform better on street circuits
                if track_type == 'street' and row['Driver'] in ['Fernando Alonso', 'Lewis Hamilton']:
                    result.loc[idx, 'TrackSpecificAdvantage'] += 0.03
                
                # Example: Some drivers perform better on power circuits
                if track_type == 'power' and row['Driver'] in ['Max Verstappen', 'Charles Leclerc']:
                    result.loc[idx, 'TrackSpecificAdvantage'] += 0.03
        
        return result