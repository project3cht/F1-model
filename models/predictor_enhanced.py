# models/predictor_enhanced.py
"""
Simplified enhanced predictor that uses the base predictor as fallback.
"""
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# Try to import necessary modules, but provide fallbacks if not available
try:
    from models.predictor import AdvancedF1Predictor
    base_predictor_available = True
except ImportError:
    base_predictor_available = False
    class DummyPredictor:
        def __init__(self, *args, **kwargs):
            pass
    AdvancedF1Predictor = DummyPredictor

class EnhancedF1Predictor:
    """
    Simplified enhanced F1 predictor that falls back to base predictor.
    """
    
    def __init__(self, 
                name: str = "EnhancedF1Predictor",
                model_dir: str = "models",
                config_path: Optional[str] = "config/config.yaml") -> None:
        """
        Initialize enhanced predictor.
        
        Args:
            name: Name of the predictor
            model_dir: Directory to save/load models
            config_path: Path to configuration file
        """
        # Set up logging
        self.logger = logging.getLogger(f'f1_prediction.{name}')
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                from config.config import load_config
                self.config = load_config(config_path)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}")
                self.config = None
        else:
            self.config = None
            self.logger.warning(f"No configuration file found at {config_path}")
        
        # Initialize base predictor
        self.base_predictor = AdvancedF1Predictor(name=f"{name}_Base", model_dir=model_dir)
        
        # Store attributes
        self.name = name
        self.model_dir = model_dir
        
        self.logger.info(f"Initialized {name} with base predictor")
    
    def train(self, historical_data, **kwargs):
        """
        Train the enhanced predictor.
        
        Args:
            historical_data: Historical race data
            
        Returns:
            Self
        """
        self.logger.info("Training base predictor")
        self.base_predictor.train(historical_data)
        
        return self
    
    def predict(self, 
               quali_data: pd.DataFrame, 
               driver_stats: Optional[pd.DataFrame] = None,
               team_stats: Optional[pd.DataFrame] = None,
               safety_car_prob: Optional[float] = None,
               rain_prob: Optional[float] = None,
               use_monte_carlo: bool = False,
               return_uncertainty: bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Make predictions with the enhanced predictor.
        
        Args:
            quali_data: Qualifying data
            driver_stats: Driver statistics
            team_stats: Team statistics
            safety_car_prob: Safety car probability
            rain_prob: Rain probability
            use_monte_carlo: Whether to use Monte Carlo simulation
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Prediction results, optionally with uncertainty information
        """
        # Use defaults from config if not provided
        if safety_car_prob is None and self.config:
            safety_car_prob = self.config.prediction.default_safety_car_prob
        elif safety_car_prob is None:
            safety_car_prob = 0.6
            
        if rain_prob is None and self.config:
            rain_prob = self.config.prediction.default_rain_prob
        elif rain_prob is None:
            rain_prob = 0.0
        
        # Log prediction parameters
        self.logger.info(f"Making predictions with safety_car_prob={safety_car_prob}, rain_prob={rain_prob}")
        
        # Use Monte Carlo simulation if requested
        if use_monte_carlo:
            return self._monte_carlo_prediction(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=safety_car_prob,
                rain_prob=rain_prob,
                return_uncertainty=return_uncertainty
            )
        else:
            # Standard prediction using base predictor
            self.logger.info("Using base predictor for standard prediction")
            return self.base_predictor.predict(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=safety_car_prob,
                rain_prob=rain_prob
            )
    
    def _monte_carlo_prediction(self,
                              quali_data: pd.DataFrame,
                              driver_stats: Optional[pd.DataFrame] = None,
                              team_stats: Optional[pd.DataFrame] = None,
                              safety_car_prob: float = 0.6,
                              rain_prob: float = 0.0,
                              return_uncertainty: bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Run Monte Carlo simulation for predictions with uncertainty.
        
        Args:
            quali_data: Qualifying data
            driver_stats: Driver statistics
            team_stats: Team statistics
            safety_car_prob: Safety car probability
            rain_prob: Rain probability
            return_uncertainty: Whether to return uncertainty information
            
        Returns:
            Prediction results, optionally with uncertainty information
        """
        self.logger.info("Running Monte Carlo simulation")
        
        # Number of simulations to run
        n_sims = 100
        if self.config and hasattr(self.config.prediction, 'simulation_runs'):
            n_sims = self.config.prediction.simulation_runs
        
        # Run multiple simulations with varying parameters
        all_results = []
        
        for i in range(n_sims):
            # Vary safety car and rain probabilities slightly
            sc_prob = min(1.0, max(0.0, safety_car_prob + np.random.normal(0, 0.1)))
            r_prob = min(1.0, max(0.0, rain_prob + np.random.normal(0, 0.1)))
            
            # Make prediction with base predictor
            sim_result = self.base_predictor.predict(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=sc_prob,
                rain_prob=r_prob
            )
            
            # Add simulation ID
            sim_result['SimulationId'] = i
            
            # Store results
            all_results.append(sim_result)
            
            # Log progress occasionally
            if (i+1) % 20 == 0:
                self.logger.info(f"Completed {i+1}/{n_sims} simulations")
        
        # Combine all simulation results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Calculate statistics
        stats = self._calculate_monte_carlo_statistics(combined_results)
        
        # Create final prediction based on average position
        final_prediction = self._create_final_prediction(stats, quali_data)
        
        if return_uncertainty:
            return {
                'predictions': final_prediction,
                'finishing_statistics': stats,
                'simulation_results': combined_results
            }
        else:
            return final_prediction
    
    def _calculate_monte_carlo_statistics(self, simulation_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics from Monte Carlo simulations.
        
        Args:
            simulation_results: Results from all simulations
            
        Returns:
            DataFrame with driver statistics
        """
        # Group by driver and calculate statistics
        driver_stats = simulation_results.groupby('Driver').agg({
            'Position': ['mean', 'std', 'min', 'max'],
            'Points': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        driver_stats.columns = [f"{col[0]}_{col[1]}" for col in driver_stats.columns]
        
        # Calculate win, podium, and points probabilities
        win_counts = simulation_results[simulation_results['Position'] == 1].groupby('Driver').size()
        podium_counts = simulation_results[simulation_results['Position'] <= 3].groupby('Driver').size()
        points_counts = simulation_results[simulation_results['Position'] <= 10].groupby('Driver').size()
        
        # Convert to probabilities
        n_sims = len(simulation_results) // len(driver_stats)
        win_probs = win_counts / n_sims
        podium_probs = podium_counts / n_sims
        points_probs = points_counts / n_sims
        
        # Add to stats
        driver_stats['WinProbability'] = win_probs
        driver_stats['PodiumProbability'] = podium_probs
        driver_stats['PointsProbability'] = points_probs
        
        # Fill NAs with 0 (drivers who never achieved that result)
        driver_stats.fillna(0, inplace=True)
        
        # Reset index to make Driver a column
        driver_stats.reset_index(inplace=True)
        
        return driver_stats
    
    def _create_final_prediction(self, stats: pd.DataFrame, quali_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create final prediction from Monte Carlo statistics.
        
        Args:
            stats: Statistics from Monte Carlo simulations
            quali_data: Original qualifying data
            
        Returns:
            Final prediction DataFrame
        """
        # Sort by mean position
        sorted_stats = stats.sort_values('Position_mean')
        
        # Create results DataFrame
        results = pd.DataFrame()
        results['Driver'] = sorted_stats['Driver']
        results['Position'] = range(1, len(sorted_stats) + 1)
        
        # Merge with quali_data to get team and grid position
        results = pd.merge(results, quali_data[['Driver', 'Team', 'Grid' if 'Grid' in quali_data.columns else 'GridPosition']], 
                         on='Driver', how='left')
        
        # Rename GridPosition to Grid if needed
        if 'GridPosition' in results.columns and 'Grid' not in results.columns:
            results.rename(columns={'GridPosition': 'Grid'}, inplace=True)
        
        # Calculate intervals
        results['IntervalSeconds'] = 0.0  # Initialize all to 0
        
        # Set intervals based on position gap
        for i, row in results.iterrows():
            if row['Position'] == 1:
                results.loc[i, 'Interval'] = "WINNER"
            else:
                # Calculate interval based on position
                pos = int(row['Position'])
                # Simple model: ~2 seconds per position, with randomness
                interval = (pos - 1) * 2 * (0.9 + 0.2 * np.random.random())
                results.loc[i, 'IntervalSeconds'] = interval
                results.loc[i, 'Interval'] = f"+{interval:.3f}s"
        
        # Add 'Interval (s)' for compatibility with visualizations
        results['Interval (s)'] = results['IntervalSeconds']
        
        # Calculate points
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        results['Points'] = results['Position'].map(lambda pos: points_system.get(pos, 0))
        
        return results
    
    def save_models(self, directory: Optional[str] = None) -> Dict[str, str]:
        """
        Save models to disk.
        
        Args:
            directory: Directory to save models (uses model_dir/timestamp if None)
            
        Returns:
            Dictionary of paths to saved models
        """
        # Create directory with timestamp if not provided
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.model_dir, f"{timestamp}_{self.name}")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        saved_paths = {}
        
        # Save base predictor
        try:
            base_dir = os.path.join(directory, "base_predictor")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                
            position_path, interval_path = self.base_predictor.save_models(
                position_filename=os.path.join(base_dir, "position_model.joblib"),
                interval_filename=os.path.join(base_dir, "interval_model.joblib")
            )
            saved_paths['base_predictor'] = base_dir
        except Exception as e:
            self.logger.error(f"Error saving base predictor: {e}")
        
        return saved_paths
    
    def load_models(self, directory: str) -> 'EnhancedF1Predictor':
        """
        Load models from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Self
        """
        # Load base predictor
        base_dir = os.path.join(directory, "base_predictor")
        if os.path.exists(base_dir):
            try:
                position_path = os.path.join(base_dir, "position_model.joblib")
                interval_path = os.path.join(base_dir, "interval_model.joblib")
                
                if os.path.exists(position_path) and os.path.exists(interval_path):
                    self.base_predictor.load_models(position_path, interval_path)
                    self.logger.info(f"Loaded base predictor from {base_dir}")
                else:
                    self.logger.warning(f"Model files not found in {base_dir}")
            except Exception as e:
                self.logger.error(f"Error loading base predictor: {e}")
        else:
            self.logger.warning(f"Base predictor directory not found: {base_dir}")
        
        return self