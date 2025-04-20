# models/predictor_enhanced.py
from models.predictor import AdvancedF1Predictor
from models.monte_carlo import MonteCarloRaceSimulator
from models.bayesian_models import BayesianRacePredictionModel
from models.sequence_models import RaceProgressionModel
from models.ensemble_predictor import EnsembleF1Predictor
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

class EnhancedF1Predictor:
    """
    Enhanced F1 predictor with advanced ML techniques and uncertainty estimation.
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
            from config.config import load_config
            self.config = load_config(config_path)
            self.logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = None
            self.logger.warning(f"No configuration file found at {config_path}, using defaults")
        
        # Initialize base predictor
        self.base_predictor = AdvancedF1Predictor(name=f"{name}_Base", model_dir=model_dir)
        
        # Initialize advanced components (but don't build until needed)
        self.monte_carlo = None
        self.bayesian_model = None
        self.sequence_model = None
        self.ensemble = None
        
        # Store attributes
        self.name = name
        self.model_dir = model_dir
    
    def train(self, 
             historical_data: pd.DataFrame,
             race_progression_data: Optional[pd.DataFrame] = None,
             use_ensemble: bool = True,
             use_bayesian: bool = False,
             use_sequence: bool = False,
             ensemble_weights: Optional[Dict[str, float]] = None) -> 'EnhancedF1Predictor':
        """
        Train the enhanced predictor with advanced components.
        
        Args:
            historical_data: Historical race data
            race_progression_data: Historical race progression data for sequence model
            use_ensemble: Whether to use ensemble approach
            use_bayesian: Whether to use Bayesian approach
            use_sequence: Whether to use sequence model
            ensemble_weights: Custom weights for ensemble
            
        Returns:
            Trained predictor
        """
        # Train base predictor first
        self.logger.info("Training base predictor")
        self.base_predictor.train(historical_data)
        
        # Train Bayesian model if requested
        if use_bayesian:
            self.logger.info("Training Bayesian model")
            self.bayesian_model = BayesianRacePredictionModel(
                model_type='hierarchical',
                samples=self.config.prediction.simulation_runs if self.config else 2000
            )
            self.bayesian_model.train(historical_data)
        
        # Train sequence model if requested and data available
        if use_sequence and race_progression_data is not None:
            self.logger.info("Training sequence model")
            self.sequence_model = RaceProgressionModel()
            
            # Define sequence features based on available columns
            sequence_features = [
                'GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor'
            ]
            
            # Add optional features if available
            for feature in ['QualifyingGapToPole', 'RaceProgress']:
                if feature in race_progression_data.columns:
                    sequence_features.append(feature)
            
            self.sequence_model.train(
                race_progression_data=race_progression_data,
                feature_cols=sequence_features
            )
        
        # Create ensemble if requested
        if use_ensemble:
            self.logger.info("Creating ensemble predictor")
            
            # Use provided weights or default
            weights = ensemble_weights or {
                'ml_model': 0.7,
                'factor_model': 0.3,
                'bayesian_model': 0.2 if use_bayesian else 0.0,
                'sequence_model': 0.2 if use_sequence else 0.0
            }
            
            # Normalize weights
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                weights = {k: v / weight_sum for k, v in weights.items()}
            
            self.ensemble = EnsembleF1Predictor(
                base_predictor=self.base_predictor,
                ensemble_weights=weights,
                logger=self.logger
            )
            
            # Add additional models if available
            if use_bayesian and self.bayesian_model is not None:
                self.ensemble.bayesian_model = self.bayesian_model
            
            if use_sequence and self.sequence_model is not None:
                self.ensemble.sequence_model = self.sequence_model
            
            # Train ensemble
            self.ensemble.train(historical_data)
        
        # Initialize Monte Carlo simulator
        self.monte_carlo = MonteCarloRaceSimulator(
            base_predictor=self.ensemble if self.ensemble else self.base_predictor,
            n_simulations=self.config.prediction.simulation_runs if self.config else 1000
        )
        
        self.logger.info("Training complete")
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
        Make predictions with advanced features and uncertainty estimates.
        
        Args:
            quali_data: Qualifying data
            driver_stats: Driver statistics
            team_stats: Team statistics
            safety_car_prob: Safety car probability (uses config default if None)
            rain_prob: Rain probability (uses config default if None)
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
        if use_monte_carlo and self.monte_carlo:
            self.logger.info("Running Monte Carlo simulation")
            
            # Run simulation
            simulation_results = self.monte_carlo.simulate_race(
                quali_data=quali_data,
                base_safety_car_prob=safety_car_prob,
                base_rain_prob=rain_prob
            )
            
            # Calculate statistics
            position_probs = self.monte_carlo.calculate_position_probabilities(simulation_results)
            finishing_stats = self.monte_carlo.calculate_finishing_statistics(simulation_results)
            
            # Create deterministic prediction from mean positions
            mean_positions = finishing_stats[['Driver', 'Position_mean']].copy()
            mean_positions = mean_positions.sort_values('Position_mean')
            mean_positions['Position'] = range(1, len(mean_positions) + 1)
            
            # Merge with quali data to get team information
            result = pd.merge(mean_positions, quali_data, on='Driver')
            
            # Add interval estimates (simplified)
            self.base_predictor._calculate_intervals(result)
            
            # Calculate points
            self.base_predictor._calculate_points(result)
            
            if return_uncertainty:
                return {
                    'predictions': result,
                    'position_probabilities': position_probs,
                    'finishing_statistics': finishing_stats,
                    'simulation_results': simulation_results
                }
            else:
                return result
        
        # Use ensemble if available, otherwise base predictor
        if self.ensemble:
            self.logger.info("Using ensemble predictor")
            return self.ensemble.predict(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=safety_car_prob,
                rain_prob=rain_prob
            )
        else:
            self.logger.info("Using base predictor")
            return self.base_predictor.predict(
                quali_data=quali_data,
                driver_stats=driver_stats,
                team_stats=team_stats,
                safety_car_prob=safety_car_prob,
                rain_prob=rain_prob
            )
    
    def predict_with_bayesian(self, 
                             quali_data: pd.DataFrame,
                             return_samples: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Make predictions using Bayesian model with uncertainty estimates.
        
        Args:
            quali_data: Qualifying data
            return_samples: Whether to return position samples
            
        Returns:
            Prediction results, optionally with position samples
        """
        if self.bayesian_model is None:
            raise ValueError("Bayesian model not trained. Call train() with use_bayesian=True first.")
        
        self.logger.info("Making predictions with Bayesian model")
        return self.bayesian_model.predict(quali_data, return_samples=return_samples)
    
    def save_models(self, directory: Optional[str] = None) -> Dict[str, str]:
        """
        Save all models to disk.
        
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
            position_path, interval_path = self.base_predictor.save_models(
                position_filename="position_model.joblib",
                interval_filename="interval_model.joblib"
            )
            saved_paths['base_predictor'] = base_dir
        except Exception as e:
            self.logger.error(f"Error saving base predictor: {e}")
        
        # Save Bayesian model if available
        if self.bayesian_model is not None:
            try:
                bayesian_dir = os.path.join(directory, "bayesian_model")
                self.bayesian_model.save(bayesian_dir)
                saved_paths['bayesian_model'] = bayesian_dir
            except Exception as e:
                self.logger.error(f"Error saving Bayesian model: {e}")
        
        # Save sequence model if available
        if self.sequence_model is not None:
            try:
                sequence_dir = os.path.join(directory, "sequence_model")
                self.sequence_model.save(sequence_dir)
                saved_paths['sequence_model'] = sequence_dir
            except Exception as e:
                self.logger.error(f"Error saving sequence model: {e}")
        
        # Save ensemble if available
        if self.ensemble is not None:
            try:
                ensemble_dir = os.path.join(directory, "ensemble")
                self.ensemble.save(ensemble_dir)
                saved_paths['ensemble'] = ensemble_dir
            except Exception as e:
                self.logger.error(f"Error saving ensemble: {e}")
        
        # Save config if available
        if self.config is not None:
            try:
                import yaml
                config_path = os.path.join(directory, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(self.config.__dict__, f)
                saved_paths['config'] = config_path
            except Exception as e:
                self.logger.error(f"Error saving config: {e}")
        
        self.logger.info(f"Saved all models to {directory}")
        return saved_paths
    
    def load_models(self, directory: str) -> 'EnhancedF1Predictor':
        """
        Load all models from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Loaded predictor
        """
        # Load base predictor
        base_dir = os.path.join(directory, "base_predictor")
        if os.path.exists(base_dir):
            try:
                position_path = os.path.join(base_dir, "position_model.joblib")
                interval_path = os.path.join(base_dir, "interval_model.joblib")
                self.base_predictor.load_models(position_path, interval_path)
                self.logger.info(f"Loaded base predictor from {base_dir}")
            except Exception as e:
                self.logger.error(f"Error loading base predictor: {e}")
        
        # Load Bayesian model if available
        bayesian_dir = os.path.join(directory, "bayesian_model")
        if os.path.exists(bayesian_dir):
            try:
                self.bayesian_model = BayesianRacePredictionModel()
                self.bayesian_model.load(bayesian_dir)
                self.logger.info(f"Loaded Bayesian model from {bayesian_dir}")
            except Exception as e:
                self.logger.error(f"Error loading Bayesian model: {e}")
        
        # Load sequence model if available
        sequence_dir = os.path.join(directory, "sequence_model")
        if os.path.exists(sequence_dir):
            try:
                self.sequence_model = RaceProgressionModel()
                self.sequence_model.load(sequence_dir)
                self.logger.info(f"Loaded sequence model from {sequence_dir}")
            except Exception as e:
                self.logger.error(f"Error loading sequence model: {e}")
        
        # Load ensemble if available
        ensemble_dir = os.path.join(directory, "ensemble")
        if os.path.exists(ensemble_dir):
            try:
                self.ensemble = EnsembleF1Predictor(base_predictor=self.base_predictor)
                self.ensemble.load(ensemble_dir, self.base_predictor)
                self.logger.info(f"Loaded ensemble from {ensemble_dir}")
            except Exception as e:
                self.logger.error(f"Error loading ensemble: {e}")
        
        # Initialize Monte Carlo simulator
        self.monte_carlo = MonteCarloRaceSimulator(
            base_predictor=self.ensemble if self.ensemble else self.base_predictor,
            n_simulations=self.config.prediction.simulation_runs if self.config else 1000
        )
        
        # Load config if available
        config_path = os.path.join(directory, "config.yaml")
        if os.path.exists(config_path):
            try:
                from config.config import Config
                import yaml
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self.config = Config(**config_dict)
                self.logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        
        self.logger.info(f"Loaded all models from {directory}")
        return self