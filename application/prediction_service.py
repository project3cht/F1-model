# application/prediction_service.py
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from domain.prediction import Prediction, PredictionId, PredictionResult
from domain.race import Race, RaceId
from domain.circuit import Circuit
from domain.driver import Driver
from models.predictor_enhanced import EnhancedF1Predictor
from features.feature_store_factory import FeatureStoreFactory
from config.config import load_config
from data.fetching import fetch_race_data
from data.processing import calculate_driver_stats, calculate_team_stats
from utils.helpers import DRIVERS

class PredictionService:
    """Application service for F1 race predictions."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize prediction service."""
        self.logger = logging.getLogger('f1_prediction.prediction_service')
        self.config = load_config(config_path)
        self.predictor = EnhancedF1Predictor(config_path=config_path)
        self.feature_store = FeatureStoreFactory.create_feature_store()
        
    def predict_race(self, 
                    race_id: str,
                    circuit: Circuit,
                    quali_data: pd.DataFrame,
                    historical_data: Optional[pd.DataFrame] = None,
                    safety_car_prob: Optional[float] = None,
                    rain_prob: Optional[float] = None,
                    use_monte_carlo: bool = False) -> Prediction:
        """Make predictions for a specific race."""
        self.logger.info(f"Making prediction for race {race_id}")
        
        # Create Race entity
        race = Race.create(
            race_id=race_id,
            circuit=circuit,
            date=datetime.now(),
            season=datetime.now().year
        )
        
        # Set default probabilities from config if not provided
        if safety_car_prob is None:
            safety_car_prob = self.config.prediction.default_safety_car_prob
        if rain_prob is None:
            rain_prob = self.config.prediction.default_rain_prob
        
        # Calculate driver and team statistics if historical data provided
        driver_stats = None
        team_stats = None
        if historical_data is not None:
            driver_stats = calculate_driver_stats(historical_data)
            team_stats = calculate_team_stats(historical_data)
        
        # Make prediction using enhanced predictor
        prediction_results = self.predictor.predict(
            quali_data=quali_data,
            driver_stats=driver_stats,
            team_stats=team_stats,
            safety_car_prob=safety_car_prob,
            rain_prob=rain_prob,
            use_monte_carlo=use_monte_carlo,
            return_uncertainty=use_monte_carlo
        )
        
        # Create Prediction domain object
        prediction = Prediction.create(
            prediction_id=f"pred_{race_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            race=race,
            safety_car_probability=safety_car_prob,
            rain_probability=rain_prob
        )
        
        # Process results based on prediction type
        if use_monte_carlo and isinstance(prediction_results, dict):
            # Monte Carlo results with uncertainty
            results_df = prediction_results['predictions']
            finishing_stats = prediction_results.get('finishing_statistics')
            
            # Add prediction results with uncertainty
            for _, row in results_df.iterrows():
                # Create Driver entity
                driver = Driver.create(
                    driver_id=row['Driver'].replace(' ', '_').lower(),
                    first_name=row['Driver'].split()[0],
                    last_name=' '.join(row['Driver'].split()[1:]),
                    team=row['Team']
                )
                
                # Get confidence score from finishing stats if available
                confidence_score = 0.0
                if finishing_stats is not None:
                    driver_stats = finishing_stats[finishing_stats['Driver'] == row['Driver']]
                    if not driver_stats.empty:
                        confidence_score = 1.0 - (driver_stats['Position_std'].iloc[0] / 10.0)
                
                prediction.add_prediction_result(
                    driver=driver,
                    predicted_position=row['Position'],
                    interval_from_leader=row.get('IntervalSeconds', 0.0),
                    confidence_score=confidence_score
                )
        else:
            # Regular prediction results
            results_df = prediction_results
            for _, row in results_df.iterrows():
                # Create Driver entity
                driver = Driver.create(
                    driver_id=row['Driver'].replace(' ', '_').lower(),
                    first_name=row['Driver'].split()[0],
                    last_name=' '.join(row['Driver'].split()[1:]),
                    team=row['Team']
                )
                
                prediction.add_prediction_result(
                    driver=driver,
                    predicted_position=row['Position'],
                    interval_from_leader=row.get('IntervalSeconds', 0.0)
                )
        
        return prediction
    
    def train_models(self, historical_data: pd.DataFrame) -> None:
        """Train prediction models using historical data."""
        self.logger.info("Training prediction models")
        self.predictor.train(historical_data)
        
    def load_models(self, model_dir: str) -> None:
        """Load pre-trained models."""
        self.logger.info(f"Loading models from {model_dir}")
        self.predictor.load_models(model_dir)
        
    def save_models(self, model_dir: str) -> Dict[str, str]:
        """Save current models."""
        self.logger.info(f"Saving models to {model_dir}")
        return self.predictor.save_models(model_dir)