# models/predictor.py
"""
Simple prediction model for F1 race results.
This is a simplified version of the predictor to make the project run.
"""
import numpy as np
import pandas as pd
import os
import logging
from utils.constants import TEAM_PERFORMANCE, DRIVER_PERFORMANCE, ROOKIES, DRIVERS

class AdvancedF1Predictor:
    """Simple F1 race prediction model."""
    
    def __init__(self, name="SimplePredictor", model_dir="models"):
        """Initialize F1 predictor."""
        self.name = name
        self.model_dir = model_dir
        self.team_performance = TEAM_PERFORMANCE
        self.driver_performance = DRIVER_PERFORMANCE
        self.rookies = ROOKIES
        self.drivers_teams = DRIVERS
        self.logger = logging.getLogger(f'f1_prediction.{name}')
        
        # Basic features
        self.position_features = ['GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
        self.interval_features = ['Position', 'GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def extract_features(self, quali_data, driver_stats=None, team_stats=None):
        """Extract features from qualifying data for prediction."""
        data = quali_data.copy()
        
        # Add team column if not present
        if 'Team' not in data.columns:
            data['Team'] = data['Driver'].map(lambda x: self.drivers_teams.get(x, 'Unknown'))
        
        # Create feature matrix
        features = pd.DataFrame()
        features['Driver'] = data['Driver']
        features['Team'] = data['Team']
        
        # Add grid position (from 'Grid' or 'GridPosition')
        if 'Grid' in data.columns:
            features['GridPosition'] = data['Grid']
        elif 'GridPosition' in data.columns:
            features['GridPosition'] = data['GridPosition']
        else:
            # If no grid position, create a random one
            features['GridPosition'] = range(1, len(data) + 1)
        
        # Add performance factors
        features['TeamPerformanceFactor'] = features['Team'].map(
            lambda team: self.team_performance.get(team, 1.0)
        )
        features['DriverPerformanceFactor'] = features['Driver'].map(
            lambda driver: self.driver_performance.get(driver, 1.0)
        )
        
        # Add rookie flag
        features['IsRookie'] = features['Driver'].apply(
            lambda driver: 1 if driver in self.rookies else 0
        )
        
        # Add track if present
        if 'Track' in data.columns:
            features['Track'] = data['Track']
        
        # Add any additional features from quali_data if they exist
        for col in ['QualifyingTime', 'QualifyingGapToPole', 'Q1', 'Q2', 'Q3']:
            if col in data.columns:
                features[col] = data[col]
        
        return features
    
    def train(self, historical_data):
        """
        Train the prediction models using historical data.
        This is a simplified implementation that doesn't actually train ML models.
        """
        self.logger.info("Training simplified prediction model")
        
        # In a real implementation, this would train ML models
        # For our simplified version, we just log that it was called
        self.logger.info("'Training' completed - using rule-based prediction instead")
        
        return self
    
    def predict(self, quali_data, driver_stats=None, team_stats=None, safety_car_prob=0.6, 
                rain_prob=0.0, track_name=None):
        """Predict race results based on qualifying data using rule-based approach."""
        # Extract features
        features = self.extract_features(quali_data, driver_stats, team_stats)
        
        # Make predictions using rule-based approach
        prediction_df = self._predict_with_factors(features, safety_car_prob, rain_prob)
        
        return prediction_df
    
    def _predict_with_factors(self, features, safety_car_prob=0.6, rain_prob=0.0):
        """Make predictions using performance factors."""
        predictions = features.copy()
        
        # Initialize predicted position with grid position
        predictions['PredictedPosition'] = predictions['GridPosition'].copy()
        
        for idx, row in predictions.iterrows():
            driver = row['Driver']
            team = row['Team']
            grid_pos = row['GridPosition']
            
            # Get performance factors
            driver_factor = self.driver_performance.get(driver, 1.0)
            team_factor = self.team_performance.get(team, 1.0)
            grid_factor = 0.99 + (grid_pos / 100)
            
            # Race day randomness (midfield has more variance)
            if 5 <= grid_pos <= 15:
                random_factor = np.random.normal(1.0, 0.05)
            else:
                random_factor = np.random.normal(1.0, 0.03)
            
            # Safety car effect
            safety_car_effect = 1.0
            if np.random.random() < safety_car_prob:
                safety_car_effect = 1.0 - 0.02 * max(0, grid_pos - 5)
                safety_car_effect = max(0.9, safety_car_effect)
            
            # Rain effect
            rain_effect = 1.0
            if np.random.random() < rain_prob:
                if driver_factor < 0.99:
                    rain_effect = 0.95  # Top drivers even better in rain
                elif driver in self.rookies:
                    rain_effect = 1.05  # Rookies struggle more in rain
            
            # Combined effect
            combined_factor = (
                driver_factor * 
                team_factor * 
                grid_factor * 
                random_factor * 
                safety_car_effect * 
                rain_effect
            )
            
            # Update predicted position
            predictions.loc[idx, 'PredictedPosition'] = grid_pos * combined_factor
        
        # Sort and assign final positions
        predictions = predictions.sort_values('PredictedPosition').reset_index(drop=True)
        predictions['Position'] = range(1, len(predictions) + 1)
        
        # Calculate intervals
        self._calculate_intervals(predictions)
        
        # Calculate points
        self._calculate_points(predictions)
        
        # Ensure all necessary columns exist
        if 'Grid' not in predictions.columns:
            predictions['Grid'] = predictions['GridPosition']
        
        # Return sorted by Position
        return predictions.sort_values('Position').reset_index(drop=True)
    
    def _calculate_intervals(self, predictions):
        """Calculate time intervals between drivers."""
        for idx, row in predictions.iterrows():
            position = row['Position']
            
            if position == 1:
                predictions.loc[idx, 'IntervalSeconds'] = 0.0
                predictions.loc[idx, 'Interval'] = "WINNER"
            else:
                pos_diff = position - 1
                base_interval = pos_diff * 2.0
                interval = base_interval * np.random.uniform(0.8, 1.2)
                predictions.loc[idx, 'IntervalSeconds'] = interval
                predictions.loc[idx, 'Interval'] = f"+{interval:.3f}s"
        
        # Also create 'Interval (s)' for compatibility with visualizations
        predictions['Interval (s)'] = predictions['IntervalSeconds']
    
    def _calculate_points(self, predictions):
        """Calculate points for each driver."""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        predictions['Points'] = predictions['Position'].map(lambda pos: points_system.get(pos, 0))
    
    def save_models(self, position_filename=None, interval_filename=None):
        """
        Save trained models to disk.
        This is a simplified implementation that doesn't actually save models.
        """
        self.logger.info("Model saving called - no actual models to save in simplified version")
        
        # Return some dummy paths for compatibility
        position_path = os.path.join(self.model_dir, "dummy_position_model.joblib")
        interval_path = os.path.join(self.model_dir, "dummy_interval_model.joblib")
        
        return position_path, interval_path
    
    def load_models(self, position_path, interval_path=None):
        """
        Load trained models from disk.
        This is a simplified implementation that doesn't actually load models.
        """
        self.logger.info("Model loading called - using simplified rule-based prediction instead")
        
        # No real models to load in this simplified implementation
        return self