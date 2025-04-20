# models/ensemble_predictor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
import joblib
import os
import logging

class FactorModelWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for factor-based model to make it compatible with scikit-learn.
    """
    
    def __init__(self, factor_model: Any):
        """
        Initialize wrapper.
        
        Args:
            factor_model: Factor-based model instance
        """
        self.factor_model = factor_model
        self.X_columns = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FactorModelWrapper':
        """
        Fit the model (no-op for factor model).
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        # Store column names for prediction
        self.X_columns = X.columns.tolist()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using factor model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.X_columns)
        
        # Use factor model to predict
        factor_predictions = self.factor_model._predict_with_factors(X)
        
        # Return predicted positions
        return factor_predictions['Position'].values

class EnsembleF1Predictor:
    """
    Ensemble prediction model combining factor-based and ML-based approaches.
    """
    
    def __init__(self, 
                base_predictor: Any,
                ensemble_weights: Optional[Dict[str, float]] = None,
                logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize ensemble predictor.
        
        Args:
            base_predictor: Base prediction model (should have ML and factor methods)
            ensemble_weights: Weights for different models in ensemble
            logger: Logger instance
        """
        self.base_predictor = base_predictor
        
        # Default weights if not provided
        self.ensemble_weights = ensemble_weights or {
            'ml_model': 0.7,
            'factor_model': 0.3,
            'bayesian_model': 0.0,  # Optional
            'sequence_model': 0.0    # Optional
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.ensemble_weights.values())
        if weight_sum > 0:
            self.ensemble_weights = {
                k: v / weight_sum for k, v in self.ensemble_weights.items()
            }
        
        # Set up logging
        self.logger = logger or logging.getLogger('f1_prediction.ensemble')
        
        # Component models
        self.ml_model = None
        self.factor_model = None
        self.bayesian_model = None
        self.sequence_model = None
        self.voting_model = None
    
    def train(self, historical_data: pd.DataFrame) -> 'EnsembleF1Predictor':
        """
        Train the ensemble model.
        
        Args:
            historical_data: Historical race data
            
        Returns:
            Trained model
        """
        self.logger.info("Training ensemble model components")
        
        # Train base predictor if not already trained
        if self.base_predictor.position_model is None:
            self.base_predictor.train(historical_data)
        
        # Create factor model wrapper
        self.factor_model = FactorModelWrapper(self.base_predictor)
        
        # Extract ML model from base predictor
        self.ml_model = self.base_predictor.position_model
        
        # Prepare features and target for ensemble training
        features = self.base_predictor.extract_features(historical_data)
        position_features = features[self.base_predictor.position_features].copy()
        
        # Fill missing values
        for col in position_features.columns:
            if position_features[col].isna().any():
                position_features[col].fillna(position_features[col].median(), inplace=True)
        
        target = historical_data['Position'].values
        
        # Create voting ensemble
        estimators = [
            ('ml_model', self.ml_model),
            ('factor_model', self.factor_model)
        ]
        
        # Add Bayesian model if available and weight > 0
        if (hasattr(self, 'bayesian_model') and 
            self.bayesian_model is not None and 
            self.ensemble_weights.get('bayesian_model', 0) > 0):
            estimators.append(('bayesian_model', self.bayesian_model))
        
        # Add sequence model if available and weight > 0
        if (hasattr(self, 'sequence_model') and 
            self.sequence_model is not None and 
            self.ensemble_weights.get('sequence_model', 0) > 0):
            estimators.append(('sequence_model', self.sequence_model))
        
        # Create weights list in same order as estimators
        weights = [self.ensemble_weights[name] for name, _ in estimators]
        
        # Create voting ensemble
        self.voting_model = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        
        # Train ensemble
        self.logger.info(f"Training voting ensemble with weights: {self.ensemble_weights}")
        self.voting_model.fit(position_features, target)
        
        return self
    
    def predict(self, 
               quali_data: pd.DataFrame,
               driver_stats: Optional[pd.DataFrame] = None,
               team_stats: Optional[pd.DataFrame] = None,
               safety_car_prob: float = 0.6,
               rain_prob: float = 0.0) -> pd.DataFrame:
        """
        Make predictions using ensemble model.
        
        Args:
            quali_data: Qualifying data
            driver_stats: Driver statistics
            team_stats: Team statistics
            safety_car_prob: Safety car probability
            rain_prob: Rain probability
            
        Returns:
            Prediction results
        """
        # Extract features
        features = self.base_predictor.extract_features(
            quali_data, driver_stats, team_stats
        )
        
        # Prepare features for prediction
        X_position = features[self.base_predictor.position_features].copy()
        
        # Handle missing features
        for feature in self.base_predictor.position_features:
            if feature not in X_position:
                X_position[feature] = 0.0
        
        # Fill missing values
        for col in X_position.columns:
            if X_position[col].isna().any():
                X_position[col].fillna(X_position[col].median() if not X_position[col].median() is np.nan else 0, inplace=True)
        
        # Apply race day variations
        self.base_predictor._apply_race_day_variations(X_position, safety_car_prob, rain_prob)
        
        # Make ensemble predictions
        predicted_positions = self.voting_model.predict(X_position)
        
        # Create results DataFrame
        results = features.copy()
        results['PredictedPosition'] = predicted_positions
        
        # Sort by predicted position and assign final positions
        results = results.sort_values('PredictedPosition')
        results['Position'] = range(1, len(results) + 1)
        
        # Calculate intervals using base predictor method
        results['RacePosition'] = results['Position']
        self.base_predictor._calculate_intervals(results)
        
        # Calculate points
        self.base_predictor._calculate_points(results)
        
        # Select and reorder columns for output
        if 'index' in results.columns:
            results = results.drop('index', axis=1)
        
        output_columns = [
            'Position', 'Driver', 'Team', 'GridPosition', 'Interval', 
            'IntervalSeconds', 'Points'
        ]
        
        # Add additional columns if available
        for col in ['GapToPole', 'QualifyingTime', 'QualifyingGapToPole']:
            if col in results.columns:
                output_columns.append(col)
        
        # Select output columns that exist in results
        available_columns = [col for col in output_columns if col in results.columns]
        
        return results[available_columns].sort_values('Position').reset_index(drop=True)
    
    def evaluate_component_models(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the performance of each component model.
        
        Args:
            test_data: Test data with actual results
            
        Returns:
            Evaluation metrics for each component
        """
        # Extract features
        features = self.base_predictor.extract_features(test_data)
        X_position = features[self.base_predictor.position_features].copy()
        
        # Fill missing values
        for col in X_position.columns:
            if X_position[col].isna().any():
                X_position[col].fillna(X_position[col].median() if not X_position[col].median() is np.nan else 0, inplace=True)
        
        # Actual positions
        y_true = test_data['Position'].values
        
        # Initialize metrics
        metrics = {}
        
        # Evaluate ML model
        y_pred_ml = self.ml_model.predict(X_position)
        metrics['ml_model_mae'] = mean_absolute_error(y_true, y_pred_ml)
        
        # Evaluate factor model
        y_pred_factor = self.factor_model.predict(X_position)
        metrics['factor_model_mae'] = mean_absolute_error(y_true, y_pred_factor)
        
        # Evaluate Bayesian model if available
        if hasattr(self, 'bayesian_model') and self.bayesian_model is not None:
            y_pred_bayes = self.bayesian_model.predict(X_position)
            metrics['bayesian_model_mae'] = mean_absolute_error(y_true, y_pred_bayes)
        
        # Evaluate sequence model if available
        if hasattr(self, 'sequence_model') and self.sequence_model is not None:
            y_pred_seq = self.sequence_model.predict(X_position)
            metrics['sequence_model_mae'] = mean_absolute_error(y_true, y_pred_seq)
        
        # Evaluate ensemble model
        y_pred_ensemble = self.voting_model.predict(X_position)
        metrics['ensemble_model_mae'] = mean_absolute_error(y_true, y_pred_ensemble)
        
        return metrics
    
    def plot_model_comparison(self, test_data: pd.DataFrame) -> plt.Figure:
        """
        Plot comparison of component models.
        
        Args:
            test_data: Test data with actual results
            
        Returns:
            Matplotlib figure
        """
        # Get metrics for all models
        metrics = self.evaluate_component_models(test_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract model names and MAE values
        models = []
        mae_values = []
        
        for key, value in metrics.items():
            if key.endswith('_mae'):
                model_name = key.replace('_mae', '')
                models.append(model_name)
                mae_values.append(value)
        
        # Create bar chart
        bars = ax.bar(models, mae_values)
        
        # Highlight ensemble model bar
        for i, model in enumerate(models):
            if model == 'ensemble_model':
                bars[i].set_color('green')
        
        # Add value labels
        for i, v in enumerate(mae_values):
            ax.text(i, v + 0.05, f"{v:.3f}", ha='center')
        
        # Set title and labels
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Mean Absolute Error (lower is better)', fontsize=12)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    # models/ensemble_predictor.py (continued)
    def save(self, directory: str) -> str:
        """
        Save the ensemble model.
        
        Args:
            directory: Directory to save model
            
        Returns:
            Path to saved model
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save voting model
        model_path = os.path.join(directory, 'ensemble_model.joblib')
        joblib.dump(self.voting_model, model_path)
        
        # Save weights
        weights_path = os.path.join(directory, 'ensemble_weights.joblib')
        joblib.dump(self.ensemble_weights, weights_path)
        
        # Only save configuration for base predictor, not the predictor itself
        # since it's large and might be saved separately
        base_config = {
            'position_features': self.base_predictor.position_features,
            'interval_features': self.base_predictor.interval_features
        }
        base_config_path = os.path.join(directory, 'base_config.joblib')
        joblib.dump(base_config, base_config_path)
        
        self.logger.info(f"Saved ensemble model to {directory}")
        return directory
    
    def load(self, directory: str, base_predictor: Any) -> 'EnsembleF1Predictor':
        """
        Load the ensemble model.
        
        Args:
            directory: Directory to load model from
            base_predictor: Base predictor instance
            
        Returns:
            Loaded model
        """
        # Load voting model
        model_path = os.path.join(directory, 'ensemble_model.joblib')
        self.voting_model = joblib.load(model_path)
        
        # Load weights
        weights_path = os.path.join(directory, 'ensemble_weights.joblib')
        self.ensemble_weights = joblib.load(weights_path)
        
        # Set base predictor
        self.base_predictor = base_predictor
        
        # Extract models from voting regressor
        for name, model in self.voting_model.estimators_:
            if name == 'ml_model':
                self.ml_model = model
            elif name == 'factor_model':
                self.factor_model = model
            elif name == 'bayesian_model':
                self.bayesian_model = model
            elif name == 'sequence_model':
                self.sequence_model = model
        
        self.logger.info(f"Loaded ensemble model from {directory}")
        return self