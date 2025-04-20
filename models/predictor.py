"""
Advanced ML-based prediction model for F1 race results.

This module provides a sophisticated model for predicting F1 race outcomes
using multi-layered machine learning techniques.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
import sys
import joblib
from models.visualization import ModelTrainingVisualizer
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import (
    ElasticNet,
    Ridge,
    Lasso
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score
)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import (
    StandardScaler, 
    RobustScaler, 
    PolynomialFeatures
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Add parent directory to path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import TEAM_PERFORMANCE, DRIVER_PERFORMANCE, ROOKIES, DRIVERS

class AdvancedF1Predictor:
    """Advanced F1 race prediction model with multi-layered machine learning."""
    
    def __init__(self, name="AdvancedF1Predictor", model_dir="models"):
        """
        Initialize advanced F1 predictor.
        
        Args:
            name (str): Name of the predictor
            model_dir (str): Directory to save/load models
        """
    
        self.name = name
        self.model_dir = model_dir
        self.team_performance = TEAM_PERFORMANCE
        self.driver_performance = DRIVER_PERFORMANCE
        self.rookies = ROOKIES
        self.drivers_teams = DRIVERS
        self.logger = logging.getLogger(f'f1_prediction.{name}')
        
        # Initialize ML models
        self.position_model = None
        self.interval_model = None
        self.position_features = None
        self.interval_features = None
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    # Initialize feature store
        self.feature_store = FeatureStore()
        
        # Register custom features if needed
        self._register_custom_features()
    
    def _register_custom_features(self):
        """Register additional custom features."""
        # Example of registering a custom feature
        self.feature_store.register_feature(
            name="custom_race_features",
            function=self._calculate_custom_race_features,
            dependencies=["basic_features", "grid_features"],
            description="Custom race-specific features",
            tags=["race", "custom"]
        )
    
    def _calculate_custom_race_features(self, data):
        """Calculate custom race-specific features."""
        # Example custom feature calculation
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        # Example: Calculate combined performance factor
        if 'TeamPerformanceFactor' in data.columns and 'DriverPerformanceFactor' in data.columns:
            result['CombinedPerformanceFactor'] = (
                data['TeamPerformanceFactor'] * data['DriverPerformanceFactor']
            )
        
        return result
    
    # models/predictor.py (continued)
    def extract_features(self, quali_data, driver_stats=None, team_stats=None):
        """
        Extract features from qualifying data for prediction.
        
        Args:
            quali_data (DataFrame): Qualifying data
            driver_stats (DataFrame): Historical driver statistics
            team_stats (DataFrame): Historical team statistics
            
        Returns:
            DataFrame: Extracted features for prediction
        """
        # Create a copy of the input data
        data = quali_data.copy()
        
        # Add external stats to the data if provided
        if driver_stats is not None:
            # Merge driver stats directly into data
            common_cols = list(set(data.columns).intersection(set(driver_stats.columns)))
            if common_cols:
                data = pd.merge(data, driver_stats, on=common_cols, how='left', suffixes=('', '_driver_stats'))
        
        if team_stats is not None:
            # Merge team stats directly into data
            common_cols = list(set(data.columns).intersection(set(team_stats.columns)))
            if common_cols:
                data = pd.merge(data, team_stats, on=common_cols, how='left', suffixes=('', '_team_stats'))
        
        # Use feature store to calculate all features
        feature_list = [
            "basic_features",
            "grid_features",
            "qualifying_features",
            "custom_race_features"
        ]
        
        # Add weather features if modeling weather effects
        if hasattr(self, '_modeling_weather') and self._modeling_weather:
            feature_list.append("weather_features")
        
        # Add track features if track is specified
        if 'Track' in data.columns:
            feature_list.append("track_features")
        
        # Get features from store
        features = self.feature_store.get_features(data, feature_list)
        
        # Log feature calculation
        self.logger.info(f"Extracted {len(features.columns)} features for prediction")
        
        return features
    
    def train(self, historical_data, use_hyperopt=True):
        """
        Train the prediction models using historical data.
        
        Args:
            historical_data (DataFrame): Historical race data
            use_hyperopt (bool): Whether to use hyperparameter optimization
            
        Returns:
            self: Trained predictor
        """
        self.logger.info("Training advanced ML models...")
        
        # Make sure the historical data has the necessary columns
        required_columns = ['Driver', 'Team', 'GridPosition', 'Position', 'Interval']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in historical data: {missing_columns}")
        
        # Rename Position to RacePosition for consistency if needed
        if 'Position' in historical_data.columns and 'RacePosition' not in historical_data.columns:
            historical_data = historical_data.rename(columns={'Position': 'RacePosition'})
        
        # Extract features using feature store
        training_features = self.extract_features(historical_data)
        
        # Train position prediction model
        self.logger.info("Training position prediction model...")
        position_model_result = self._train_position_model(training_features, use_hyperopt)
        self.position_model = position_model_result['model']
        self.position_features = position_model_result['features']
        
        # Train interval prediction model
        self.logger.info("Training interval prediction model...")
        interval_model_result = self._train_interval_model(training_features)
        self.interval_model = interval_model_result['model']
        self.interval_features = interval_model_result['features']
        
        self.logger.info("Training complete")
        return self
    
    def predict(self, quali_data, driver_stats=None, team_stats=None, safety_car_prob=0.6, 
                rain_prob=0.0, track_name=None):
        """
        Predict race results based on qualifying data.
        
        Args:
            quali_data (DataFrame): Qualifying data with driver and grid position
            driver_stats (DataFrame): Historical driver statistics (optional)
            team_stats (DataFrame): Historical team statistics (optional)
            safety_car_prob (float): Probability of safety car appearance
            rain_prob (float): Probability of rain
            track_name (str): Name of the track (optional)
            
        Returns:
            DataFrame: Predicted race results
        """
        # Set weather modeling flag to include weather features if rain probability > 0
        self._modeling_weather = rain_prob > 0
        
        # Add track name if provided
        if track_name and 'Track' not in quali_data.columns:
            quali_data = quali_data.copy()
            quali_data['Track'] = track_name
        
        # Extract features using feature store
        features = self.extract_features(quali_data, driver_stats, team_stats)
        
        # Determine prediction mode
        if self.position_model is not None and self.interval_model is not None:
            # ML model-based prediction
            return self._predict_with_ml(features, safety_car_prob, rain_prob)
        else:
            # Factor-based prediction (fallback)
            return self._predict_with_factors(features, safety_car_prob, rain_prob)
    
    def _predict_with_ml(self, features, safety_car_prob=0.6, rain_prob=0.0):
        """
        Make predictions using the trained ML models.
        
        Args:
            features (DataFrame): Feature matrix
            safety_car_prob (float): Safety car probability
            rain_prob (float): Rain probability
            
        Returns:
            DataFrame: Predicted results
        """
        # Prepare features for position prediction
        X_position = features[self.position_features].copy()
        
        # Handle missing features
        for feature in self.position_features:
            if feature not in X_position:
                X_position[feature] = 0.0
        
        # Fill missing values
        for col in X_position.columns:
            if X_position[col].isna().any():
                X_position[col].fillna(X_position[col].median() if not X_position[col].median() is np.nan else 0, inplace=True)
        
        # Apply race day variations based on safety car and rain
        self._apply_race_day_variations(X_position, safety_car_prob, rain_prob)
        
        # Predict positions
        predicted_positions = self.position_model.predict(X_position)
        
        # Create results DataFrame
        results = features.copy()
        results['PredictedPosition'] = predicted_positions
        
        # Sort by predicted position and assign final positions
        results = results.sort_values('PredictedPosition')
        results['Position'] = range(1, len(results) + 1)
        
        # Prepare features for interval prediction
        # Add race positions to features
        results['RacePosition'] = results['Position']
        
        X_interval = results[self.interval_features].copy()
        
        # Handle missing features
        for feature in self.interval_features:
            if feature not in X_interval:
                X_interval[feature] = 0.0
        
        # Fill missing values
        for col in X_interval.columns:
            if X_interval[col].isna().any():
                X_interval[col].fillna(X_interval[col].median() if not X_interval[col].median() is np.nan else 0, inplace=True)
        
        # Predict intervals
        if hasattr(self.interval_model, 'predict'):
            # Standard model
            intervals = self.interval_model.predict(X_interval)
        else:
            # Synthetic model
            intervals = self.interval_model['predict'](X_interval)
        
        # Add intervals to results
        results['IntervalSeconds'] = intervals
        
        # Format intervals
        for idx, row in results.iterrows():
            if row['Position'] == 1:
                results.loc[idx, 'Interval'] = "WINNER"
            else:
                results.loc[idx, 'Interval'] = f"+{row['IntervalSeconds']:.3f}s"
        
        # Calculate points
        self._calculate_points(results)
        
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
    
    def _apply_race_day_variations(self, features, safety_car_prob, rain_prob):
        """
        Apply race day variations to features.
        
        Args:
            features (DataFrame): Feature matrix
            safety_car_prob (float): Safety car probability
            rain_prob (float): Rain probability
            
        Returns:
            None (modifies features in-place)
        """
        # Apply safety car effect
        if np.random.random() < safety_car_prob:
            # Safety car tends to bunch up the field
            # Add a feature to capture this
            if 'GridPosition' in features.columns:
                features['SafetyCarEffect'] = 0.98 + (features['GridPosition'] / 100)
            else:
                features['SafetyCarEffect'] = 0.99
        
        # Apply rain effect
        if np.random.random() < rain_prob:
            # Rain increases randomness and favors skilled drivers
            features['RainEffect'] = features['DriverPerformanceFactor'].apply(
                lambda x: 0.95 if x < 0.99 else 1.05
            )
        
        # Add general race day randomness
        features['RaceDayRandomness'] = np.random.normal(1.0, 0.02, size=len(features))
    
    def _predict_with_factors(self, quali_data, driver_stats=None, team_stats=None, 
                             safety_car_prob=0.6, rain_prob=0.0):
        """
        Make predictions using performance factors (fallback).
        
        Args:
            quali_data (DataFrame): Qualifying data
            driver_stats (DataFrame): Historical driver statistics
            team_stats (DataFrame): Historical team statistics
            safety_car_prob (float): Safety car probability
            rain_prob (float): Rain probability
            
        Returns:
            DataFrame: Predicted results
        """
        # Copy input data to avoid modifying the original
        predictions = quali_data.copy()
        
        # If Team column is missing, add it based on DRIVERS dictionary
        if 'Team' not in predictions.columns:
            predictions['Team'] = predictions['Driver'].apply(
                lambda driver: self.drivers_teams.get(driver, "Unknown Team")
            )
        
        # Initialize predicted positions with grid positions as the baseline
        predictions['PredictedPosition'] = predictions['GridPosition'].copy()
        
        # Apply performance factors to adjust predicted positions
        for idx, row in predictions.iterrows():
            driver = row['Driver']
            team = row['Team']
            grid_pos = row['GridPosition']
            
            # Get base performance factors
            driver_factor = self.driver_performance.get(driver, 1.0)
            team_factor = self.team_performance.get(team, 1.0)
            
            # Adjust for grid position (better grid = slight advantage)
            grid_factor = 0.99 + (grid_pos / 100)  # 0.99 for P1, 1.09 for P10, etc.
            
            # Randomness factor (race day variations)
            # More randomness for midfield positions
            if 5 <= grid_pos <= 15:
                random_factor = np.random.normal(1.0, 0.05)  # higher variation
            else:
                random_factor = np.random.normal(1.0, 0.03)  # lower variation
            
            # Safety car effect (tends to bunch up the field)
            safety_car_effect = 1.0
            if np.random.random() < safety_car_prob:
                # Safety car helps backmarkers more than frontrunners
                safety_car_effect = 1.0 - 0.02 * max(0, grid_pos - 5)
                safety_car_effect = max(0.9, safety_car_effect)  # Max 10% advantage
            
            # Rain effect (increases randomness, helps skilled drivers)
            rain_effect = 1.0
            if np.random.random() < rain_prob:
                # Rain creates more unpredictability
                # Great drivers perform better in rain
                if driver_factor < 0.99:  # Top drivers
                    rain_effect = 0.95  # 5% advantage
                elif driver in self.rookies:
                    rain_effect = 1.05  # 5% disadvantage for rookies
            
            # Final combined factor
            combined_factor = (
                driver_factor * 
                team_factor * 
                grid_factor * 
                random_factor *
                safety_car_effect *
                rain_effect
            )
            
            # Apply to predicted position
            base_position = grid_pos
            adjusted_position = base_position * combined_factor
            
            # Update predicted position
            predictions.loc[idx, 'PredictedPosition'] = adjusted_position
        
        # Sort by predicted position
        predictions = predictions.sort_values('PredictedPosition')
        
        # Assign final positions (1, 2, 3, etc.)
        predictions['Position'] = range(1, len(predictions) + 1)
        
        # Calculate intervals
        self._calculate_intervals(predictions)
        
        # Calculate points
        self._calculate_points(predictions)
        
        # Return sorted predictions
        return predictions.sort_values('Position').reset_index(drop=True)
    
    def _calculate_intervals(self, predictions):
        """
        Calculate time intervals between drivers.
        
        Args:
            predictions (DataFrame): Predictions DataFrame
            
        Returns:
            None (modifies predictions in-place)
        """
        # Calculate intervals based on position differences
        winner_idx = predictions[predictions['Position'] == 1].index[0]
        
        for idx, row in predictions.iterrows():
            position = row['Position']
            
            if position == 1:
                # Winner has zero interval
                predictions.loc[idx, 'IntervalSeconds'] = 0.0
                predictions.loc[idx, 'Interval'] = "WINNER"
            else:
                # Generate a plausible interval
                # Interval increases with position gap, with some randomness
                pos_diff = position - 1  # Difference from winner
                base_interval = pos_diff * 2.0  # ~2 seconds per position
                
                # Add randomness
                interval = base_interval * np.random.uniform(0.8, 1.2)
                
                # Store interval
                predictions.loc[idx, 'IntervalSeconds'] = interval
                predictions.loc[idx, 'Interval'] = f"+{interval:.3f}s"
    
    def _calculate_points(self, predictions):
        """
        Calculate points for each driver.
        
        Args:
            predictions (DataFrame): Predictions DataFrame
            
        Returns:
            None (modifies predictions in-place)
        """
        # F1 points system
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # Calculate points
        predictions['Points'] = predictions['Position'].map(
            lambda pos: points_system.get(pos, 0)
        )
    
    def save_models(self, position_filename=None, interval_filename=None):
        """
        Save trained models to disk.
        
        Args:
            position_filename (str): Filename for position model
            interval_filename (str): Filename for interval model
            
        Returns:
            tuple: Paths to saved model files
        """
        if self.position_model is None or self.interval_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Default filenames
        if position_filename is None:
            position_filename = f"position_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        if interval_filename is None:
            interval_filename = f"interval_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        # Full paths
        position_path = os.path.join(self.model_dir, position_filename)
        interval_path = os.path.join(self.model_dir, interval_filename)
        
        # Save position model
        position_data = {
            'model': self.position_model,
            'features': self.position_features,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(position_data, position_path)
        self.logger.info(f"Saved position model to {position_path}")
        
        # Save interval model
        interval_data = {
            'model': self.interval_model,
            'features': self.interval_features,
            'is_synthetic': hasattr(self.interval_model, 'predict') == False,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(interval_data, interval_path)
        self.logger.info(f"Saved interval model to {interval_path}")
        
        return position_path, interval_path
    
    def load_models(self, position_path, interval_path):
        """
        Load trained models from disk.
        
        Args:
            position_path (str): Path to position model
            interval_path (str): Path to interval model
            
        Returns:
            self: Model with loaded models
        """
        # Load position model
        try:
            position_data = joblib.load(position_path)
            self.position_model = position_data['model']
            self.position_features = position_data['features']
            self.logger.info(f"Loaded position model from {position_path}")
            
            # Log timestamp if available
            if 'timestamp' in position_data:
                self.logger.info(f"Position model trained on: {position_data['timestamp']}")
        except Exception as e:
            self.logger.error(f"Error loading position model: {e}")
            raise
        
        # Load interval model
        try:
            interval_data = joblib.load(interval_path)
            self.interval_model = interval_data['model']
            self.interval_features = interval_data['features']
            self.logger.info(f"Loaded interval model from {interval_path}")
            
            # Log timestamp if available
            if 'timestamp' in interval_data:
                self.logger.info(f"Interval model trained on: {interval_data['timestamp']}")
        except Exception as e:
            self.logger.error(f"Error loading interval model: {e}")
            raise
        
        return self
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data (DataFrame): Test data with actual race results
            
        Returns:
            dict: Evaluation metrics
        """
        if self.position_model is None or self.interval_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Make predictions
        predictions = self.predict(test_data)
        
        # Extract actual positions and intervals
        if 'RacePosition' in test_data.columns:
            actual_positions = test_data['RacePosition']
        elif 'Position' in test_data.columns:
            actual_positions = test_data['Position']
        else:
            self.logger.warning("No actual positions found in test data")
            actual_positions = None
        
        if 'IntervalSeconds' in test_data.columns:
            actual_intervals = test_data['IntervalSeconds']
        elif 'Interval_sec' in test_data.columns:
            actual_intervals = test_data['Interval_sec']
        else:
            # Try to convert text intervals to seconds
            if 'Interval' in test_data.columns:
                try:
                    actual_intervals = test_data['Interval'].apply(
                        lambda x: 0.0 if x == 'WINNER' else float(x.strip('+s')) if isinstance(x, str) else np.nan
                    )
                except:
                    self.logger.warning("Could not extract actual intervals from test data")
                    actual_intervals = None
            else:
                self.logger.warning("No actual intervals found in test data")
                actual_intervals = None
        
        # Calculate metrics
        metrics = {}
        
        # Position metrics
        if actual_positions is not None:
            # Merge predictions with actual positions
            merged = pd.merge(
                predictions[['Driver', 'Position']],
                test_data[['Driver', actual_positions.name]],
                on='Driver',
                how='inner'
            )
            
            # Calculate metrics
            position_mae = mean_absolute_error(merged[actual_positions.name], merged['Position'])
            position_rmse = np.sqrt(mean_squared_error(merged[actual_positions.name], merged['Position']))
            
            # Count exact positions correct
            exact_correct = (merged[actual_positions.name] == merged['Position']).sum()
            exact_pct = 100 * exact_correct / len(merged)
            
            # Count positions within 1 place
            within_one = (abs(merged[actual_positions.name] - merged['Position']) <= 1).sum()
            within_one_pct = 100 * within_one / len(merged)
            
            metrics['position'] = {
                'mae': position_mae,
                'rmse': position_rmse,
                'exact_correct': exact_correct,
                'exact_pct': exact_pct,
                'within_one': within_one,
                'within_one_pct': within_one_pct
            }
            
            self.logger.info(f"Position MAE: {position_mae:.2f}, Exact: {exact_pct:.1f}%, Within ±1: {within_one_pct:.1f}%")
        
        # Interval metrics
        if actual_intervals is not None:
            # Merge predictions with actual intervals
            merged = pd.merge(
                predictions[['Driver', 'IntervalSeconds']],
                pd.DataFrame({'Driver': test_data['Driver'], 'ActualInterval': actual_intervals}),
                on='Driver',
                how='inner'
            )
            
            # Calculate metrics
            interval_mae = mean_absolute_error(merged['ActualInterval'], merged['IntervalSeconds'])
            interval_rmse = np.sqrt(mean_squared_error(merged['ActualInterval'], merged['IntervalSeconds']))
            
            metrics['interval'] = {
                'mae': interval_mae,
                'rmse': interval_rmse
            }
            
            self.logger.info(f"Interval MAE: {interval_mae:.2f}s, RMSE: {interval_rmse:.2f}s")
        
        return metrics
    
    def initialize_visualizer(self, output_dir="visualizations"):
        """
        Initialize the training visualizer for monitoring model training.
        
        Args:
            output_dir (str): Directory to save visualization outputs
            
        Returns:
            self: Updated predictor instance
        """
        self.visualizer = ModelTrainingVisualizer(
            name=f"{self.name}_Visualizer",
            output_dir=output_dir
        )
        self.logger.info(f"Initialized model training visualizer with output to {output_dir}")
        return self

    def _train_position_model_with_visualization(self, training_data, use_hyperopt=True):
        """
        Train position prediction model with visualization of training process.
        
        Args:
            training_data (DataFrame): Training data
            use_hyperopt (bool): Whether to use hyperparameter optimization
            
        Returns:
            dict: Model and features
        """
        # Define base features
        base_features = [
            'GridPosition', 'RelativeGridPosition', 'TeamPerformanceFactor', 
            'DriverPerformanceFactor', 'IsRookie'
        ]
        
        # Add additional features if available
        optional_features = [
            'DriverAvgFinish', 'TeamAvgFinish', 'QualifyingGapToPole', 
            'AvgPositionsGained', 'FrontRowStart', 'DirtySideStart', 
            'BackOfGrid', 'GridPositionSquared', 'GridPositionLog'
        ]
        
        # Include optional features that are available
        features = base_features + [f for f in optional_features if f in training_data.columns]
        
        # Create feature matrix
        X = training_data[features].copy()
        
        # Create target vector
        y = training_data['RacePosition']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a multi-layered model similar to before
        # Layer 1: Feature preprocessing
        preprocessing = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_regression, k=min(len(features), len(X_train))))
        ])
        
        # If using hyperopt, optimize parameters with visualization
        if use_hyperopt and hasattr(self, 'visualizer'):
            # Create base RF model for hyperparameter optimization
            rf_base = RandomForestRegressor(random_state=42)
            
            # Define parameter ranges to try
            rf_params = {
                'n_estimators': np.arange(50, 300, 50),
                'max_depth': np.arange(3, 15, 2)
            }
            
            # Visualize validation curves for key parameters
            for param_name, param_range in rf_params.items():
                self.visualizer.visualize_validation_curve(
                    rf_base, X_train, y_train, 
                    param_name=param_name,
                    param_range=param_range,
                    title=f"Position Model - {param_name}"
                )
        
        # Layer 2: Base models (possibly with optimized parameters)
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=200, 
                max_depth=8, 
                min_samples_split=5,
                random_state=42
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=42
            )),
            ('ridge', Ridge(alpha=1.0, random_state=42))
        ]
        
        # Layer 3: Meta-model
        meta_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
        
        # Create stacked model
        stacked_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=min(5, len(X_train) // 10),  # Adjust CV based on data size
            n_jobs=-1
        )
        
        # Create full pipeline
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', stacked_model)
        ])
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Position Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Create visualizations if visualizer is available
        if hasattr(self, 'visualizer'):
            # Visualize model performance
            self.visualizer.create_model_dashboard(
                model_pipeline, X_train, y_train, X_test, y_test, 
                feature_names=features,
                model_name="Position Model"
            )
            
            # Visualize feature importance
            self.visualizer.visualize_feature_importance(
                model_pipeline, features, title="Position Model - Feature Importance"
            )
            
            # Visualize residuals
            self.visualizer.visualize_residuals(
                model_pipeline, X_test, y_test, title="Position Model - Residuals"
            )
        
        return {
            'model': model_pipeline,
            'features': features
        }

    def _train_interval_model_with_visualization(self, training_data):
        """
        Train interval prediction model with visualization of training process.
        
        Args:
            training_data (DataFrame): Training data
            
        Returns:
            dict: Model and features
        """
        # Check if we have interval data
        if 'Interval' not in training_data.columns and 'IntervalSeconds' not in training_data.columns:
            # Create a synthetic model since we don't have interval data
            return self._create_synthetic_interval_model()
        
        # Convert text intervals to seconds if necessary
        if 'IntervalSeconds' not in training_data.columns:
            training_data['IntervalSeconds'] = training_data['Interval'].apply(
                lambda x: 0.0 if x == 'WINNER' else float(x.strip('+s')) if isinstance(x, str) else np.nan
            )
        
        # Define features for interval prediction
        base_features = [
            'RacePosition', 'GridPosition', 'RelativeGridPosition', 
            'TeamPerformanceFactor', 'DriverPerformanceFactor'
        ]
        
        # Add additional features if available
        optional_features = [
            'QualifyingGapToPole', 'GridPositionSquared', 'FrontRowStart',
            'DirtySideStart', 'BackOfGrid', 'DriverAvgFinish', 'TeamAvgFinish'
        ]
        
        # Include optional features that are available
        features = base_features + [f for f in optional_features if f in training_data.columns]
        
        # Create feature matrix
        X = training_data[features].copy()
        
        # Add derived features
        X['PositionSquared'] = X['RacePosition'] ** 2
        X['PositionLog'] = np.log1p(X['RacePosition'])
        
        # Calculate grid-to-race position delta
        X['GridRacePosDelta'] = X['GridPosition'] - X['RacePosition']
        X['PosDeltaAbs'] = np.abs(X['GridRacePosDelta'])
        
        # Create target vector
        y = training_data['IntervalSeconds']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Update features list with the derived features
        features = list(X.columns)
        
        # If we don't have enough samples, use synthetic model
        if len(X) < 20:
            self.logger.warning("Not enough interval data samples, using synthetic model")
            return self._create_synthetic_interval_model()
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # If using visualizer, optimize parameters
        if hasattr(self, 'visualizer'):
            # Create base RF model for hyperparameter optimization
            rf_base = RandomForestRegressor(random_state=42)
            
            # Define parameter ranges to try
            rf_params = {
                'n_estimators': np.arange(50, 300, 50),
                'max_depth': np.arange(3, 15, 2)
            }
            
            # Visualize validation curves for key parameters
            for param_name, param_range in rf_params.items():
                self.visualizer.visualize_validation_curve(
                    rf_base, X_train, y_train, 
                    param_name=param_name,
                    param_range=param_range,
                    title=f"Interval Model - {param_name}"
                )
        
        # Create a multi-layered model
        # Layer 1: Feature preprocessing
        preprocessing = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])
        
        # Layer 2: Base models
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=200, 
                max_depth=8,
                random_state=42
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )),
            ('svr', SVR(
                kernel='rbf',
                C=10.0,
                epsilon=0.2
            ))
        ]
        
        # Create voting regressor
        voting_model = VotingRegressor(
            estimators=base_models,
            weights=[0.4, 0.4, 0.2]
        )
        
        # Create full pipeline
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', voting_model)
        ])
        
        # Train the model with sample weights (more weight to smaller intervals)
        sample_weights = 1.0 / (y_train + 1.0)  # Higher weight for smaller intervals
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Interval Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Create visualizations if visualizer is available
        if hasattr(self, 'visualizer'):
            # Visualize model performance
            self.visualizer.create_model_dashboard(
                model_pipeline, X_train, y_train, X_test, y_test, 
                feature_names=features,
                model_name="Interval Model"
            )
            
            # Visualize feature importance
            self.visualizer.visualize_feature_importance(
                model_pipeline, features, title="Interval Model - Feature Importance"
            )
            
            # Visualize residuals
            self.visualizer.visualize_residuals(
                model_pipeline, X_test, y_test, title="Interval Model - Residuals"
            )
        
        return {
            'model': model_pipeline,
            'features': features,
            'is_synthetic': False
        }

    def train_with_visualization(self, historical_data, visualize=True, output_dir="visualizations", use_hyperopt=True):
        """
        Train the prediction models using historical data with visualization.
        
        Args:
            historical_data (DataFrame): Historical race data
            visualize (bool): Whether to generate visualizations
            output_dir (str): Directory to save visualizations
            use_hyperopt (bool): Whether to use hyperparameter optimization
            
        Returns:
            self: Trained predictor
        """
        self.logger.info("Training advanced ML models with visualization...")
        
        # Initialize visualizer if needed
        if visualize and not hasattr(self, 'visualizer'):
            self.initialize_visualizer(output_dir)
        
        # Make sure the historical data has the necessary columns
        required_columns = ['Driver', 'Team', 'GridPosition', 'Position', 'Interval']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in historical data: {missing_columns}")
        
        # Rename Position to RacePosition for consistency if needed
        if 'Position' in historical_data.columns and 'RacePosition' not in historical_data.columns:
            historical_data = historical_data.rename(columns={'Position': 'RacePosition'})
        
        # Extract features for training
        training_features = self.extract_features(historical_data)
        
        # Train position prediction model with visualization
        self.logger.info("Training position prediction model...")
        position_model_result = self._train_position_model_with_visualization(training_features, use_hyperopt)
        self.position_model = position_model_result['model']
        self.position_features = position_model_result['features']
        
        # Train interval prediction model with visualization
        self.logger.info("Training interval prediction model...")
        interval_model_result = self._train_interval_model_with_visualization(training_features)
        self.interval_model = interval_model_result['model']
        self.interval_features = interval_model_result['features']
        
        self.logger.info("Training complete with visualizations")
        return self

    def evaluate_with_visualization(self, test_data, output_dir=None):
        """
        Evaluate the model on test data with detailed visualizations.
        
        Args:
            test_data (DataFrame): Test data with actual race results
            output_dir (str): Directory to save visualizations
            
        Returns:
            dict: Evaluation metrics
        """
        if self.position_model is None or self.interval_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Initialize visualizer if needed
        if not hasattr(self, 'visualizer'):
            output_dir = output_dir or "visualizations"
            self.initialize_visualizer(output_dir)
        
        # Make predictions
        predictions = self.predict(test_data)
        
        # Extract actual positions and intervals
        if 'RacePosition' in test_data.columns:
            actual_positions = test_data['RacePosition']
        elif 'Position' in test_data.columns:
            actual_positions = test_data['Position']
        else:
            self.logger.warning("No actual positions found in test data")
            actual_positions = None
        
        # Process for intervals
        if 'IntervalSeconds' in test_data.columns:
            actual_intervals = test_data['IntervalSeconds']
        elif 'Interval_sec' in test_data.columns:
            actual_intervals = test_data['Interval_sec']
        else:
            # Try to convert text intervals to seconds
            if 'Interval' in test_data.columns:
                try:
                    actual_intervals = test_data['Interval'].apply(
                        lambda x: 0.0 if x == 'WINNER' else float(x.strip('+s')) if isinstance(x, str) else np.nan
                    )
                except:
                    self.logger.warning("Could not extract actual intervals from test data")
                    actual_intervals = None
            else:
                self.logger.warning("No actual intervals found in test data")
                actual_intervals = None
        
        # Calculate metrics
        metrics = {}
        
        # Position metrics
        if actual_positions is not None:
            # Merge predictions with actual positions
            merged = pd.merge(
                predictions[['Driver', 'Position']],
                test_data[['Driver', actual_positions.name]],
                on='Driver',
                how='inner'
            )
            
            # Calculate metrics
            position_mae = mean_absolute_error(merged[actual_positions.name], merged['Position'])
            position_rmse = np.sqrt(mean_squared_error(merged[actual_positions.name], merged['Position']))
            
            # Count exact positions correct
            exact_correct = (merged[actual_positions.name] == merged['Position']).sum()
            exact_pct = 100 * exact_correct / len(merged)
            
            # Count positions within 1 place
            within_one = (abs(merged[actual_positions.name] - merged['Position']) <= 1).sum()
            within_one_pct = 100 * within_one / len(merged)
            
            metrics['position'] = {
                'mae': position_mae,
                'rmse': position_rmse,
                'exact_correct': exact_correct,
                'exact_pct': exact_pct,
                'within_one': within_one,
                'within_one_pct': within_one_pct
            }
            
            self.logger.info(f"Position MAE: {position_mae:.2f}, Exact: {exact_pct:.1f}%, Within ±1: {within_one_pct:.1f}%")
            
            # Create position comparison visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatter plot
            ax.scatter(merged[actual_positions.name], merged['Position'], alpha=0.6)
            
            # Add a perfect prediction line
            min_val = min(merged[actual_positions.name].min(), merged['Position'].min())
            max_val = max(merged[actual_positions.name].max(), merged['Position'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add driver labels
            for _, row in merged.iterrows():
                ax.text(row[actual_positions.name] + 0.1, row['Position'], 
                    row['Driver'], fontsize=8, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('Actual Position')
            ax.set_ylabel('Predicted Position')
            ax.set_title('Predicted vs Actual Positions')
            
            # Add stats as text
            stats_text = (f"Position Metrics:\n"
                        f"MAE: {position_mae:.2f}\n"
                        f"RMSE: {position_rmse:.2f}\n"
                        f"Exact: {exact_pct:.1f}%\n"
                        f"Within ±1: {within_one_pct:.1f}%")
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure
            if hasattr(self, 'visualizer'):
                plt.tight_layout()
                filename = "position_prediction_evaluation.png"
                filepath = os.path.join(self.visualizer.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved position evaluation to {filepath}")
            
        # Interval metrics
        if actual_intervals is not None:
            # Merge predictions with actual intervals
            merged = pd.merge(
                predictions[['Driver', 'IntervalSeconds']],
                pd.DataFrame({'Driver': test_data['Driver'], 'ActualInterval': actual_intervals}),
                on='Driver',
                how='inner'
            )
            
            # Calculate metrics
            interval_mae = mean_absolute_error(merged['ActualInterval'], merged['IntervalSeconds'])
            interval_rmse = np.sqrt(mean_squared_error(merged['ActualInterval'], merged['IntervalSeconds']))
            
            metrics['interval'] = {
                'mae': interval_mae,
                'rmse': interval_rmse
            }
            
            self.logger.info(f"Interval MAE: {interval_mae:.2f}s, RMSE: {interval_rmse:.2f}s")
            
            # Create interval comparison visualization
            if hasattr(self, 'visualizer'):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create scatter plot
                ax.scatter(merged['ActualInterval'], merged['IntervalSeconds'], alpha=0.6)
                
                # Add a perfect prediction line
                min_val = min(merged['ActualInterval'].min(), merged['IntervalSeconds'].min())
                max_val = max(merged['ActualInterval'].max(), merged['IntervalSeconds'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add driver labels
                for _, row in merged.iterrows():
                    if row['Driver'] != 'WINNER':  # Skip winner (0 interval)
                        ax.text(row['ActualInterval'] + 0.1, row['IntervalSeconds'], 
                            row['Driver'], fontsize=8, alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel('Actual Interval (seconds)')
                ax.set_ylabel('Predicted Interval (seconds)')
                ax.set_title('Predicted vs Actual Intervals')
                
                # Add stats as text
                stats_text = (f"Interval Metrics:\n"
                            f"MAE: {interval_mae:.2f}s\n"
                            f"RMSE: {interval_rmse:.2f}s")
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Save figure
                plt.tight_layout()
                filename = "interval_prediction_evaluation.png"
                filepath = os.path.join(self.visualizer.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved interval evaluation to {filepath}")
        
        # Generate a combined metrics visualization
        if hasattr(self, 'visualizer'):
            # Create a summary report figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Position metrics
            if 'position' in metrics:
                ax1.bar(['MAE', 'RMSE'], 
                    [metrics['position']['mae'], metrics['position']['rmse']], 
                    color=['blue', 'orange'])
                
                ax1.set_title('Position Prediction Metrics')
                ax1.set_ylabel('Error')
                
                # Add values on top of bars
                for i, v in enumerate([metrics['position']['mae'], metrics['position']['rmse']]):
                    ax1.text(i, v + 0.1, f"{v:.2f}", ha='center')
                
                # Add a horizontal line at the bottom of each bar
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add accuracy info as text
                accuracy_text = (f"Exact Positions: {metrics['position']['exact_pct']:.1f}%\n"
                                f"Within ±1 Position: {metrics['position']['within_one_pct']:.1f}%")
                
                ax1.text(0.5, 0.85, accuracy_text, transform=ax1.transAxes, fontsize=12,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
            
            # Interval metrics
            if 'interval' in metrics:
                ax2.bar(['MAE', 'RMSE'], 
                    [metrics['interval']['mae'], metrics['interval']['rmse']], 
                    color=['blue', 'orange'])
                
                ax2.set_title('Interval Prediction Metrics')
                ax2.set_ylabel('Error (seconds)')
                
                # Add values on top of bars
                for i, v in enumerate([metrics['interval']['mae'], metrics['interval']['rmse']]):
                    ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center')
                
                # Add a horizontal line at the bottom of each bar
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add title
            plt.suptitle('Model Evaluation Metrics Summary', fontsize=16)
            
            # Save figure
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            filename = "model_evaluation_summary.png"
            filepath = os.path.join(self.visualizer.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved evaluation summary to {filepath}")
        
        return metrics
    

# models/predictor.py
from utils.logging_utils import F1PredictionLogger, handle_exceptions
from typing import Dict, List, Optional, Union

logger = F1PredictionLogger('predictor', level='INFO')

class PredictionService:
    """Service for making F1 race predictions."""
    
    @handle_exceptions(logger)
    def predict(self, quali_data: pd.DataFrame, 
                safety_car_prob: float = 0.6,
                rain_prob: float = 0.0) -> pd.DataFrame:
        """
        Predict race results from qualifying data.
        
        Args:
            quali_data: Qualifying data with drivers and grid positions
            safety_car_prob: Probability of safety car appearance
            rain_prob: Probability of rain during race
            
        Returns:
            DataFrame with predicted race results
        """
        logger.info("Making race predictions", 
                   context={'drivers': len(quali_data), 
                           'safety_car_prob': safety_car_prob,
                           'rain_prob': rain_prob})
        
        # Prediction logic here...
        
        return predictions
    
# models/predictor.py
from typing import Dict, List, Optional, Union, Tuple, Callable, TypedDict, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

class DriverStat(TypedDict):
    Driver: str
    AvgFinish: float
    AvgGrid: float
    AvgPositionsGained: float
    FinishingRate: float
    Races: int

class TeamStat(TypedDict):
    Team: str
    AvgFinish: float
    AvgGrid: float
    AvgPoints: float
    TotalPoints: float
    Races: int

class ModelResult(TypedDict):
    model: Union[BaseEstimator, Dict[str, Callable]]
    features: List[str]
    is_synthetic: bool

class AdvancedF1Predictor:
    """Advanced F1 race prediction model with multi-layered machine learning."""
    
    def __init__(self, name: str = "AdvancedF1Predictor", model_dir: str = "models") -> None:
        """Initialize predictor."""
        self.name: str = name
        self.model_dir: str = model_dir
        self.team_performance: Dict[str, float] = {}
        self.driver_performance: Dict[str, float] = {}
        self.rookies: List[str] = []
        self.drivers_teams: Dict[str, str] = {}
        self.position_model: Optional[BaseEstimator] = None
        self.interval_model: Optional[Union[BaseEstimator, Dict[str, Callable]]] = None
        self.position_features: Optional[List[str]] = None
        self.interval_features: Optional[List[str]] = None
    
    def extract_features(self, 
                        quali_data: pd.DataFrame, 
                        driver_stats: Optional[pd.DataFrame] = None, 
                        team_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract features from qualifying data."""
        # Implementation here...
        return features
    
    def train(self, historical_data: pd.DataFrame, use_hyperopt: bool = True) -> 'AdvancedF1Predictor':
        """Train prediction models using historical data."""
        # Implementation here...
        return self
    
    def _train_position_model(self, training_data: pd.DataFrame, use_hyperopt: bool = True) -> ModelResult:
        """Train position prediction model."""
        # Implementation here...
        return {
            'model': model_pipeline,
            'features': features
        }
    
    def _train_interval_model(self, training_data: pd.DataFrame) -> ModelResult:
        """Train interval prediction model."""
        # Implementation here...
        return {
            'model': model_pipeline,
            'features': features,
            'is_synthetic': False
        }
    
    def predict(self, 
               quali_data: pd.DataFrame, 
               driver_stats: Optional[pd.DataFrame] = None, 
               team_stats: Optional[pd.DataFrame] = None, 
               safety_car_prob: float = 0.6, 
               rain_prob: float = 0.0, 
               track_name: Optional[str] = None) -> pd.DataFrame:
        """Predict race results based on qualifying data."""
        # Implementation here...
        return predictions