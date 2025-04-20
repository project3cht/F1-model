"""
Advanced ML-based prediction model for F1 race results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    cross_val_score,
    validation_curve,
    learning_curve
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
from sklearn.base import BaseEstimator, clone

# Add parent directory to path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import TEAM_PERFORMANCE, DRIVER_PERFORMANCE, ROOKIES, DRIVERS
from features.feature_store import FeatureStore
from models.visualization import ModelTrainingVisualizer

class AdvancedF1Predictor:
    """Advanced F1 race prediction model with multi-layered machine learning."""
    
    def __init__(self, name="AdvancedF1Predictor", model_dir="models"):
        """Initialize advanced F1 predictor."""
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
        self.visualizer = None
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize feature store
        self.feature_store = FeatureStore()
        
        # Register custom features if needed
        self._register_custom_features()
    
    def _register_custom_features(self):
        """Register additional custom features."""
        self.feature_store.register_feature(
            name="custom_race_features",
            function=self._calculate_custom_race_features,
            dependencies=["basic_features", "grid_features"],
            description="Custom race-specific features",
            tags=["race", "custom"]
        )
    
    def _calculate_custom_race_features(self, data):
        """Calculate custom race-specific features."""
        result = pd.DataFrame()
        result['Driver'] = data['Driver']
        
        if 'TeamPerformanceFactor' in data.columns and 'DriverPerformanceFactor' in data.columns:
            result['CombinedPerformanceFactor'] = (
                data['TeamPerformanceFactor'] * data['DriverPerformanceFactor']
            )
        
        return result
    def extract_features(self, quali_data, driver_stats=None, team_stats=None):
        """Extract features from qualifying data for prediction."""
        data = quali_data.copy()
        
        # Add external stats to the data if provided
        if driver_stats is not None:
            common_cols = list(set(data.columns).intersection(set(driver_stats.columns)))
            if common_cols:
                data = pd.merge(data, driver_stats, on=common_cols, how='left', suffixes=('', '_driver_stats'))
        
        if team_stats is not None:
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
        
        self.logger.info(f"Extracted {len(features.columns)} features for prediction")
        
        return features
    
    def train(self, historical_data, use_hyperopt=True):
        """Train the prediction models using historical data."""
        self.logger.info("Training advanced ML models...")
        
        # Make sure the historical data has the necessary columns
        required_columns = ['Driver', 'Team', 'GridPosition', 'Position']
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
    
    def _train_position_model(self, training_data, use_hyperopt=True):
        """Train position prediction model."""
        # Define base features
        base_features = [
            'GridPosition', 'TeamPerformanceFactor', 
            'DriverPerformanceFactor', 'IsRookie'
        ]
        
        # Add additional features if available
        optional_features = [
            'FrontRowStart', 'DirtySideStart', 
            'BackOfGrid', 'GridPositionSquared', 'GridPositionLog',
            'QualifyingGapToPole', 'CombinedPerformanceFactor'
        ]
        
        # Include optional features that are available
        features = base_features + [f for f in optional_features if f in training_data.columns]
        
        # Create feature matrix and target vector
        X = training_data[features].copy()
        y = training_data['RacePosition'] if 'RacePosition' in training_data.columns else training_data['Position']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        preprocessing = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])
        
        # Create base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42)),
            ('ridge', Ridge(alpha=1.0, random_state=42))
        ]
        
        # Create stacked model
        stacked_model = StackingRegressor(
            estimators=base_models,
            final_estimator=ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        # Create full pipeline
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', stacked_model)
        ])
        
        # Train model
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Position Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        return {
            'model': model_pipeline,
            'features': features
        }
    
    def _train_interval_model(self, training_data):
        """Train interval prediction model."""
        # Check if we have interval data
        if 'Interval' not in training_data.columns and 'IntervalSeconds' not in training_data.columns:
            return self._create_synthetic_interval_model()
        
        # Convert text intervals to seconds if necessary
        if 'IntervalSeconds' not in training_data.columns:
            training_data['IntervalSeconds'] = training_data['Interval'].apply(
                lambda x: 0.0 if isinstance(x, str) and x == 'WINNER' else (
                    float(x.replace('+', '').replace('s', '')) if isinstance(x, str) else np.nan
                )
            )
        
        # Define features for interval prediction
        base_features = ['RacePosition', 'GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
        
        # Include optional features that are available
        optional_features = ['GridPositionSquared', 'FrontRowStart', 'DirtySideStart', 'BackOfGrid']
        features = base_features + [f for f in optional_features if f in training_data.columns]
        
        # Create feature matrix and target vector
        X = training_data[features].copy()
        y = training_data['IntervalSeconds']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # If insufficient data, use synthetic model
        if len(X) < 20:
            self.logger.warning("Not enough interval data samples, using synthetic model")
            return self._create_synthetic_interval_model()
        
        # Create preprocessing pipeline
        preprocessing = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])
        
        # Create base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
            ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.2))
        ]
        
        # Create voting regressor
        voting_model = VotingRegressor(estimators=base_models, weights=[0.4, 0.4, 0.2])
        
        # Create full pipeline
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', voting_model)
        ])
        
        # Train model
        model_pipeline.fit(X, y)
        
        return {
            'model': model_pipeline,
            'features': features,
            'is_synthetic': False
        }
    
    def _create_synthetic_interval_model(self):
        """Create a synthetic interval prediction model when no interval data is available."""
        self.logger.warning("Creating synthetic interval model due to lack of training data")
        
        def synthetic_predict(X):
            """Synthetic interval prediction based on position differences."""
            intervals = []
            for _, row in X.iterrows():
                position = row['RacePosition'] if 'RacePosition' in row else row['Position']
                if position == 1:
                    interval = 0.0
                else:
                    base_interval = (position - 1) * 1.5
                    noise = np.random.normal(0, 0.5)
                    interval = max(0, base_interval + noise)
                intervals.append(interval)
            return np.array(intervals)
        
        return {
            'model': {'predict': synthetic_predict},
            'features': ['RacePosition', 'Position'],
            'is_synthetic': True
        }
    
    def predict(self, quali_data, driver_stats=None, team_stats=None, safety_car_prob=0.6, 
                rain_prob=0.0, track_name=None):
        """Predict race results based on qualifying data."""
        # Set weather modeling flag
        self._modeling_weather = rain_prob > 0
        
        # Add track name if provided
        if track_name and 'Track' not in quali_data.columns:
            quali_data = quali_data.copy()
            quali_data['Track'] = track_name
        
        # Extract features
        features = self.extract_features(quali_data, driver_stats, team_stats)
        
        # Determine prediction mode
        if self.position_model is not None and self.interval_model is not None:
            return self._predict_with_ml(features, safety_car_prob, rain_prob)
        else:
            return self._predict_with_factors(features, safety_car_prob, rain_prob)
    
    def _predict_with_ml(self, features, safety_car_prob=0.6, rain_prob=0.0):
        """Make predictions using the trained ML models."""
        # Create features for position prediction
        X_position = features[self.position_features].copy()
        
        # Handle missing features and fill NaN values
        for feature in self.position_features:
            if feature not in X_position.columns:
                X_position[feature] = 0.0
        
        for col in X_position.columns:
            if X_position[col].isna().any():
                X_position[col].fillna(X_position[col].median(), inplace=True)
        
        # Apply race day variations
        self._apply_race_day_variations(X_position, safety_car_prob, rain_prob)
        
        # Predict positions
        predicted_positions = self.position_model.predict(X_position)
        
        # Create results DataFrame
        results = features.copy()
        results['PredictedPosition'] = predicted_positions
        results = results.sort_values('PredictedPosition').reset_index(drop=True)
        results['Position'] = range(1, len(results) + 1)
        
        # Prepare features for interval prediction
        results['RacePosition'] = results['Position']
        X_interval = results[self.interval_features].copy()
        
        # Handle missing features and fill NaN values  
        for feature in self.interval_features:
            if feature not in X_interval.columns:
                X_interval[feature] = 0.0
        
        for col in X_interval.columns:
            if X_interval[col].isna().any():
                X_interval[col].fillna(X_interval[col].median(), inplace=True)
        
        # Predict intervals
        if hasattr(self.interval_model, 'predict'):
            intervals = self.interval_model.predict(X_interval)
        else:
            intervals = self.interval_model['predict'](X_interval)
        
        # Add intervals to results
        results['IntervalSeconds'] = intervals
        for idx, row in results.iterrows():
            if row['Position'] == 1:
                results.loc[idx, 'Interval'] = "WINNER"
            else:
                results.loc[idx, 'Interval'] = f"+{row['IntervalSeconds']:.3f}s"
        
        # Calculate points
        self._calculate_points(results)
        
        # Select output columns
        output_columns = ['Position', 'Driver', 'Team', 'GridPosition', 'Interval', 'IntervalSeconds', 'Points']
        for col in ['GapToPole', 'QualifyingTime', 'QualifyingGapToPole']:
            if col in results.columns:
                output_columns.append(col)
        
        available_columns = [col for col in output_columns if col in results.columns]
        return results[available_columns].sort_values('Position').reset_index(drop=True)
    
    def _apply_race_day_variations(self, features, safety_car_prob, rain_prob):
        """Apply race day variations to features."""
        # Safety car effect
        features['SafetyCarEffect'] = 1.0
        if np.random.random() < safety_car_prob:
            if 'GridPosition' in features.columns:
                features['SafetyCarEffect'] = 0.98 + (features['GridPosition'] / 100)
            else:
                features['SafetyCarEffect'] = 0.99
        
        # Rain effect
        features['RainEffect'] = 1.0
        if np.random.random() < rain_prob:
            if 'DriverPerformanceFactor' in features.columns:
                features['RainEffect'] = features['DriverPerformanceFactor'].apply(
                    lambda x: 0.95 if x < 0.99 else 1.05
                )
            else:
                features['RainEffect'] = 1.02
        
        # General randomness
        features['RaceDayRandomness'] = np.random.normal(1.0, 0.02, size=len(features))
    
    def _predict_with_factors(self, features, safety_car_prob=0.6, rain_prob=0.0):
        """Make predictions using performance factors (fallback)."""
        predictions = features.copy()
        
        # Ensure team column exists
        if 'Team' not in predictions.columns:
            predictions['Team'] = predictions['Driver'].apply(
                lambda x: self.drivers_teams.get(x, 'Unknown')
            )
        
        predictions['PredictedPosition'] = predictions['GridPosition'].copy()
        
        for idx, row in predictions.iterrows():
            driver = row['Driver']
            team = row['Team']
            grid_pos = row['GridPosition']
            
            # Get performance factors
            driver_factor = self.driver_performance.get(driver, 1.0)
            team_factor = self.team_performance.get(team, 1.0)
            grid_factor = 0.99 + (grid_pos / 100)
            
            # Race day randomness
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
                    rain_effect = 0.95
                elif driver in self.rookies:
                    rain_effect = 1.05
            
            # Combined effect
            combined_factor = (driver_factor * team_factor * grid_factor * 
                              random_factor * safety_car_effect * rain_effect)
            
            predictions.loc[idx, 'PredictedPosition'] = grid_pos * combined_factor
        
        # Sort and assign final positions
        predictions = predictions.sort_values('PredictedPosition').reset_index(drop=True)
        predictions['Position'] = range(1, len(predictions) + 1)
        
        # Calculate intervals and points
        self._calculate_intervals(predictions)
        self._calculate_points(predictions)
        
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
    
    def _calculate_points(self, predictions):
        """Calculate points for each driver."""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        predictions['Points'] = predictions['Position'].map(lambda pos: points_system.get(pos, 0))
    
    def save_models(self, position_filename=None, interval_filename=None):
        """Save trained models to disk."""
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
        """Load trained models from disk."""
        try:
            position_data = joblib.load(position_path)
            self.position_model = position_data['model']
            self.position_features = position_data['features']
            self.logger.info(f"Loaded position model from {position_path}")
            
            if 'timestamp' in position_data:
                self.logger.info(f"Position model trained on: {position_data['timestamp']}")
        except Exception as e:
            self.logger.error(f"Error loading position model: {e}")
            raise
        
        try:
            interval_data = joblib.load(interval_path)
            self.interval_model = interval_data['model']
            self.interval_features = interval_data['features']
            self.logger.info(f"Loaded interval model from {interval_path}")
            
            if 'timestamp' in interval_data:
                self.logger.info(f"Interval model trained on: {interval_data['timestamp']}")
        except Exception as e:
            self.logger.error(f"Error loading interval model: {e}")
            raise
        
        return self
    
    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        if self.position_model is None or self.interval_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Make predictions
        predictions = self.predict(test_data)
        
        # Extract actual results
        if 'RacePosition' in test_data.columns:
            actual_positions = test_data['RacePosition']
        elif 'Position' in test_data.columns:
            actual_positions = test_data['Position']
        else:
            self.logger.warning("No actual positions found in test data")
            actual_positions = None
        
        if 'IntervalSeconds' in test_data.columns:
            actual_intervals = test_data['IntervalSeconds']
        else:
            self.logger.warning("No actual intervals found in test data")
            actual_intervals = None
        
        # Calculate metrics
        metrics = {}
        
        # Position metrics
        if actual_positions is not None:
            merged = pd.merge(
                predictions[['Driver', 'Position']],
                test_data[['Driver', actual_positions.name]],
                on='Driver',
                how='inner'
            )
            
            position_mae = mean_absolute_error(merged[actual_positions.name], merged['Position'])
            position_rmse = np.sqrt(mean_squared_error(merged[actual_positions.name], merged['Position']))
            exact_correct = (merged[actual_positions.name] == merged['Position']).sum()
            exact_pct = 100 * exact_correct / len(merged)
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
            merged = pd.merge(
                predictions[['Driver', 'IntervalSeconds']],
                pd.DataFrame({'Driver': test_data['Driver'], 'ActualInterval': actual_intervals}),
                on='Driver',
                how='inner'
            )
            
            interval_mae = mean_absolute_error(merged['ActualInterval'], merged['IntervalSeconds'])
            interval_rmse = np.sqrt(mean_squared_error(merged['ActualInterval'], merged['IntervalSeconds']))
            
            metrics['interval'] = {
                'mae': interval_mae,
                'rmse': interval_rmse
            }
            
            self.logger.info(f"Interval MAE: {interval_mae:.2f}s, RMSE: {interval_rmse:.2f}s")
        
        return metrics