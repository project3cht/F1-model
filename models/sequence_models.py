# models/sequence_models.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RaceProgressionModel:
    """
    LSTM model to predict race progression and finishing positions.
    This model captures how positions change throughout the race.
    """
    
    def __init__(self, 
                sequence_length: int = 4, 
                hidden_units: int = 64, 
                model_path: Optional[str] = None) -> None:
        """
        Initialize race progression model.
        
        Args:
            sequence_length: Number of race stages to consider (e.g., start, 25%, 50%, 75%)
            hidden_units: Number of LSTM hidden units
            model_path: Path to pre-trained model
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(self.hidden_units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_units // 2),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Final position prediction
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
    def _prepare_sequences(self, 
                          race_data: pd.DataFrame, 
                          feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data from race progression data.
        
        Args:
            race_data: DataFrame with race progression (multiple position snapshots)
            feature_cols: Feature columns to use
            
        Returns:
            X: Sequence features
            y: Target positions
        """
        sequences = []
        targets = []
        
        # Group by race and driver
        for (race_id, driver), group in race_data.groupby(['RaceId', 'Driver']):
            # Sort by race progress
            group = group.sort_values('RaceProgress')
            
            # Skip if we don't have enough sequence points
            if len(group) < self.sequence_length:
                continue
                
            # Extract features
            features = group[feature_cols].values
            
            # Extract target (final position)
            final_position = group['Position'].iloc[-1]
            
            sequences.append(features)
            targets.append(final_position)
        
        return np.array(sequences), np.array(targets).reshape(-1, 1)
    
    def train(self, 
             race_progression_data: pd.DataFrame,
             feature_cols: List[str],
             epochs: int = 50,
             batch_size: int = 32,
             validation_split: float = 0.2) -> Dict:
        """
        Train the race progression model.
        
        Args:
            race_progression_data: DataFrame with race progression data
            feature_cols: Feature columns to use
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        # Prepare sequences
        X, y = self._prepare_sequences(race_progression_data, feature_cols)
        
        # Scale features and targets
        self.feature_scaler = StandardScaler()
        X_scaled = np.array([self.feature_scaler.fit_transform(seq) for seq in X])
        
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Build model
        self._build_model((X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        return history.history
    
    def predict_race_progression(self, 
                               initial_data: pd.DataFrame, 
                               feature_cols: List[str]) -> pd.DataFrame:
        """
        Predict full race progression and final positions.
        
        Args:
            initial_data: Initial race data (grid positions, etc.)
            feature_cols: Feature columns to use
            
        Returns:
            DataFrame with predicted race progression
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Prepare initial sequence
        initial_sequence = initial_data[feature_cols].values
        initial_sequence = self.feature_scaler.transform(initial_sequence)
        
        # Reshape for LSTM
        sequence = initial_sequence.reshape(1, len(initial_sequence), len(feature_cols))
        
        # Predict final position
        predicted_scaled = self.model.predict(sequence)
        predicted_position = self.target_scaler.inverse_transform(predicted_scaled)
        
        # Add prediction to result
        result = initial_data.copy()
        result['PredictedPosition'] = predicted_position.flatten()
        
        return result
    
    def save(self, directory: str) -> str:
        """
        Save the model and scalers.
        
        Args:
            directory: Directory to save model
            
        Returns:
            Path to saved model
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save Keras model
        model_path = os.path.join(directory, 'lstm_model')
        self.model.save(model_path)
        
        # Save scalers
        scaler_path = os.path.join(directory, 'scalers.joblib')
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, scaler_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.joblib')
        joblib.dump({
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units
        }, metadata_path)
        
        return model_path
    
    def load(self, directory: str) -> 'RaceProgressionModel':
        """
        Load the model and scalers.
        
        Args:
            directory: Directory to load model from
            
        Returns:
            Loaded model
        """
        # Load Keras model
        model_path = os.path.join(directory, 'lstm_model')
        self.model = keras.models.load_model(model_path)
        
        # Load scalers
        scaler_path = os.path.join(directory, 'scalers.joblib')
        scalers = joblib.load(scaler_path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.joblib')
        metadata = joblib.load(metadata_path)
        self.sequence_length = metadata['sequence_length']
        self.hidden_units = metadata['hidden_units']
        
        return self