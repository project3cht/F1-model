# config/config.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import os

@dataclass
class ModelConfig:
    model_type: str
    params: Dict
    features: List[str]
    
@dataclass
class DataConfig:
    historical_data_path: str
    sample_size: int
    validation_split: float
    
@dataclass
class PredictionConfig:
    default_safety_car_prob: float
    default_rain_prob: float
    simulation_runs: int
    
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    prediction: PredictionConfig
    log_level: str
    output_dir: str
    
def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        # Return default config if file not found
        default_config = {
            'model': {
                'model_type': 'ensemble',
                'params': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
                'features': ['GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
            },
            'data': {
                'historical_data_path': 'data/historical_races.csv',
                'sample_size': 1000,
                'validation_split': 0.2
            },
            'prediction': {
                'default_safety_car_prob': 0.6,
                'default_rain_prob': 0.0,
                'simulation_runs': 1000
            },
            'log_level': 'INFO',
            'output_dir': 'results'
        }
        
        # Create the config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write the default config to the file
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        # Convert dictionaries to dataclasses
        model_config = ModelConfig(**default_config['model'])
        data_config = DataConfig(**default_config['data'])
        prediction_config = PredictionConfig(**default_config['prediction'])
        
        return Config(
            model=model_config,
            data=data_config,
            prediction=prediction_config,
            log_level=default_config.get('log_level', 'INFO'),
            output_dir=default_config.get('output_dir', 'results')
        )
    
    try:    
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dictionaries to dataclasses
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        prediction_config = PredictionConfig(**config_dict['prediction'])
        
        return Config(
            model=model_config,
            data=data_config,
            prediction=prediction_config,
            log_level=config_dict.get('log_level', 'INFO'),
            output_dir=config_dict.get('output_dir', 'results')
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default config if loading fails
        model_config = ModelConfig(
            model_type='ensemble',
            params={'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            features=['GridPosition', 'TeamPerformanceFactor', 'DriverPerformanceFactor']
        )
        data_config = DataConfig(
            historical_data_path='data/historical_races.csv',
            sample_size=1000,
            validation_split=0.2
        )
        prediction_config = PredictionConfig(
            default_safety_car_prob=0.6,
            default_rain_prob=0.0,
            simulation_runs=1000
        )
        
        return Config(
            model=model_config,
            data=data_config,
            prediction=prediction_config,
            log_level='INFO',
            output_dir='results'
        )