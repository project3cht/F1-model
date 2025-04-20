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
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
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