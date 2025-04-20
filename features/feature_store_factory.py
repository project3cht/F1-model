# features/feature_store_factory.py
"""
Factory for creating and configuring feature stores.
"""
import os
from typing import Optional, Dict, Any
import logging

class FeatureStore:
    """Simple feature store implementation."""
    
    def __init__(self, cache_dir="cache/features", enable_caching=True, log_level="INFO"):
        """Initialize feature store."""
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        self.logger = logging.getLogger("f1_prediction.feature_store")
        self.logger.setLevel(getattr(logging, log_level))
        
    def get_features(self, data, feature_names=None, use_cache=True):
        """
        Get features from data.
        This is a simplified implementation that just returns the input data.
        """
        return data

class FeatureStoreFactory:
    """Factory for creating configured feature stores."""
    
    @staticmethod
    def create_feature_store(config_path: Optional[str] = None) -> FeatureStore:
        """
        Create a feature store with configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured feature store
        """
        # Default configuration
        config = {
            "cache": {
                "enabled": True,
                "directory": "cache/features",
                "ttl_hours": 24
            },
            "logging": {
                "level": "INFO"
            },
            "features": {
                "default": [
                    "basic_features",
                    "grid_features",
                    "qualifying_features"
                ]
            }
        }
        
        # Try to load configuration if provided
        loaded_config = None
        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                # Merge configs
                if loaded_config:
                    FeatureStoreFactory._deep_update(config, loaded_config)
            except Exception as e:
                logging.warning(f"Failed to load feature store config: {e}")
        
        # Create feature store with configuration
        feature_store = FeatureStore(
            cache_dir=config["cache"]["directory"],
            enable_caching=config["cache"]["enabled"],
            log_level=config["logging"]["level"]
        )
        
        return feature_store
    
    @staticmethod
    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with another nested dictionary.
        
        Args:
            d: Base dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                FeatureStoreFactory._deep_update(d[k], v)
            else:
                d[k] = v
        return d