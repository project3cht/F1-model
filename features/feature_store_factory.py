# features/feature_store_factory.py
"""
Factory for creating and configuring feature stores.
"""
import yaml
import os
from typing import Optional, Dict, Any
from features.feature_store import FeatureStore
import logging

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
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
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