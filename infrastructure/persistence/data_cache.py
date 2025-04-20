# infrastructure/persistence/data_cache.py
"""
Data cache implementation for API results.

This module provides caching functionality to store and retrieve API results.
"""

import os
import pickle
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("f1_prediction.persistence.data_cache")

class DataCache:
    """
    Cache for storing and retrieving data to reduce API calls.
    
    Implements disk-based caching with TTL (time to live) and memory cache.
    """
    
    def __init__(
        self, 
        cache_dir: str = "cache/data",
        ttl_hours: int = 24,
        enable_memory_cache: bool = True
    ):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory for storing cache files
            ttl_hours: Cache time-to-live in hours
            enable_memory_cache: Whether to use in-memory caching
        """
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.enable_memory_cache = enable_memory_cache
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
        
        # Memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Data cache initialized: {cache_dir}, TTL: {ttl_hours}h")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found or expired
        """
        # Check memory cache first
        if self.enable_memory_cache:
            with self.lock:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    
                    # Check if entry is expired
                    if datetime.now() - cache_entry['timestamp'] <= self.ttl:
                        logger.debug(f"Memory cache hit: {key}")
                        return cache_entry['data']
                    else:
                        # Remove expired entry
                        del self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pickle")
        
        if os.path.exists(cache_file):
            try:
                # Check file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                
                if datetime.now() - mod_time <= self.ttl:
                    # Read cache file
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    logger.debug(f"Disk cache hit: {key}")
                    
                    # Update memory cache
                    if self.enable_memory_cache:
                        with self.lock:
                            self.memory_cache[key] = {
                                'data': data,
                                'timestamp': mod_time
                            }
                    
                    return data
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
                    logger.debug(f"Removed expired cache: {key}")
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def set(self, key: str, data: Any) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Update memory cache
        if self.enable_memory_cache:
            with self.lock:
                self.memory_cache[key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
        
        # Update disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pickle")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Cache updated: {key}")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
    
    def clear(self, key_prefix: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            key_prefix: Optional key prefix to clear only matching entries
            
        Returns:
            Number of cleared entries
        """
        cleared_count = 0
        
        # Clear memory cache
        if self.enable_memory_cache:
            with self.lock:
                if key_prefix:
                    # Clear entries with matching prefix
                    keys_to_clear = [k for k in self.memory_cache.keys() if k.startswith(key_prefix)]
                    for k in keys_to_clear:
                        del self.memory_cache[k]
                    cleared_count += len(keys_to_clear)
                else:
                    # Clear all entries
                    cleared_count += len(self.memory_cache)
                    self.memory_cache.clear()
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pickle"):
                key = filename[:-7]  # Remove .pickle extension
                
                if not key_prefix or key.startswith(key_prefix):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing cache file {filename}: {e}")
        
        logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of cleaned up entries
        """
        cleanup_count = 0
        
        # Clean up memory cache
        if self.enable_memory_cache:
            with self.lock:
                now = datetime.now()
                keys_to_clear = [
                    k for k, v in self.memory_cache.items()
                    if now - v['timestamp'] > self.ttl
                ]
                
                for k in keys_to_clear:
                    del self.memory_cache[k]
                
                cleanup_count += len(keys_to_clear)
        
        # Clean up disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pickle"):
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    # Check file modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    
                    if datetime.now() - mod_time > self.ttl:
                        # Remove expired cache file
                        os.remove(cache_file)
                        cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Error checking cache file {filename}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} expired cache entries")
        return cleanup_count