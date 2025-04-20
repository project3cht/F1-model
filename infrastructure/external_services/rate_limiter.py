# infrastructure/external_services/rate_limiter.py
"""
Rate limiter implementation for API clients.

This module provides rate limiting functionality to prevent exceeding API rate limits.
"""

import time
import threading
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("f1_prediction.external_services.rate_limiter")

class RateLimiter:
    """
    Rate limiter for API calls to prevent exceeding rate limits.
    
    Implements a token bucket algorithm to manage API call rates.
    """
    
    def __init__(
        self, 
        calls_per_hour: int = 300,
        max_burst: Optional[int] = None,
        min_interval: float = 0.05
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_hour: Maximum number of calls allowed per hour
            max_burst: Maximum burst size (defaults to 10% of hourly limit)
            min_interval: Minimum interval between calls in seconds
        """
        self.calls_per_hour = calls_per_hour
        self.tokens_per_second = calls_per_hour / 3600.0
        self.max_tokens = max_burst or max(int(calls_per_hour * 0.1), 1)
        self.min_interval = min_interval
        
        self.tokens = self.max_tokens
        self.last_refill = datetime.now()
        self.last_call_time = datetime.now() - timedelta(seconds=min_interval)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Rate limiter initialized: {calls_per_hour} calls/hour, "
                   f"{self.max_tokens} max burst, {min_interval:.2f}s min interval")
    
    def _refill_tokens(self):
        """Refill token bucket based on elapsed time."""
        with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_refill).total_seconds()
            
            # Add new tokens based on elapsed time
            new_tokens = elapsed * self.tokens_per_second
            self.tokens = min(self.tokens + new_tokens, self.max_tokens)
            
            # Update last refill time
            self.last_refill = now
    
    def wait(self) -> float:
        """
        Wait if necessary to comply with rate limits.
        
        Returns:
            float: Actual time waited in seconds
        """
        with self.lock:
            # Ensure minimum interval between calls
            now = datetime.now()
            since_last_call = (now - self.last_call_time).total