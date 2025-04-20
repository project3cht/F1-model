# utils/logging_utils.py
import logging
from typing import Dict, Any, Optional
import json
import traceback
from functools import wraps

class F1PredictionLogger:
    """Enhanced logger for F1 prediction project."""
    
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level, logging.INFO))
        
        # Add handlers if not already added
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler('f1_prediction.log')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.info(message)
    
    def error(self, message: str, error: Optional[Exception] = None, 
              context: Optional[Dict[str, Any]] = None):
        """Log error message with exception details and optional context."""
        if error:
            error_details = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            }
            message = f"{message} | Error: {json.dumps(error_details)}"
        
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
            
        self.logger.error(message)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.warning(message)

# Decorator for exception handling
def handle_exceptions(logger):
    """Decorator to handle and log exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    error=e,
                    context={'args': str(args), 'kwargs': str(kwargs)}
                )
                raise
        return wrapper
    return decorator