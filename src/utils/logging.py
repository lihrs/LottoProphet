# -*- coding: utf-8 -*-
"""
Logging utilities for LottoProphet
"""

import logging
import os
from typing import Optional
from .config import config


def setup_logging(log_level: Optional[str] = None, 
                  log_file: Optional[str] = None,
                  console_output: bool = True) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    # Get configuration
    log_level = log_level or config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logger
    logger = logging.getLogger('lottoprophet')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if console_output and config.get('logging.console_handler', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.get('logging.file_handler', True):
        log_file = log_file or config.get_log_path()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'lottoprophet.{name}')


class LoggerMixin:
    """
    Mixin class to add logging capability to any class
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        Get logger for this class
        
        Returns:
            Logger instance
        """
        return get_logger(self.__class__.__name__)


# Setup default logging
default_logger = setup_logging()