# -*- coding: utf-8 -*-
"""
Configuration management for LottoProphet
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for the application
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or self._get_default_config_path()
        self._config = self._load_default_config()
        self._load_config()
        
    def _get_default_config_path(self) -> str:
        """
        Get default configuration file path
        
        Returns:
            Default config file path
        """
        home_dir = Path.home()
        config_dir = home_dir / '.lottoprophet'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'config.json')
        
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Data paths
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            
            # Model settings
            'model': {
                'lstm_crf': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.5,
                    'bidirectional': True,
                    'use_attention': False,
                    'use_residual': False
                },
                'ml_models': {
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    },
                    'gradient_boosting': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'random_state': 42
                    }
                }
            },
            
            # Training settings
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'validation_split': 0.2
            },
            
            # Data processing
            'data_processing': {
                'sequence_length': 10,
                'feature_engineering': {
                    'use_statistical_features': True,
                    'use_frequency_features': True,
                    'use_pattern_features': True
                }
            },
            
            # UI settings
            'ui': {
                'theme': 'default',
                'window_size': [1200, 800],
                'font_size': 12
            },
            
            # Logging
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_handler': True,
                'console_handler': True
            },
            
            # GPU settings
            'gpu': {
                'use_gpu': True,
                'device_id': 0
            },
            
            # Lottery specific settings
            'lottery': {
                'ssq': {
                    'red_range': [1, 33],
                    'blue_range': [1, 16],
                    'red_count': 6,
                    'blue_count': 1
                },
                'dlt': {
                    'red_range': [1, 35],
                    'blue_range': [1, 12],
                    'red_count': 5,
                    'blue_count': 2
                }
            }
        }
        
    def _load_config(self) -> None:
        """
        Load configuration from file
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._merge_config(self._config, file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load config file {self.config_file}: {e}")
                
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries
        
        Args:
            base: Base configuration dictionary
            update: Update configuration dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def save_config(self) -> None:
        """
        Save current configuration to file
        """
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Failed to save config file {self.config_file}: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.lstm_crf.hidden_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary
        
        Args:
            updates: Dictionary of updates
        """
        self._merge_config(self._config, updates)
        
    def get_data_path(self, lottery_type: str) -> str:
        """
        Get data file path for specific lottery type
        
        Args:
            lottery_type: Lottery type ('ssq' or 'dlt')
            
        Returns:
            Data file path
        """
        data_dir = self.get('data_dir', 'data')
        return os.path.join(data_dir, lottery_type, f'{lottery_type}_history.csv')
        
    def get_model_path(self, lottery_type: str, model_type: str) -> str:
        """
        Get model file path
        
        Args:
            lottery_type: Lottery type ('ssq' or 'dlt')
            model_type: Model type
            
        Returns:
            Model file path
        """
        models_dir = self.get('models_dir', 'models')
        return os.path.join(models_dir, lottery_type, f'{model_type}.pkl')
        
    def get_log_path(self) -> str:
        """
        Get log file path
        
        Returns:
            Log file path
        """
        logs_dir = self.get('logs_dir', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, 'lottoprophet.log')
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()


# Global configuration instance
config = Config()