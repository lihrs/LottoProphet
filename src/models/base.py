# -*- coding: utf-8 -*-
"""
Base model interface for lottery prediction models
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseLotteryModel(ABC):
    """
    Base abstract class for all lottery prediction models
    """
    
    def __init__(self, lottery_type: str):
        """
        Initialize the base model
        
        Args:
            lottery_type: Type of lottery ('ssq' for 双色球, 'dlt' for 大乐透)
        """
        self.lottery_type = lottery_type
        self.is_trained = False
        
    @abstractmethod
    def train(self, data: Any, **kwargs) -> None:
        """
        Train the model with given data
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        pass
        
    @abstractmethod
    def predict(self, **kwargs) -> Tuple[List[int], List[int]]:
        """
        Make predictions
        
        Args:
            **kwargs: Prediction parameters
            
        Returns:
            Tuple of (red_numbers, blue_numbers)
        """
        pass
        
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        pass
        
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model information
        """
        return {
            'lottery_type': self.lottery_type,
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__
        }