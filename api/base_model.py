from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import torch
import numpy as np

class PuzzleSolverModel(ABC):
    """Base class for puzzle solver models"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the model"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and prepare for inference"""
        pass
    
    @abstractmethod
    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Solve the puzzle and return the solution
        
        Args:
            puzzle_img: Tensor representing the scrambled puzzle image
            
        Returns:
            Tuple containing:
            - Tensor of the reconstructed image
            - Additional solution information (ordering, etc.)
        """
        pass