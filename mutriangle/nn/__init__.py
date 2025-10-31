"""
Neural Network module for the MuTriangle agent.
Contains the model definition and a wrapper for inference and training interface.
"""

from .model import MuTriangleNet
from .network import NeuralNetwork

__all__ = [
    "MuTriangleNet",
    "NeuralNetwork",
]
