# holomind/__init__.py

"""
HoloMind: A simple CPU-based neural network framework.

This package provides basic building blocks for creating and training neural networks.
"""

# Importing core components
from .layers import Dense  # Fully connected layer
from .optimizers import SGD  # Stochastic Gradient Descent optimizer
from .models import Model  # Model class to manage layers and training
from .loss import MeanSquaredError  # Mean Squared Error loss function
# from .autograd import Autograd  # Uncomment when autograd is implemented
# from .utils import some_util_function  # Uncomment when utility functions are added

# Exposing the main classes and functions
__all__ = [
    'Dense',
    'SGD',
    'Model',
    'MeanSquaredError',
    # 'Autograd',  # Uncomment when autograd is implemented
    # 'some_util_function'  # Uncomment when utility functions are added
]