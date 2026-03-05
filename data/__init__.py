"""Data package for Human Body State Monitoring.

This package contains dataset classes and data loading utilities for the WESAD
wearable stress and affect detection dataset.

Main Components:
- WESAD: PyTorch Dataset class for WESAD data with LOSO support
"""

from .dataset import WESAD

__all__ = ['WESAD']

__version__ = '1.0.0'
