"""Model package for Human Body State Monitoring.

This package contains all the neural network components for stress and affect
detection from biometric signals using the WESAD dataset.

Main Components:
- StressModel: Complete end-to-end model
- Encoder: Pyramidal BiLSTM feature extraction
- Decoder: Classification head
- pBLSTM: Pyramidal Bidirectional LSTM layer
- LockedDropout: Temporal dropout for LSTM
- Utilities: Pack, Unpack, PermuteBlock for tensor operations
"""

from .StressModel import StressModel
from .Encoder import Encoder
from .Decoder import Decoder
from .pBLSTM import pBLSTM
from .LockedDropout import LockedDropout
from .Permute import PermuteBlock
from .Pack import Pack
from .Unpack import Unpack

__all__ = [
    'StressModel',
    'Encoder',
    'Decoder',
    'pBLSTM',
    'LockedDropout',
    'PermuteBlock',
    'Pack',
    'Unpack',
]

__version__ = '1.0.0'
__author__ = 'Rishit Saxena'
