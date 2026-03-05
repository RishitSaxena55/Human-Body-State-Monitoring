import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class Unpack(torch.nn.Module):
    """Module to unpack packed sequences back to padded format.
    
    This module wraps PyTorch's pad_packed_sequence functionality.
    """
    
    def __init__(self):
        """Initialize the Unpack module."""
        super().__init__()

    def forward(self, x_packed):
        """Unpack padded sequences from PackedSequence object.
        
        Args:
            x_packed: PackedSequence object
            
        Returns:
            Tuple of (unpacked_tensor, lengths) where unpacked_tensor is of shape
            (batch, max_len, features) and lengths is a tensor of actual sequence lengths
        """
        x_unpacked, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        return x_unpacked, x_lens
