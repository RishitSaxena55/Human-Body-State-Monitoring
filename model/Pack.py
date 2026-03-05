import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Pack(torch.nn.Module):
    """Module to pack padded sequences for efficient LSTM processing.
    
    This module wraps PyTorch's pack_padded_sequence functionality.
    """
    
    def __init__(self):
        """Initialize the Pack module."""
        super().__init__()

    def forward(self, x, x_lens):
        """Pack padded sequences.
        
        Args:
            x: Padded input tensor of shape (batch, max_len, features)
            x_lens: Tensor of actual sequence lengths for each sample
            
        Returns:
            PackedSequence object for efficient processing
        """
        x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False, batch_first=True)
        return x_packed