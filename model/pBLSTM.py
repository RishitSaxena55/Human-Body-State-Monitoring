import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class pBLSTM(torch.nn.Module):
    """Pyramidal Bidirectional LSTM (pBLSTM) layer.
    
    This layer reduces the temporal dimension by a factor of 2 at each step
    while doubling the feature dimension by concatenating consecutive frames.
    This hierarchical structure helps capture multi-resolution context.
    
    The pyramidal structure works by:
    1. Unpacking the input sequences
    2. Concatenating every two consecutive frames (increasing feature dimension)
    3. Reducing sequence length by half
    4. Passing through bidirectional LSTM
    """
    
    def __init__(self, input_size, hidden_size=128):
        """Initialize the pBLSTM layer.
        
        Args:
            input_size (int): Size of input features (will be doubled internally)
            hidden_size (int): Hidden size of the LSTM layer. Default: 128
        """
        super(pBLSTM, self).__init__()
        self.blstm1 = nn.LSTM(
            input_size * 2, hidden_size, batch_first=True, 
            bidirectional=True, dropout=0.2
        )
        self._init_weights()

    def forward(self, x_packed):
        """Forward pass through pBLSTM.
        
        Args:
            x_packed: PackedSequence object from previous layer
            
        Returns:
            out: PackedSequence object containing LSTM outputs
        """
        x_unpacked, lens_unpacked = pad_packed_sequence(x_packed, batch_first=True)
        x_reshaped, x_lens_reshaped = self.trunc_reshape(x_unpacked, lens_unpacked)
        x_packed = pack_padded_sequence(x_reshaped, x_lens_reshaped, enforce_sorted=False, batch_first=True)
        out, _ = self.blstm1(x_packed)

        return out

    def trunc_reshape(self, x, x_lens):
        """Reshape input by concatenating consecutive frames.
        
        Handles odd-length sequences by truncating the last frame if necessary.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            x_lens: Tensor of sequence lengths
            
        Returns:
            Tuple of (reshaped_x, new_lens) where reshaped_x has shape
            (batch, time//2, features*2) and new_lens is updated accordingly
        """
        T = x.shape[1]
        if T % 2 != 0:
            x = x[:, :-1, :]
            x_lens = x_lens - 1

        B, T, F = x.shape
        x = torch.reshape(x, (B, T // 2, F * 2))
        x_lens = torch.clamp(x_lens // 2, min=1)

        return x, x_lens

    def _init_weights(self):
        """Initialize LSTM weights using standard initialization schemes."""
        for name, param in self.blstm1.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
