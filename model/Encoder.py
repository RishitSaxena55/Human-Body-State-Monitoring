import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .Permute import PermuteBlock
from .pBLSTM import pBLSTM
from .LockedDropout import LockedDropout
from .Pack import Pack
from .Unpack import Unpack


class Encoder(torch.nn.Module):
    """Encoder module using Pyramidal BiLSTM architecture.
    
    The encoder transforms variable-length biometric signal sequences into
    fixed-size embeddings. It uses:
    - Conv1d for initial feature embedding
    - Pyramidal BiLSTM layers for hierarchical sequence encoding
    - Adaptive average pooling to create fixed-size output
    - Locked dropout for regularization during training
    """
    
    def __init__(self, input_size, encoder_hidden_size=128):
        """Initialize the Encoder.
        
        Args:
            input_size (int): Number of input features from biometric signals
            encoder_hidden_size (int): Hidden size for pBLSTM layers. Default: 128
        """
        super(Encoder, self).__init__()
        self.permute = PermuteBlock()
        self.embedding = nn.Conv1d(input_size, 128, kernel_size=3, padding=1, stride=1)

        self.pBLSTMs = torch.nn.Sequential(
            pBLSTM(128, encoder_hidden_size),
            pBLSTM(2 * encoder_hidden_size, encoder_hidden_size),
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.locked_dropout = LockedDropout()
        self.pack = Pack()
        self.unpack = Unpack()

        self._init_weights()

    def forward(self, x, x_lens):
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            x_lens: Tensor of actual sequence lengths for each sample
            
        Returns:
            Tuple of (encoder_outputs, encoder_lens) where:
            - encoder_outputs: Fixed-size embeddings of shape (batch, 2*hidden_size)
            - encoder_lens: Updated sequence lengths after pooling
        """
        # Embedding phase: convert to (batch, 128, seq_len)
        x = self.permute(x)
        x = self.embedding(x)
        x = self.permute(x)  # Back to (batch, seq_len, 128)

        # Pyramidal BiLSTM layers with packing/unpacking for efficiency
        for layer in self.pBLSTMs:
            x = self.pack(x, x_lens)
            x = layer(x)
            x, x_lens = self.unpack(x)
            x = self.permute(x)
            x = self.locked_dropout(x)
            x = self.permute(x)

        encoder_outputs, encoder_lens = (x, x_lens)

        # Adaptive pooling to get fixed-size representation
        encoder_outputs = self.permute(encoder_outputs)  # (batch, features, seq_len)
        encoder_outputs = self.pooling(encoder_outputs)  # (batch, features, 1)
        encoder_outputs = encoder_outputs.squeeze(-1)  # (batch, features)

        return encoder_outputs, encoder_lens

    def _init_weights(self):
        """Initialize Conv1d weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
