import torch
import torch.nn as nn

from .Encoder import Encoder
from .Decoder import Decoder


class StressModel(torch.nn.Module):
    """Complete Stress Detection Model combining Encoder and Decoder.
    
    This model processes variable-length biometric signal sequences and
    classifies them into one of four stress states (Baseline, Stress,
    Amusement, or Meditation).
    
    Architecture:
    - Input: Variable-length biometric signals (~62 features)
    - Encoder: Pyramidal BiLSTM-based feature extraction
    - Decoder: Feed-forward classifier
    - Output: 4-class predictions
    """
    
    def __init__(self, input_size, embed_size=128, output_size=4):
        """Initialize the StressModel.
        
        Args:
            input_size (int): Number of input biometric features
            embed_size (int): Embedding/hidden size for encoder. Default: 128
            output_size (int): Number of output classes. Default: 4
        """
        super().__init__()
        self.encoder = Encoder(input_size, embed_size)
        self.decoder = Decoder(embed_size, output_size)

    def forward(self, x, x_lens):
        """Forward pass through the complete model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            x_lens: Tensor of actual sequence lengths for each sample
            
        Returns:
            Tuple of (predictions, encoder_lens) where:
            - predictions: Logits tensor of shape (batch, output_size)
            - encoder_lens: Sequence lengths (for info purposes)
        """
        encoder_out, encoder_lens = self.encoder(x, x_lens)
        decoder_out = self.decoder(encoder_out)
        return decoder_out, encoder_lens
