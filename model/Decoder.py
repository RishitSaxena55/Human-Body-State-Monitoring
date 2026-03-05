import torch
import torch.nn as nn


class Decoder(torch.nn.Module):
    """Decoder/Classifier module that maps encoder embeddings to class predictions.
    
    Takes the fixed-size embeddings from the encoder and uses a feed-forward
    neural network to classify the stress state into one of 4 categories:
    - Baseline
    - Stress
    - Amusement
    - Meditation
    """
    
    def __init__(self, embed_size, output_size=4):
        """Initialize the Decoder.
        
        Args:
            embed_size (int): Size of input embeddings from encoder
            output_size (int): Number of output classes. Default: 4
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )
        self._init_weights()

    def forward(self, encoder_out):
        """Forward pass through the decoder.
        
        Args:
            encoder_out: Encoder embeddings of shape (batch, 2*embed_size)
            
        Returns:
            Logits tensor of shape (batch, output_size) for classification
        """
        out = self.mlp(encoder_out)
        return out

    def _init_weights(self):
        """Initialize linear layer weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
