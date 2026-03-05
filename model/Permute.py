import torch
import torch.nn as nn


class PermuteBlock(torch.nn.Module):
    """Permute layer that transposes dimensions 1 and 2 of input tensor.
    
    Used to convert between (batch, time, features) and (batch, features, time)
    formats for compatibility with different layer types (e.g., Conv1d vs LSTM).
    """
    
    def forward(self, x):
        """Transpose dimensions 1 and 2 of input tensor.
        
        Args:
            x: Input tensor of shape (batch, dim1, dim2)
            
        Returns:
            Transposed tensor of shape (batch, dim2, dim1)
        """
        return x.transpose(1, 2)