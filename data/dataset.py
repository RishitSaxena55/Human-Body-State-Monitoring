import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class WESAD(Dataset):
    """WESAD (Wearable Stress and Affect Detection) Dataset wrapper.
    
    This dataset class handles the WESAD time-series data and prepares it for
    training using Leave-One-Subject-Out (LOSO) cross-validation.
    
    Attributes:
        X_seq (list): List of input sequences as tensors
        y_seq (list): List of labels for each sequence
        lens (list): List of sequence lengths
        length (int): Total number of samples in the dataset
    """
    
    def __init__(self, df, subjects_to_include=None, feature_cols=None):
        """Initialize the WESAD dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe containing WESAD data with columns:
                - 'subject id': Subject identifier
                - 'Time': Timestamp or time index
                - 'label': Class label (0-3 for baseline, stress, amusement, meditation)
                - Features: Biometric signal features
            subjects_to_include (list, optional): List of subject IDs to include.
                If None, all subjects are included.
            feature_cols (list): List of feature column names to use as input
        """
        self.X_seq, self.y_seq, self.lens = [], [], []
        grouped = df.groupby('subject id')

        for subject_id, group in grouped:
            if subjects_to_include is not None and subject_id not in subjects_to_include:
                continue

            group = group.sort_values('Time')
            X = group[feature_cols].values
            y = group['label'].values[0]

            self.X_seq.append(torch.tensor(X, dtype=torch.float32))
            self.y_seq.append(y)
            self.lens.append(len(X))

        self.length = len(self.y_seq)

    def __getitem__(self, ind):
        """Get a single sample from the dataset.
        
        Args:
            ind (int): Index of the sample to retrieve
            
        Returns:
            tuple: (X, y) where X is the feature sequence and y is the label
        """
        X = self.X_seq[ind]
        y = torch.tensor(self.y_seq[ind], dtype=torch.int64)
        return X, y

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length

    def collate_fn(self, batch):
        """Collate function for DataLoader to handle variable-length sequences.
        
        Pads sequences to the same length within a batch and returns sequence lengths
        for proper handling of variable-length sequences in model.
        
        Args:
            batch (list): List of (X, y) tuples
            
        Returns:
            tuple: (padded_X, y_labels, sequence_lengths)
                - padded_X: Tensor of shape (batch_size, max_len, num_features)
                - y_labels: Tensor of shape (batch_size,)
                - sequence_lengths: Tensor of original sequence lengths for each sample
        """
        batch_X, batch_y = zip(*batch)
        X_lens = [X.shape[0] for X in batch_X]
        batch_X = pad_sequence(batch_X, batch_first=True)

        return batch_X, torch.tensor(batch_y, dtype=torch.int64), torch.tensor(X_lens)