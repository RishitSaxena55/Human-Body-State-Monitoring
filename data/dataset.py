class WESAD(Dataset):
  def __init__(self, df, subjects_to_include=None, feature_cols=None):

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
    X = self.X_seq[ind]
    y = torch.tensor(self.y_seq[ind], dtype=torch.int64)

    return X, y

  def __len__(self):
    return self.length

  def collate_fn(self, batch):
    batch_X, batch_y = zip(*batch)

    X_lens = [X.shape[0] for X in batch_X]

    batch_X = pad_sequence(batch_X, batch_first=True)

    return batch_X, torch.tensor(batch_y, dtype=torch.int64), torch.tensor(X_lens)