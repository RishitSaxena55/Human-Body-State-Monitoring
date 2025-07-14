class pBLSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size=128):
    super(pBLSTM, self).__init__()

    self.blstm1 = nn.LSTM(input_size*2, hidden_size, batch_first=True, bidirectional=True, dropout=0.2)
    self._init_weights()

  def forward(self, x_packed):
    x_unpacked, lens_unpacked = pad_packed_sequence(x_packed, batch_first=True)

    x_reshaped, x_lens_reshaped = self.trunc_reshape(x_unpacked, lens_unpacked)

    x_packed = pack_padded_sequence(x_reshaped, x_lens_reshaped, enforce_sorted=False, batch_first=True)

    out, _ = self.blstm1(x_packed)

    return out

  def trunc_reshape(self, x, x_lens):
    T = x.shape[1]
    if T % 2 != 0:
      x = x[:, :-1, :]
      x_lens = x_lens - 1

    B, T, F = x.shape

    x = torch.reshape(x, (B, T//2, F*2))
    x_lens = torch.clamp(x_lens // 2, min=1)

    return x, x_lens

  def _init_weights(self):
    for name, param in self.blstm1.named_parameters():
      if 'weight_ih' in name:
        nn.init.xavier_uniform_(param.data)
      elif 'weight_hh' in name:
        nn.init.orthogonal_(param.data)
      elif 'bias' in name:
        param.data.fill_(0)
        n = param.size(0)
        param.data[n//4:n//2].fill_(1)
