class Encoder(torch.nn.Module):
  def __init__(self, input_size, encoder_hidden_size=128):
    super(Encoder, self).__init__()

    self.permute = PermuteBlock()
    self.embedding = nn.Conv1d(input_size, 128, kernel_size=3, padding=1, stride=1)

    self.pBLSTMs = torch.nn.Sequential(
        pBLSTM(128, encoder_hidden_size),
        pBLSTM(2*encoder_hidden_size, encoder_hidden_size),
    )

    self.pooling = nn.AdaptiveAvgPool1d(1)
    self.locked_dropout = LockedDropout()

    self.pack = Pack()
    self.unpack = Unpack()

    self._init_weights()

  def forward(self, x, x_lens):
    x = self.permute(x)
    x = self.embedding(x)
    x = self.permute(x)

    for layer in self.pBLSTMs:
      x = self.pack(x, x_lens)
      x = layer(x)
      x, x_lens = self.unpack(x)
      x = self.permute(x)
      x = self.locked_dropout(x)
      x = self.permute(x)

    encoder_outputs, encoder_lens = (x, x_lens)

    encoder_outputs = self.permute(encoder_outputs)
    encoder_outputs = self.pooling(encoder_outputs)
    encoder_outputs = encoder_outputs.squeeze(-1)

    return encoder_outputs, encoder_lens

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
