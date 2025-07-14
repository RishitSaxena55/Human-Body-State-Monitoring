class Decoder(torch.nn.Module):
  def __init__(self, embed_size, output_size=4):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(2*embed_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, output_size)
      )



    self._init_weights()

  def forward(self, encoder_out):
    out = self.mlp(encoder_out)

    return out

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
