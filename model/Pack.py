class Pack(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, x_lens):
    x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False, batch_first=True)

    return x_packed