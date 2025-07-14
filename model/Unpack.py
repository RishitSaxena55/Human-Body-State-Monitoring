class Unpack(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x_packed):
    x_unpacked, x_lens = pad_packed_sequence(x_packed, batch_first=True)

    return x_unpacked, x_lens
