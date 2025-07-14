class PermuteBlock(torch.nn.Module):
  def forward(self, x):
    return x.transpose(1, 2)