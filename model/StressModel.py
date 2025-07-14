class StressModel(torch.nn.Module):
  def __init__(self, input_size, embed_size=128, output_size=4):
    super().__init__()

    self.encoder = Encoder(input_size, embed_size)
    self.decoder = Decoder(embed_size, output_size)

  def forward(self, x, x_lens):
    encoder_out, encoder_lens = self.encoder(x, x_lens)

    decoder_out = self.decoder(encoder_out)

    return decoder_out, encoder_lens
