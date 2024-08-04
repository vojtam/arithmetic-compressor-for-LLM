import copy
import arithmetic_compressor.arithmetic_coding as AE
from tqdm.notebook import tqdm 

# Compress using arithmetic encoding


class AECompressor:
  def __init__(self, model, adapt=True) -> None:
    self.adapt = adapt
    # clone model, so we dont mutate original
    self.model = copy.deepcopy(model)
    self.__model = copy.deepcopy(model)  # for decoding

  def compress(self, data):
    encoder = AE.Encoder()

    for symbol in tqdm(data, desc = "Compression"):
      # Use the model to predict the probability of the next symbol
      cdf = self.model.cdf()

      # encode the symbol
      encoder.encode_symbol(cdf, symbol)

      if self.adapt:
        # update the model with the new symbol
        self.model.update(symbol)
    encoder.finish()
    return encoder.get_encoded()

  def decompress(self, encoded, length_encoded):
    decoded = []
    model = self.__model
    decoder = AE.Decoder(encoded)
    for _ in tqdm(range(length_encoded), desc = "Decompression"):
      # probability of the next symbol
      probability = model.cdf()

      # decode symbol
      symbol = decoder.decode_symbol(probability)

      if self.adapt:
        # update model
        model.update(symbol)

      decoded += [symbol]
    return decoded
