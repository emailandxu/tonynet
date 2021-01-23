import tensorflow as tf
from .pre_encoder import PreEncoder
class Encoder(tf.keras.Model):
  def __init__(self,pre_encoder_input_dim, pre_encoder_output_dim, enc_units, is_audio):
    """如果是音频，前编码是convs，如果是文本，前编码是embedding
    """
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.is_audio = is_audio
    if self.is_audio:
      self.pre_encoder = PreEncoder(pre_encoder_input_dim,pre_encoder_output_dim)
    else:
      self.pre_encoder = tf.keras.layers.Embedding(pre_encoder_input_dim, pre_encoder_output_dim)


    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.pre_encoder(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self, batch_sz):
    return tf.zeros((batch_sz, self.enc_units))

