import tensorflow as tf

class Seq2SeqModel(tf.keras.Model):
    def __init__(self,encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
