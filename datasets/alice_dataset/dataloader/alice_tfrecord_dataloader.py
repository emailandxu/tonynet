import tensorflow as tf
from .tokenizer import tokenize_dataset

class Dataloader:
  def __init__(self, filename):
    self.raw_dataset = tf.data.TFRecordDataset(filename)
    self.feature_description = {
        'spec_feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'trans_feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

  def _select_parse_function(self,select="all"):
    def _parse_function(example_proto):
      # Parse the input `tf.Example` proto using the dictionary above.
      example = tf.io.parse_single_example(example_proto, self.feature_description) 
      spec, trans = example['spec_feature'], example['trans_feature']
      # return spec,trans
      spec, trans = tf.transpose(tf.io.parse_tensor(spec, tf.float32)), tf.io.parse_tensor(trans, tf.string)
      if select == "spec":
        return spec
      elif select == "trans":
        return trans
      elif select == "all":
        return spec, trans
      else:
        return spec,trans
    return _parse_function

  def get_alice_asr_dataset(self):
    trans_tensor_dataset, trans_tokenizer = self.get_alice_text_dataset()
    spec_dateset = self.get_alice_spec_dataset()
    alice_asr_ds = tf.data.Dataset.zip((spec_dateset,trans_tensor_dataset))
    return alice_asr_ds, trans_tokenizer

  def get_alice_text_dataset(self):
    raw_text_ds = self.raw_dataset.map(self._select_parse_function(select="trans"))
    trans_tensor_dataset,trans_tokenizer = tokenize_dataset(raw_text_ds)
    return trans_tensor_dataset, trans_tokenizer

  def get_alice_spec_dataset(self):
    return self.raw_dataset.map(self._select_parse_function(select="spec"))