import tensorflow as tf
import numpy as np

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_serialize_asr_example(serialized_spec, serialized_trans):
    def serialize_asr_example(serialized_spec, serialized_trans):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        feature = { 
            "spec_feature" : bytes_feature(serialized_spec),
            "trans_feature" : bytes_feature(serialized_trans)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    tf_string = tf.py_function(
        serialize_asr_example,
        (serialized_spec, serialized_trans),    # pass these args to the above function.
        tf.string     # the return type is `tf.string`.
    )
    return tf.reshape(tf_string, ()) # The result is a scalar

def write_tfrecord(serialized_features_ds, filepath='alice_asr.tfrecord'):
  writer = tf.data.experimental.TFRecordWriter(filepath)
  writer.write(serialized_features_ds)