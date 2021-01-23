import tensorflow as tf
import tensorflow_io as tfio

def tf_to_spectrogram(filepath):
    """"/mnt/tonynet/AliceCorpus/train/100001.wav"""
    audio = tfio.audio.AudioIOTensor(filepath, dtype=tf.int16)
    audio_tensor = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
    audio_tensor = tf.squeeze(audio_tensor,axis=[-1])
    spectrogram = tfio.experimental.audio.spectrogram(
        audio_tensor, nfft=512, window=512, stride=256)
    return spectrogram