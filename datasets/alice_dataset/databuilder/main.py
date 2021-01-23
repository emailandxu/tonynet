import tensorflow as tf
import numpy as np
import os
import pathlib
from util import *


alice_audio_root = pathlib.Path("./AliceCorpus/")
alice_transcript_root = "./AliceCorpus/"

alice_transcript_ds = tf.data.TextLineDataset([os.path.join(
    alice_transcript_root, filename) for filename in ("eval.txt","train.txt")]) # 因为eval首字母排序在train前面，否则无法和文件对齐
alice_audio_file_ds = tf.data.Dataset.list_files(
    [str(alice_audio_root/subset/"*.wav") for subset in ("train", "eval")], shuffle=False) # 默认是乱序，必须显式指定shuffle=False才可以按顺序读取文件名


def _show_alice_audio_filenames(take=5):
    for item in alice_audio_file_ds.take(take):
        print(item.numpy())


def _show_alice_audio_ds(take=5):
    for item in alice_audio_file_ds.take(take):
        print(item.numpy())

def _show_asr_ds(take=5):
    for filename, transcript in tf.data.Dataset.zip((alice_audio_file_ds,alice_transcript_ds)).take(take):
        print(filename.numpy(), transcript.numpy())

def main():
    # 音频频谱图映射合并序列化映射
    serialized_spec_ds = alice_audio_file_ds.map(
        tf_to_spectrogram).map(tf.io.serialize_tensor)
    # 文本序列化映射
    serialized_transcript_ds = alice_transcript_ds.map(tf.io.serialize_tensor)
    # 将音频和文本合并到一起, 频谱图在前，文本在后
    serialized_asr_ds = tf.data.Dataset.zip(
        (serialized_spec_ds, serialized_transcript_ds))
    # 生成tf.Example的序列化
    serialized_features_ds = serialized_asr_ds.map(tf_serialize_asr_example)
    # 持久化到.tfrecord文件中
    write_tfrecord(serialized_features_ds, filename="alice_asr.tfrecord")


if __name__ == "__main__":
    _show_asr_ds(-1)
