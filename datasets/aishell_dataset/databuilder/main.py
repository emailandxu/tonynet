import tensorflow as tf
import numpy as np
import os
import pathlib
import tensorflow_io as tfio
from .tfrecord_util import tf_serialize_asr_example, write_tfrecord
import pickle

class AishellDatasetBuilder():

    def __init__(self, subset, trans_mode="text", audio_mode="path", ds_model="raw", aishell_audio_root="/home/tony/D/corpus/data_aishell/wav", aishell_transcript_root="/home/tony/D/corpus/data_aishell/transcript", tfrecord_root="/home/tony/D/corpus/data_aishell/tfrecord/"):
        """ 
        trans_mode: text, tensor 
        audio_mode: path, spec
        ds_model: raw, tfrecord
        """
        
        self.subset = subset
        self.trans_mode = trans_mode
        self.audio_mode = audio_mode
        self.ds_model = ds_model

        self.aishell_audio_root = aishell_audio_root
        self.aishell_transcript_root = aishell_transcript_root
        self.tfrecord_path = os.path.join(tfrecord_root, f"{self.subset}-asr.tfrecod")
        self.trans_tokenizer_path = os.path.join(tfrecord_root, "tokenizer.pkl")


        self.feature_description = {
            'spec_feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'trans_feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }


        if self.ds_model == "raw":
            self._trans_dict, self.trans_tokenizer = AishellDatasetBuilder.build_transcript(self.aishell_transcript_root, self.trans_mode)
            self.asr_ds = AishellDatasetBuilder.build_asr_ds([subset],self._trans_dict, self.aishell_audio_root, self.audio_mode)
        elif self.ds_model == "tfrecord":
            self.asr_ds, self.trans_tokenizer = AishellDatasetBuilder.read_tfrecord(self.trans_tokenizer_path, self.tfrecord_path, self.feature_description)


        self.start_tok_idx = self.trans_tokenizer.word_index['<start>']
        self.end_tok_idx = self.trans_tokenizer.word_index['<end>']
        self.pad_tok_idx = 0


    def __call__(self):
       return self.asr_ds

    def write_tfrecord(self):
        # saving tokenizer
        print("saving tokenizer...")
        with open(self.trans_tokenizer_path, 'wb') as handle:
            pickle.dump(self.trans_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        spec_ds = self.asr_ds.map(lambda spec, trans: spec)
        trans_ds = self.asr_ds.map(lambda spec, trans: trans)

        # 音频频谱图映射合并序列化映射
        serialized_spec_ds = spec_ds.map(tf.io.serialize_tensor)
        # 文本序列化映射
        serialized_trans_ds = trans_ds.map(tf.io.serialize_tensor)
        # 将音频和文本合并到一起, 频谱图在前，文本在后
        serialized_asr_ds = tf.data.Dataset.zip(
            (serialized_spec_ds, serialized_trans_ds))
        # 生成tf.Example的序列化
        serialized_features_ds = serialized_asr_ds.map(tf_serialize_asr_example)

        print("serializing asr dataset...")

        # 持久化到.tfrecord文件中
        write_tfrecord(serialized_features_ds, filepath=self.tfrecord_path)
    
    @staticmethod
    def read_tfrecord(trans_tokenizer_path, tfrecord_path, feature_description):
        print("reading records...")
        tokenizer = None
        # loading
        with open(trans_tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        def _parse_function(example_proto):
            # Parse the input `tf.Example` proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description) 
            spec, trans = example['spec_feature'], example['trans_feature']
            spec, trans = tf.transpose(tf.io.parse_tensor(spec, tf.float32)), tf.io.parse_tensor(trans, tf.int32)
            return spec, trans

        raw_ds = tf.data.TFRecordDataset(tfrecord_path).map(_parse_function)
        spec_ds = raw_ds.map(lambda spec, trans: spec)
        trans_ds = raw_ds.map(lambda spec, trans: trans)
        return tf.data.Dataset.zip((spec_ds, trans_ds)), tokenizer

    @staticmethod
    def build_asr_ds(subsets, trans_dict, aishell_audio_root, audio_mode):
        file_ds = AishellDatasetBuilder._build_audio_file_ds(subsets, aishell_audio_root)
        def filter():
            audio_files = []
            trans = []
            for filepath in file_ds:
                filename = pathlib.Path(filepath.numpy().decode()).stem
                if filename not in trans_dict:
                    continue
                else:
                    audio_files.append(filepath)
                    trans.append(trans_dict[filename])
            return audio_files, trans

        audio_files, trans = filter()
        trans_ds = tf.data.Dataset.from_tensor_slices(trans)
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_files)

        audio_ds_dict = {
            "path": audio_ds,
            "spec": audio_ds.map(AishellDatasetBuilder.tf_to_spectrogram)
        }

        return tf.data.Dataset.zip((audio_ds_dict[audio_mode],trans_ds))
        
    @staticmethod
    def build_transcript(aishell_transcript_root, trans_mode):
        def tokenize(trans):
            trans_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            trans_tokenizer.fit_on_texts(trans)
            trans_tensors = trans_tokenizer.texts_to_sequences(trans)
            trans_tensors = tf.keras.preprocessing.sequence.pad_sequences(trans_tensors, padding='post')
            trans_tensors = tuple(np.array(tensor) for tensor in trans_tensors)
            return trans_tensors, trans_tokenizer
        audiokey_trans = [i.split(" ", 1) for i in open(f"{aishell_transcript_root}/aishell_transcript_v0.8.txt", "r", encoding="utf-8")]
        trans = ["<start> " + t.strip() + " <end>" for _,t in audiokey_trans]
        audiokey = [key for key,_ in audiokey_trans]
        # converted to indexes
        trans_tensors, trans_tokenizer = tokenize(trans)

        trans_dict = {
            "text": dict(zip(audiokey,trans)),
            "tensor": dict(zip(audiokey,trans_tensors))
        }
        return trans_dict[trans_mode], trans_tokenizer

    @staticmethod
    def tf_to_spectrogram(filepath):
        """"/mnt/tonynet/AliceCorpus/train/100001.wav"""
        audio = tfio.audio.AudioIOTensor(filepath, dtype=tf.int16)
        audio_tensor = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
        audio_tensor = tf.squeeze(audio_tensor,axis=[-1])
        spectrogram = tfio.experimental.audio.spectrogram(
            audio_tensor, nfft=79, window=600, stride=460)
        return spectrogram
    

    @staticmethod
    def _build_audio_file_ds(subsets, aishell_audio_root):
        return tf.data.Dataset.list_files(
            # 原版解压
            # [str(self.aishell_audio_root/subset/"*/*.wav") for subset in ("train","dev","test") ],
            # wav文件提升一级目录
            [f"{aishell_audio_root}/{subset}/*.wav" for subset in subsets ], 
            
            shuffle=False) # 默认是乱序，必须显式指定shuffle=False才可以按顺序读取文件名

   

if __name__ == "__main__":
    aishell = AishellDatasetBuilder("train",trans_mode="tensor", audio_mode="spec", ds_model="tfrecord")
    last_idx = 0
    for idx, (trans, spec) in enumerate(aishell()):
        print(trans.shape, spec.shape)
        last_idx = idx
    print(last_idx)
    # aishell.write_tfrecord()


