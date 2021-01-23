import tensorflow as tf
import numpy as np
import os
import pathlib
import tensorflow_io as tfio



class AishellDatasetBuilder():

    def __init__(self, subset, trans_mode="text", audio_mode="path", aishell_audio_root="/home/tony/D/corpus/data_aishell/wav", aishell_transcript_root="/home/tony/D/corpus/data_aishell/transcript"):
        """ 
        trans_mode: text, tensor 
        audio_mode: path, spec
        """

        self.subset = subset
        self.trans_mode = trans_mode
        self.audio_mode = audio_mode

        self.aishell_audio_root = aishell_audio_root
        self.aishell_transcript_root = aishell_transcript_root

        self.trans_dict, self.trans_tokenizer = AishellDatasetBuilder.build_transcript(self.aishell_transcript_root, self.trans_mode)
        self.asr_ds = AishellDatasetBuilder.build_asr_ds([subset],self.trans_dict, self.aishell_audio_root, self.audio_mode)


        self.start_tok_idx = self.trans_tokenizer.word_index['<start>']
        self.end_tok_idx = self.trans_tokenizer.word_index['<end>']
        self.pad_tok_idx = 0

    def __call__(self):
       return self.asr_ds

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

        return tf.data.Dataset.zip((trans_ds, audio_ds_dict[audio_mode]))
        

if __name__ == "__main__":
    test_asr_zipped_ds = AishellDatasetBuilder("test",trans_mode="tensor", audio_mode="spec")
    for trans, spec in test_asr_zipped_ds().take(2):
        print(trans.numpy())
        print(spec.numpy())
