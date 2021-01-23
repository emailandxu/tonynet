import re
import numpy as np
import tensorflow as tf
import unicodedata
# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
    return w

def tokenize(trans):
  trans_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  trans_tokenizer.fit_on_texts(trans)
  tensor = trans_tokenizer.texts_to_sequences(trans)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  return tensor, trans_tokenizer

def tokenize_dataset(raw_trans_dataset):
    trans = [preprocess_sentence(trans.numpy().decode()) for trans in raw_trans_dataset]
    print("\n".join(trans[:2]))
    trans_tensor, trans_tokenizer = tokenize(trans)
    trans_tensor_dataset = tf.data.Dataset.from_tensor_slices(trans_tensor)
    return trans_tensor_dataset, trans_tokenizer