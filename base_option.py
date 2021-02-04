import argparse
import tensorflow as tf


# --- gpu memory growth ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--is_audio", action="store_true")
parser.add_argument("--dataset", default="./AliceCorpus/alice_asr.tfrecord")
parser.add_argument("--epoch", type=int)
parser.add_argument("--batch_sz", type=int)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--tensorboard_dir", type=str)
parser.add_argument("--checkpoint_name", type=str)
parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
parser.add_argument("--lr_schedule", action="store_true")
parser.add_argument("--eval_teacher", action="store_true")
parser.add_argument("--corpus", type=str, choices=["alice", "aishell"], required=True)
parser.add_argument("--arch", type=str, choices=["transformer", "gru"], required=True)
parser.add_argument("--specdim", type=int,required=True)
