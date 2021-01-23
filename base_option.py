import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--is_audio", action="store_true")
parser.add_argument("--dataset", default="./AliceCorpus/alice_asr.tfrecord")
parser.add_argument("--epoch", type=int)
parser.add_argument("--batch_sz", type=int)
parser.add_argument("--checkpoint_dir", type=str)