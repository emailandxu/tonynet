import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-a","--isaudio", action="store_true")
parser.add_argument("-d","--dataset", default="./AliceCorpus/alice_asr.tfrecord")