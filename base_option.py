import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser = argparse.add_argument("-a","--audio", action="store_true")
parser.add_argument("-d","--dataset", default="~/AliceCorpus/alice_asr.tfrecord")⏎