from models import Transformer
class ModelFactory():
    def __init__(self):
        pass

    def __call__(self, specdim, vocab_tar_size, args):
        if args.arch == "transformer":
            dmodel = 512
            return Transformer(4, dmodel, dmodel, 4, 512, specdim, vocab_tar_size)
        elif args.arch == "gru":
            pass
