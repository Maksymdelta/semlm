import semlm.nbest
import semlm.sentence


def read_nbest(filename):
    "Read a Kaldi n-best file."
    with open(filename) as f:
        for line in f:
            print(f)

