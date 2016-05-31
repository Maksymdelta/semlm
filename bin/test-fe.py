import argparse

from semlm.kaldi import read_nbest_file
from semlm.features import UnigramFE

def main():
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    # parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    print(args)
    # colorama.init()
    # Read the n-best lists
    nbests = list(read_nbest_file(args.nbest_file))

    print(len(nbests))
    fe = UnigramFE()
    for nbest in nbests:
        for s in nbest.sentences:
            features = fe.extract(s)
            print(features)


if __name__ == "__main__":
    main()
