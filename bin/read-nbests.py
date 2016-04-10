#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_nbests, read_nbest_files, read_ref
from semlm.evaluation_util import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file")
    parser.add_argument("ac_file")
    parser.add_argument("lm_file")
    parser.add_argument("ref_file")
    args = parser.parse_args()
    nbests = read_nbest_files(args.nbest_file, args.ac_file, args.lm_file)

    # for nbest in nbests:
    #     nbest.print_()

    refs = read_ref(args.ref_file)
    for nbest in nbests:
        id_ = nbest.id_
        ref = refs[id_]
        # for s in nbest.sentences:
        #     evaluate(ref, s)
        evaluate(ref, nbest.sentences[0])


if __name__ == "__main__":
    main()
