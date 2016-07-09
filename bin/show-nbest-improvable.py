#!/usr/bin/env python3

# This could be in another package...

import argparse
import semlm.evaluation_util

from semlm.kaldi import read_nbest_file
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate
from semlm.evaluation_util import REFERENCES
from semlm.nbest_util import evaluate_nbests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()

    nbests = list(read_nbest_file(args.nbest_file))
    refs = read_transcript_table(args.ref_file)
    semlm.evaluation_util.REFERENCES = refs

    overall_eval = evaluate_nbests(nbests)
    for nbest in nbests:
        # print(nbest)
        nbest.print_ref_hyp_best()
        # pass
    print(overall_eval)

if __name__ == "__main__":
    main()
