#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_nbest_file
# from semlm.kaldi_better import read_nbest_file
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate
from semlm.evaluation_util import REFERENCES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()

    nbests = read_nbest_file(args.nbest_file)
    refs = read_transcript_table(args.ref_file)
    REFERENCES = refs
    evals = []
    for nbest in nbests:
        id_ = nbest.id_
        ref = refs[id_]
        for s in nbest.sentences:
            e = evaluate(refs, s)
            s.eval_ = e
            evals.append(e)
            # print(s)
        # print(nbest)
        nbest.print_ref_hyp_best()
    overall_eval = sum(evals[1:], evals[0])
    print(overall_eval)

if __name__ == "__main__":
    main()
