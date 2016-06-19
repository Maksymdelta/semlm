#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_transcript
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file")
    parser.add_argument("hyp_file")
    args = parser.parse_args()

    print(args.ref_file)
    ref_table = read_transcript_table(args.ref_file)
    hyps = read_transcript(args.hyp_file)

    errs = 0
    evals = []
    for hyp in hyps:
        ref = ref_table[hyp.id_]
        eval_ = evaluate(ref_table, hyp)
        evals.append(eval_)

    print(sum(evals[1:], evals[0]))

if __name__ == "__main__":
    main()
