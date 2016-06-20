#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_transcript
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate
from semlm.evaluation_util import evaluate_hyps


def arg_parser():
    desc="""Compute the WER between two 'transcript'-like files.
    The first token of each line should be the ID.  The order of the
    lines doesn't matter."""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    parser.add_argument("hyp_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    ref_table = read_transcript_table(args.ref_file)
    hyps = read_transcript(args.hyp_file)
    print(evaluate_hyps(hyps, ref_table))

if __name__ == "__main__":
    main()