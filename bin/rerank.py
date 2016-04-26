#!/usr/bin/env python3

import argparse
import operator
import logging
import termcolor
import colorama
import semlm.evaluation_util

from semlm.kaldi import read_nbest_file
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate
from semlm.nbest_util import evaluate_nbests
from semlm.sentence_util import print_sentence_scores
from semlm.sentence import Sentence
from semlm.scores import monotone
from semlm.reranking import rerank_nbests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    colorama.init()
    nbests = list(read_nbest_file(args.nbest_file))
    refs = read_transcript_table(args.ref_file)
    semlm.evaluation_util.REFERENCES = refs
    overall_eval = evaluate_nbests(nbests)
    print('BEFORE:')
    print(evaluate_nbests(nbests))
    rerank_nbests(nbests, Sentence.score)
    print('AFTER:')
    print(evaluate_nbests(nbests))

if __name__ == "__main__":
    main()
