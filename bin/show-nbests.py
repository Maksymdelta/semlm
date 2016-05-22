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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", nargs='?', type=argparse.FileType('r')) # optional
    args = parser.parse_args()
    colorama.init()
    nbests = list(read_nbest_file(args.nbest_file))
    if args.ref_file:
        refs = read_transcript_table(args.ref_file)
        semlm.evaluation_util.REFERENCES = refs
        overall_eval = evaluate_nbests(nbests)
    # print(nbests)
    for nbest in nbests:
        print('NBEST:')
        print(nbest)
        for s in nbest.sentences:
            print_sentence_scores(s)
        if not monotone(nbest.sentences, comparison=operator.lt, key=Sentence.score):
            print(termcolor.colored('WARNING: Non-montonic scores', 'red', attrs=['bold']))
        print('\n\n')
    print(overall_eval)


if __name__ == "__main__":
    main()
