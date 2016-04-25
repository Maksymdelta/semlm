#!/usr/bin/env python3

import argparse
import semlm.evaluation_util
import matplotlib.pyplot as plt


from semlm.kaldi import read_nbest_file
from semlm.kaldi import read_transcript_table
from semlm.nbest_util import evaluate_nbests
from semlm.nbest_util import evaluate_nbests_oracle
from semlm.nbest_util import evals_by_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    parser.add_argument('--plot', '-p', default=False,  action='store_true')
    
    args = parser.parse_args()

    nbests = list(read_nbest_file(args.nbest_file))
    refs = read_transcript_table(args.ref_file)
    semlm.evaluation_util.REFERENCES = refs
    overall_eval = evaluate_nbests(nbests)
    print('Overall eval:')
    print(overall_eval)
    print('\nOracle eval:')
    print(evaluate_nbests_oracle(nbests))

    evals = evals_by_depth(nbests)
    wers = list(map(lambda x: x.wer(), evals))

    if args.plot:
        plt.plot(wers)
        plt.ylim(ymin=0)
        plt.show()

    
if __name__ == "__main__":
    main()
