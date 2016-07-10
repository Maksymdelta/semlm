#!/usr/bin/env python3

# This is the main logic I'm working on now.

import argparse
import operator
import logging
import termcolor
import colorama
import itertools
import asr_tools.evaluation_util

from sklearn.linear_model import Perceptron, SGDClassifier, LinearRegression, LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from asr_tools.evaluation_util import evaluate
from semlm.features import generate_training_pairs, pair_to_dict, features_to_dict
from semlm.feature_extractor import UnigramFE, ProFE
from asr_tools.kaldi import read_nbest_file, read_transcript_table
from asr_tools.nbest_util import evaluate_nbests, print_nbest, evaluate_nbests_oracle
from asr_tools.sentence import Sentence
from semlm.sklearn import print_feature_weights, evaluate_model, examples_to_matrix
from asr_tools.scores import monotone
from semlm.util import load_references, print_eval, print_train_test_eval, print_nbests, extract_dict_examples
from semlm.example import Example

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    return parser.parse_args()


def main():
    args = parse_args()

    colorama.init()
    # Read the n-best lists and references
    nbests = list(read_nbest_file(args.nbest_file))    
    load_references(args.ref_file)
    evaluate_nbests(nbests) 
    # Figure out all the features

    fe = UnigramFE()
    pro_fe = ProFE()    

    # Create train/test examples...
    examples = []
    for nbest in nbests:
        for s1, s2 in itertools.combinations(nbest.sentences, 2):
            features = pro_fe.extract(s1, s2, fe)
            class_ = 1 if s1.wer() < s2.wer() else -1
            example = Example(class_, features)
            examples.append(example)
            # print(example)

    # Converts the Example objects to sk-learn objects (matrices)
    print('# of examples: {}'.format(len(examples)))
    vec, data = examples_to_matrix(examples)

    # Do a little printing:
    print('DATA:')
    print(data)
    print('Inverse transformed data (first 5):')
    print(vec.inverse_transform(data)[:5])
    

if __name__ == "__main__":
    main()
