#!/usr/bin/env python3

# Linear models:
# http://scikit-learn.org/stable/modules/linear_model.html#linear-model
# See this too:
# http://scikit-learn.org/stable/modules/classes.html

import argparse
import operator
import logging
import termcolor
import colorama
import semlm.evaluation_util

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from semlm.kaldi import read_nbest_file
from semlm.evaluation_util import evaluate
from semlm.kaldi import read_transcript_table
from semlm.nbest_util import evaluate_nbests
from semlm.nbest_util import print_nbest
from semlm.nbest_util import evaluate_nbests_oracle
from semlm.sentence import Sentence
from semlm.scores import monotone

from semlm.pairwise_ranking import generate_training_pairs
from semlm.features import pair_to_dict

from semlm.sklearn import print_feature_weights
from semlm.sklearn import evaluate_model

def load_references(f, evaluate=False):
    refs = read_transcript_table(f)
    semlm.evaluation_util.REFERENCES = refs

def print_eval(nbests):
    eval = evaluate_nbests(nbests)
    print('Eval:')
    print(eval)
    print('Oracle eval:')
    print(evaluate_nbests_oracle(nbests))

def main():
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    colorama.init()
    # Read the n-best lists
    nbests = list(read_nbest_file(args.nbest_file))

    # Read in the references
    load_references(args.ref_file)
    print('# of n-bests: {}'.format(len(nbests)))

    # Print evaluation
    print_eval(nbests)

    # Print the n-best lists
    # for nbest in nbests:
    #     print('NBEST:')
    #     print_nbest(nbest, acscore=True, lmscore=True, tscore=True, maxwords=10)

    # Convert the n-best lists to training examples.
    pairs, classifications = generate_training_pairs(nbests)
    feature_dicts = list(map(pair_to_dict, pairs))
    print('# of training pairs generated: {}'.format(len(pairs)))
    assert len(feature_dicts) == len(pairs)

    # Do feature extraction
    vec = DictVectorizer()

    # This is building a "csr_matrix" object
    vec.fit(feature_dicts)
    print(type(vec).__name__)
    print(vec)
    print(vec.vocabulary_)
    print(vec.get_params())
    exit

    features = vec.fit_transform(feature_dicts)
    print(type(features).__name__)
    print(features)
    exit()

    # Train a perceptron or other model.
    # Perceptron, SGDClassifier, LinearRegression
    model = LogisticRegression()
    model.fit(feature_array, classifications)

    # Print feature weights and do a pairwise evaluation of the model
    print_feature_weights(model, vec)
    evaluate_model(model, (feature_array, classifications))


if __name__ == "__main__":
    main()
