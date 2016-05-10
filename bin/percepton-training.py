#!/usr/bin/env python3


# Methods
# decision_function(X)Predict confidence scores for samples.
# densify()Convert coefficient matrix to dense array format.
# fit(X, y[, coef_init, intercept_init, ...])Fit linear model with Stochastic Gradient Descent.
# fit_transform(X[, y])Fit to data, then transform it.
# get_params([deep])Get parameters for this estimator.
# partial_fit(X, y[, classes, sample_weight])Fit linear model with Stochastic Gradient Descent.
# predict(X)Predict class labels for samples in X.
# score(X, y[, sample_weight])Returns the mean accuracy on the given test data and labels.
# set_params(*args, **kwargs)
# sparsify()Convert coefficient matrix to sparse format.
# transform(*args, **kwargs)DEPRECATED: Support to use estimators as feature selectors will be removed in version 0.19.


import argparse
import operator
import logging
import termcolor
import colorama
import semlm.evaluation_util

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron

from semlm.kaldi import read_nbest_file
from semlm.kaldi import read_transcript_table
from semlm.evaluation_util import evaluate
from semlm.nbest_util import evaluate_nbests
from semlm.sentence_util import print_sentence_scores
from semlm.sentence import Sentence
from semlm.scores import monotone

from semlm.perceptron_training import generate_training_pairs
from semlm.perceptron_training import pair_to_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    colorama.init()
    nbests = list(read_nbest_file(args.nbest_file))
    refs = read_transcript_table(args.ref_file)
    semlm.evaluation_util.REFERENCES = refs
    print(len(nbests))
    overall_eval = evaluate_nbests(nbests)
    pairs, classifications = generate_training_pairs(nbests)
    feature_dicts = list(map(pair_to_dict, pairs))
    # print(len(pairs))
    # print(len(feature_dicts))
    vec = DictVectorizer()
    feature_array = vec.fit_transform(feature_dicts).toarray()
    # print(len(feature_array))
    print(vec.get_feature_names())
    print(len(feature_array))
    print(len(feature_array[0]))
    p = Perceptron(verbose=5, n_iter=10)
    p.fit(feature_array, classifications)
    print(p.coef_)
    print(len(p.coef_[0]))

    # print(p.get_params())

    for name, val in zip(vec.get_feature_names(), p.coef_[0]):
        print ('{:20} {}'.format(name, val))

    
    
if __name__ == "__main__":
    main()
