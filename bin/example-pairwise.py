#!/usr/bin/env python3

import argparse
import operator
import logging
import termcolor
import colorama
import semlm.evaluation_util

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier, LinearRegression, LogisticRegression
from semlm.evaluation_util import evaluate
from semlm.features import generate_training_pairs, pair_to_dict, features_to_dict
from semlm.feature_extractor import UnigramFE
from semlm.kaldi import read_nbest_file, read_transcript_table
from semlm.nbest_util import evaluate_nbests, print_nbest, evaluate_nbests_oracle
from semlm.sentence import Sentence
from semlm.sklearn import print_feature_weights, evaluate_model
from semlm.scores import monotone
from semlm.util import load_references, print_eval, print_train_test_eval, print_nbests, extract_dict_examples

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
    train_nbests = nbests[:len(nbests) // 2]
    test_nbests = nbests[len(nbests) // 2:]

    print('Training/test/total nbests: {}/{}/{}'.format(len(train_nbests),
                                                        len(test_nbests),
                                                        len(nbests)))
    load_references(args.ref_file)
    
    # Print evaluation
    print_train_test_eval(train_nbests, test_nbests)
    # print_nbests(nbests)

    # Figure out all the features
    vec = DictVectorizer()
    train_dict, train_class = extract_dict_examples(train_nbests, vec)
    test_dict, test_class = extract_dict_examples(test_nbests, vec)
    # This is building a "csr_matrix" object--compressed sparse row
    vec.fit(train_dict + test_dict)
    print(vec)
    train_feat = vec.transform(train_dict)
    test_feat = vec.transform(test_dict)

    print('Vocab sample:            {}'.format(vec.feature_names_[:10]))
    print('Params object:           {}'.format(vec.get_params()))
    print('Feature representation:  {}'.format(type(train_feat).__name__))
    print('Feature representation:  {}'.format(type(test_feat).__name__))
    print('Train feature array dim: {dim[0]} x {dim[1]}'.format(dim=train_feat.shape))
    print('Test feature array dim:  {dim[0]} x {dim[1]}'.format(dim=test_feat.shape))

    # Train a perceptron or other model. e.g. Perceptron, SGDClassifier, LinearRegression
    print('Training model:')
    # model = LogisticRegression(verbose=10) # penalty='l2')
    model = Perceptron() # penalty='l2')
    model.fit(train_feat, train_class)

    # Print feature weights and do a pairwise evaluation of the model on training data.
    # print_feature_weights(model, vec)
    print('Eval on train data:')
    evaluate_model(model, (train_feat, train_class))
    print('Eval on test data:')
    evaluate_model(model, (test_feat, test_class))
    

if __name__ == "__main__":
    main()
