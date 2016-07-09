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
    load_references(args.ref_file)
    evaluate_nbests(nbests) 
    # Figure out all the features
    vec = DictVectorizer()
    feature_dict, classes = extract_dict_examples(nbests, vec)
    
    # This is building a "csr_matrix" object--compressed sparse row
    vec.fit(feature_dict)
    print(vec)
    print(dir(vec))
    data = vec.transform(feature_dict)
    # test_feat = vec.transform(test_dict)

    print('Vocab sample:            {}'.format(vec.feature_names_[:10]))

    # This is supposed to be a mapping from features to values
    print('Params object:           {}'.format(vec.get_params()))
    print('Feature representation:  {}'.format(type(data).__name__))
    print('Train feature array dim: {dim[0]} x {dim[1]}'.format(dim=data.shape))

    print('Params...')
    # We don't care about params.  Just vocabulary_ and feature_names_
    
    s = nbests[0].sentences[0]

    # This is my representation of features.
    fe = UnigramFE()
    features = fe.extract(s)
    print('Features:')
    print(features)

    # This is a dict representation of features
    feat_dict = features_to_dict(features)  # True values go to 1.0
    print('Feature dict:')
    print(feat_dict)

    # This is a sklearn/numpy representation of features
    feat_vec = vec.transform(feat_dict)
    print('sklearn features:')
    print(feat_vec)
    print(type(feat_vec).__name__)
    print(dir(feat_vec))
    print(feat_vec.shape)
    print(feat_vec.data)

    # .indicies has the feature IDs. we can use them but we're ignoring the 'value'
    print(feat_vec.indices)
    print(feat_vec.indptr)

    for i in feat_vec.indices:
        print(vec.feature_names_[i])
    
    
    # print(vec.feature_names_[33])
    

    

if __name__ == "__main__":
    main()
