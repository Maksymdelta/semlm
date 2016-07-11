#!/usr/bin/env python3

import argparse
import colorama

from asr_tools.kaldi import read_nbest_file

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron

from semlm.feature_extractor import UnigramFE
from semlm.sklearn import print_feature_weights, evaluate_model, examples_to_matrix
from semlm.util import load_references
from semlm.util import print_train_test_eval
from semlm.pro import create_pro_examples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    return parser.parse_args()

def print_info(vec, train_data, test_data):
    print('Vocab sample:            {}'.format(vec.feature_names_[:10]))
    print('Params object:           {}'.format(vec.get_params()))
    print('Feature representation:  {}'.format(type(train_data).__name__))
    print('Feature representation:  {}'.format(type(test_data).__name__))
    print('Train feature array dim: {dim[0]} x {dim[1]}'.format(dim=train_data.shape))
    print('Test feature array dim:  {dim[0]} x {dim[1]}'.format(dim=test_data.shape))


def main():
    args = parse_args()

    colorama.init()
    # Read the n-best lists and references
    nbests = list(read_nbest_file(args.nbest_file))
    train_nbests = nbests[:len(nbests) // 2]
    test_nbests = nbests[len(nbests) // 2:]
    print()
    print('Training/test/total nbests: {}/{}/{}'.format(len(train_nbests),
                                                        len(test_nbests),
                                                        len(nbests)))
    load_references(args.ref_file)

    # Print evaluation
    print_train_test_eval(train_nbests, test_nbests)
    # print_nbests(nbests)

    # Create train/test examples
    fe = UnigramFE()
    train_examples = create_pro_examples(train_nbests, fe)
    test_examples = create_pro_examples(test_nbests, fe)

    # Converts the Example objects to sklearn objects (matrices)
    print()
    print('# of train examples: {}'.format(len(train_examples)))
    print('# of test examples: {}'.format(len(test_examples)))

    # Convert everything into feature IDs
    # These next three should be factored out
    vec = DictVectorizer()
    feature_dicts = map(lambda x: x.features, train_examples + test_examples)
    vec.fit(feature_dicts)
    train_data, train_classes = examples_to_matrix(train_examples, vec)
    test_data, test_classes = examples_to_matrix(test_examples, vec)

    print_info(vec, train_data, test_data)

    # Train a perceptron or other model. e.g. Perceptron, SGDClassifier, LinearRegression
    print()
    print('Training model:')
    # model = LogisticRegression(verbose=10, penalty='l2', C=1.0)
    model = Perceptron(verbose=10, eta0=0.1, n_iter=10)  # penalty='l2')
    model.fit(train_data, train_classes)

    # Print feature weights and do a pairwise evaluation of the model on training data.
    print_feature_weights(model, vec)
    print()
    print('Eval on train data:')
    evaluate_model(model, (train_data, train_classes))
    print()
    print('Eval on test data:')
    evaluate_model(model, (test_data, test_classes))


if __name__ == "__main__":
    main()
