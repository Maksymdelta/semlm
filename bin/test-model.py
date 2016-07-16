#!/usr/bin/env python3

"""
Given some nbest lists and a reference...

 - Split the nbests into a train and test set.
 - Create PRO train and test sets.
 - Vectorize everything into numeric features.
 - Train an sk-learn model, and print pairwise accuracy.
 - Create a WSLM.
"""

# Let's factor stuff out of this file to clean it up.

import argparse
import colorama

from asr_tools.kaldi import read_nbest_file
from asr_tools.evaluation_util import set_global_references
from asr_tools.util import print_train_test_eval


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from semlm.feature_extractor import UnigramFE
from semlm.sklearn import examples_to_matrix
from semlm.sklearn import print_feature_weights
from semlm.pro import create_pro_examples
from semlm.model import wslm


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
    set_global_references(args.ref_file)

    # Print evaluation
    print_train_test_eval(train_nbests, test_nbests)

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
    model = LogisticRegression(verbose=10, penalty='l2', C=1.0)
    model = Perceptron(verbose=10, eta0=1.0, n_iter=10)  # penalty='l2')
    model.fit(train_data, train_classes)

    # Print feature weights
    print_feature_weights(model, vec)

    # Print evaluation
    print_train_test_eval(train_nbests, test_nbests)

    # At this point we want to create a model with the feature weights...
    # Then we'll want to re-rank with the model, and re-evaluate
    # Need a sentence to feature vector function...

    s = train_nbests[0].sentences[0]
    # Feature vector is a csr_matrix object (scipy)
    lm = wslm(vec, fe, model.coef_)
    print(lm.score(s))

    # NEXT: RERANK WITH THIS MODEL


    
if __name__ == "__main__":
    main()


