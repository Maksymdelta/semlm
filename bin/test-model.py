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
import colorama   # Not using this yet...

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron

from asr_tools.kaldi import read_nbest_file
from asr_tools.evaluation_util import set_global_references
from asr_tools.nbest_util import print_train_test_eval, evaluate_nbests
from asr_tools.reranking import rerank_nbests
from asr_tools.sentence_util import print_sentence

from semlm.feature_extractor import UnigramFE
from semlm.model import wslm
from semlm.pro import create_pro_examples
from semlm.sklearn import examples_to_matrix, print_feature_weights



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    return parser.parse_args()

def print_data_info(vec, train_data, test_data):
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


    # print(evaluate_nbests(train_nbests))
    evaluate_nbests(nbests)

    # Create train/test examples
    fe = UnigramFE()
    train_examples = create_pro_examples(train_nbests, fe)
    test_examples = create_pro_examples(test_nbests, fe)
    print('\n# of train examples: {}'.format(len(train_examples)))
    print('# of test examples: {}'.format(len(test_examples)))

    # Convert everything into feature IDs.  We have to featurize them all together before
    # we transform them. These next three could be factored out?
    vec = DictVectorizer()
    feature_dicts = map(lambda x: x.features, train_examples + test_examples)
    vec.fit(feature_dicts)
    train_data, train_classes = examples_to_matrix(train_examples, vec)
    test_data, test_classes = examples_to_matrix(test_examples, vec)

    print_data_info(vec, train_data, test_data)

    # Train a perceptron or other model. e.g. Perceptron, SGDClassifier, LinearRegression
    print('\nTraining model:')
    # model = LogisticRegression(verbose=10, penalty='l2', C=1.0)
    model = Perceptron(verbose=10, eta0=1.0, n_iter=10)  # penalty='l2')
    model.fit(train_data, train_classes)

    # Print feature weights
    print_feature_weights(model, vec)

    # Print pairwise evaluation
    print(evaluate_nbests(train_nbests))
    print(evaluate_nbests(test_nbests))

    # At this point we want to create a model with the feature weights...
    # Then we'll want to re-rank with the model, and re-evaluate
    # Need a sentence to feature vector function...
    # Feature vector is a csr_matrix object (scipy)    
    lm = wslm(vec, fe, model.coef_)
    print('Re-ranking n-best lists')
    func = lambda x: lm.score(x)
    # The re-ranking ops appear that they are distructive
    rerank_nbests(train_nbests, func)
    rerank_nbests(test_nbests, func)
    print(evaluate_nbests(train_nbests))
    print(evaluate_nbests(test_nbests))


        
if __name__ == "__main__":
    main()


