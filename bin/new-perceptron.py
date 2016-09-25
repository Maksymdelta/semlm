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
import logging
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from asr_tools.kaldi import read_nbest_file
from asr_tools.evaluation_util import set_global_references
from asr_tools.nbest_util import evaluate_nbests
from asr_tools.reranking import rerank_nbests

from semlm.feature_extractor import UnigramFE
from semlm.model import wslm

from semlm.pro import nbest_pairs
from semlm.perceptron import perceptron

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


def feature_extract_sents(sentences):
    fe = UnigramFE()
    feat_dict_list = []
    for sent in sentences:
        feats = fe.extract(sent)
        sent.features = feats
        feat_dict_list.add(feats)
    vec = DictVectorizer()
    vec.fit(feat_dict_list)
    for sent in sentences:
        sent.fv = vec.transform(sent.features)


def main():
    args = parse_args()
    colorama.init()
    # logger = logging.getLogger('asr_tools')

    # Read the n-best lists and references
    nbests = list(read_nbest_file(args.nbest_file))
    train_nbests = nbests[:len(nbests) // 2]
    test_nbests = nbests[len(nbests) // 2:]
    print()
    print('Training/test/total nbests: {}/{}/{}'.format(len(train_nbests),
                                                        len(test_nbests),
                                                        len(nbests)))
    set_global_references(args.ref_file)

    print("Evaluating train n-bests...")
    print(evaluate_nbests(train_nbests))
    print("Evaluating test n-bests...")
    print(evaluate_nbests(test_nbests))
    print("Evaluating all (just in case)...?")
    evaluate_nbests(nbests)

    # Do feature extraction.
    # Need a better abstraction for feature extraction I think...
    # Should be able to do something like extract_features(s1)
    print("Extracting features...")
    fe = UnigramFE()
    features_list = []
    for nbest in nbests:
        for sentence in nbest.sentences:
            features = fe.extract(sentence)
            features_list.extend([features])

    print("Vectorizing features...")
    vec = DictVectorizer()
    vec.fit(features_list)
    # Give the feature extractor the vectorizer
    fe.set_vec(vec)

    # Now that we have a feature vectorizer, we can extract feature IDs
    print('Extracting feature IDs...')
    for nbest in nbests:
        for sentence in nbest.sentences:
            feature_ids = fe.extract_ids(sentence)
            sentence.feature_vector = feature_ids
            
    # Need an initial set of weights...
    print('Initializing model...')
    params = np.zeros((1, len(vec.vocabulary_)))
    model = wslm(vec, fe, params)

    for e in range(1):
        print('Epoch: {}'.format(e))
        # Creates an iterator of pairs...
        pair_iter = nbest_pairs(train_nbests)
        perceptron(pair_iter, model)
        print('Re-ranking n-best lists...')
        func = lambda x: model.score(x)
        # The re-ranking ops appear that they are distructive
        rerank_nbests(train_nbests, func)
        rerank_nbests(test_nbests, func)
        print("Train evaluation:")
        print(evaluate_nbests(train_nbests))
        print("Test evaluation:")
        print(evaluate_nbests(test_nbests))

    model.print_feature_weights()
    # Rerank with the model and print results...
    # Iterate.
    
    # Create train/test examples
    # fe = UnigramFE()
    # train_examples = create_pro_examples(train_nbests, fe)
    # test_examples = create_pro_examples(test_nbests, fe)
    # logger.critical('TEST')
        
    # print('\n# of train examples: {}'.format(len(train_examples)))
    # print('# of test examples: {}'.format(len(test_examples)))

    # # Convert everything into feature IDs.  We have to featurize them all together before
    # # we transform them. These next three could be factored out?
    # vec = DictVectorizer()
    # feature_dicts = map(lambda x: x.features, train_examples + test_examples)
    # vec.fit(feature_dicts)
    # train_data, train_classes = examples_to_matrix(train_examples, vec)
    # test_data, test_classes = examples_to_matrix(test_examples, vec)
    # print_data_info(vec, train_data, test_data)

    # # Train a perceptron or other model. e.g. Perceptron, SGDClassifier, LinearRegression
    # print('\nTraining model:')
    # # model = LogisticRegression(verbose=10, penalty='l2', C=1.0)
    # model = Perceptron(verbose=10, eta0=1.0, n_iter=10)  # penalty='l2')
    # model.fit(train_data, train_classes)

    # # Print feature weights
    # print_feature_weights(model, vec)

    # # Print pairwise evaluation
    # print(evaluate_nbests(train_nbests))
    # print(evaluate_nbests(test_nbests))

    # # At this point we want to create a model with the feature weights...
    # # Then we'll want to re-rank with the model, and re-evaluate
    # # Need a sentence to feature vector function...
    # # Feature vector is a csr_matrix object (scipy)    
    # lm = wslm(vec, fe, model.coef_)
    # print('Re-ranking n-best lists')
    # func = lambda x: lm.score(x)
    # # The re-ranking ops appear that they are distructive
    # rerank_nbests(train_nbests, func)
    # rerank_nbests(test_nbests, func)
    # print(evaluate_nbests(train_nbests))
    # print(evaluate_nbests(test_nbests))


        
if __name__ == "__main__":
    main()




