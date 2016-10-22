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
import logging
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from asr_tools.kaldi import read_nbest_file
from asr_tools.evaluation_util import set_global_references
from asr_tools.nbest_util import evaluate_nbests
from asr_tools.reranking import rerank_nbests

from semlm.feature_extractor import UnigramFE
from semlm.model import WSLM

# These are doing the hard work...
from semlm.pro import nbest_pairs, nbest_pairs_random, nbest_hyp_best_pairs
from semlm.perceptron import perceptron

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nbest_file')
    parser.add_argument('ref_file', type=argparse.FileType('r'))
    pickle_group = parser.add_mutually_exclusive_group()
    pickle_group.add_argument('--load-pickle', action='store_true', default=False)
    pickle_group.add_argument('--save-pickle', action='store_true', default=False)
    parser.add_argument('-n', '--nbest_sample_size', type=int, default=100)
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-r', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-s', '--selection-mode', choices=['all', 'random', 'hyp_best'])
    parser.add_argument('-f', '--print-features', action='store_true', default=False)
    return parser.parse_args()

def print_data_info(vec, train_data, test_data):
    print('Vocab sample:            {}'.format(vec.feature_names_[:10]))
    print('Params object:           {}'.format(vec.get_params()))
    print('Feature representation:  {}'.format(type(train_data).__name__))
    print('Feature representation:  {}'.format(type(test_data).__name__))
    print('Train feature array dim: {dim[0]} x {dim[1]}'.format(dim=train_data.shape))
    print('Test feature array dim:  {dim[0]} x {dim[1]}'.format(dim=test_data.shape))

def feature_extract_sents(sentences):
    """Run feature extraction on the given set of sentences."""
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
    # logger = logging.getLogger('semlm')
    # logger.critical('TEST')
    set_global_references(args.ref_file)
    
    if args.load_pickle:
        print("Reading n-bests from pickle file...")
        with open(args.nbest_file, 'rb') as f:
            nbests = pickle.load(f)
    else:
        with open(args.nbest_file, 'r') as f:
            print("Reading n-bests from text...")
            nbests = list(read_nbest_file(f))
            print("Evaluating n-bests...")
            evaluate_nbests(nbests)

    if args.save_pickle:
        with open(args.nbest_file + '.pickle', 'wb') as f:
            pickle.dump(nbests, f)

    train_nbests = nbests[0:len(nbests) // 2]
    test_nbests = nbests[len(nbests) // 2:]
    
    print()
    print('Training/test/total nbests: {}/{}/{}'.format(len(train_nbests),
                                                        len(test_nbests),
                                                        len(nbests)))
    print("Evaluating train n-bests...")
    print(evaluate_nbests(train_nbests))
    print("Evaluating test n-bests...")
    print(evaluate_nbests(test_nbests))
    
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

    # Now we have a giant list of dicts representing all the features in all of the n-bests
            
    print("Vectorizing features...")
    vec = DictVectorizer()
    vec.fit(features_list)
    # Give the feature extractor the vectorizer, so it can return int features
    fe.set_vec(vec)

    # Now that we have a feature vectorizer, we can extract feature IDs
    # This gives the sentence its IDs.
    print('Extracting feature IDs...')
    for nbest in nbests:
        for sentence in nbest.sentences:
            feature_ids = fe.extract_ids(sentence)
            sentence.feature_vector = feature_ids
            
    # Need an initial set of weights and initial model
    print('Initializing model...')
    params = np.zeros((1, len(vec.vocabulary_)))
    model = WSLM(vec, fe, params)

    # Do an initial scoring and re-ranking
    func = lambda x: model.score(x)
    # The re-ranking ops appear that they are destructive
    rerank_nbests(train_nbests, func)
    rerank_nbests(test_nbests, func)        
    print('INITIAL EVAL:')
    print("Train evaluation:")
    print(evaluate_nbests(train_nbests))
    print("Test evaluation:")
    print(evaluate_nbests(test_nbests))

    print('=======')
    print('Beginning training...')
    for e in range(args.epochs):
        print('Epoch: {}'.format(e+1))
        # Creates an iterator of pairs...
        # Some of these will have to be reranked for the next iteration to do anything
        if args.selection_mode == 'all':
            pair_iter = nbest_pairs(train_nbests)
        elif args.selection_mode == 'random':
            pair_iter = nbest_pairs_random(train_nbests, args.nbest_sample_size)
        elif args.selection_mode == 'hyp_best':
            pair_iter = nbest_hyp_best_pairs(train_nbests)
        else:
            raise Exception('Unknown selection method: {}'.format(args.selection_mode))
        perceptron(pair_iter, model, rate=args.learning_rate)
        print('Re-ranking n-best lists...')
        func = lambda x: model.score(x)
        # The re-ranking ops appear that they are destructive
        rerank_nbests(train_nbests, func)
        rerank_nbests(test_nbests, func)
        print("Train evaluation:")
        print(evaluate_nbests(train_nbests))
        print("Test evaluation:")
        print(evaluate_nbests(test_nbests))
        print('=======')


    # Do a final scoring and re-ranking
    func = lambda x: model.score(x)
    # The re-ranking ops appear that they are distructive
    rerank_nbests(train_nbests, func)
    rerank_nbests(test_nbests, func)

    print('=======')
    print('FINAL EVAL:')
    print("Train evaluation:")
    print(evaluate_nbests(train_nbests))
    print("Test evaluation:")
    print(evaluate_nbests(test_nbests))
        
    if args.print_features:
        model.print_feature_weights(max=100)

        
if __name__ == "__main__":
    main()




