
import asr_tools.evaluation_util

from semlm.features import generate_training_pairs
from semlm.features import pair_to_dict
from asr_tools.nbest_util import evaluate_nbests, print_nbest, evaluate_nbests_oracle
from asr_tools.kaldi import read_transcript_table

def load_references(f, evaluate=False):
    refs = read_transcript_table(f)
    asr_tools.evaluation_util.REFERENCES = refs

def print_eval(nbests):
    eval = evaluate_nbests(nbests)
    print('Eval:')
    print(eval)
    print('Oracle eval:')
    print(evaluate_nbests_oracle(nbests))

def print_train_test_eval(train_nbests, test_nbests):
    print()
    print('Train eval:')
    print_eval(train_nbests)
    print()
    print('Test eval:')
    print_eval(test_nbests)

def print_nbests(nbests):
    # Print the n-best lists
    for nbest in nbests:
        print('NBEST:')
        print_nbest(nbest, acscore=True, lmscore=True, tscore=True, maxwords=10, print_instances=True)

def extract_dict_examples(nbests, vec):
    # Convert the n-best lists to training examples.
    pairs, classifications = generate_training_pairs(nbests)
    print(len(pairs))
    print(len(classifications))
    feature_dicts = list(map(pair_to_dict, pairs))
    print('# of pairs:    {}'.format(len(pairs)))
    assert(len(feature_dicts) == len(pairs) == len(classifications))
    return feature_dicts, classifications
