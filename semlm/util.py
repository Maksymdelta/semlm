
import semlm.evaluation_util

from semlm.nbest_util import evaluate_nbests, print_nbest, evaluate_nbests_oracle
from semlm.kaldi import read_transcript_table

def load_references(f, evaluate=False):
    refs = read_transcript_table(f)
    semlm.evaluation_util.REFERENCES = refs

def print_eval(nbests):
    eval = evaluate_nbests(nbests)
    print('Eval:')
    print(eval)
    print('Oracle eval:')
    print(evaluate_nbests_oracle(nbests))

def print_train_test_eval(train_nbests, test_nbests):
    print('Train eval:')
    print_eval(train_nbests)
    print('Test eval:')
    print_eval(test_nbests)

def print_nbests(nbests):
    # Print the n-best lists
    for nbest in nbests:
        print('NBEST:')
        print_nbest(nbest, acscore=True, lmscore=True, tscore=True, maxwords=10, print_instances=True)
