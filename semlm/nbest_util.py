import semlm.evaluation_util
from semlm.evaluation_util import evaluate
from semlm.evaluation_util import sum_evals

def nbest_oracle_sort(nbest, n=None):
    nbest = nbest.copy()
    if n: nbest.sentences = nbest.sentences[:n]
    nbest.sentences = sorted(nbest.sentences, key=lambda x: x.eval_.wer())
    return nbest

def nbest_best_sentence(nbest, n=None):
    sentences = nbest.sentences
    if n: sentences = sentences[:n]
    return min(sentences, key=lambda x: x.wer())

def nbest_oracle_eval(nbest, n=None):
    return nbest_best_sentence(nbest, n=n).eval_

# This does more than its name implies
def evaluate_nbest(nbest, force=False):
    id_ = nbest.id_
    for s in nbest.sentences:
        if force or s.eval_ is None:    # Only recompute the evaluation if not already computed.
            e = evaluate(semlm.evaluation_util.REFERENCES, s)
            s.eval_ = e
    return nbest.sentences[0].eval_

def evaluate_nbests(nbests):
    evals = list(map(evaluate_nbest, nbests))
    return sum_evals(evals)

def evaluate_nbests_oracle(nbests):
    evals = list(map(nbest_oracle_eval, nbests))
    return sum_evals(evals)

def evals_by_depth(nbests, n=100):
    evals_by_depth = [None] * n
    for i in range(n):
        evals = []
        for nbest in nbests:
            evals.append(nbest_oracle_eval(nbest, i+1))
        evals_by_depth[i] = sum_evals(evals)
    return evals_by_depth
