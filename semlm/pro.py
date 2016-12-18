import random
import itertools

from semlm.example import Example

"""
Create paired examples for pairwise ranking optimization.
"""

def nbest_pairs(nbests):
    """Return an iterator of ALL pairs."""
    for nbest in nbests:
        # Can't we just return this iterator?
        for s1, s2 in itertools.combinations(nbest.sentences, 2):
            yield (s1, s2)

def nbest_pairs_random(nbests, n):
    """Return an iterator of n random pairs from each n-best."""
    for nbest in nbests:
        for i in range(n):
            if len(nbest.sentences) > 1:
                yield(random.sample(nbest.sentences, 2))

def nbest_hyp_best_pairs(nbests):
    """Return an iterator of best and oracle pairs."""
    for nbest in nbests:
        hyp = nbest.hyp()
        best = nbest.oracle_hyp()
        yield(hyp, best)
