import itertools

from semlm.feature_extractor import ProFE
from semlm.example import Example

"""
Create paired examples for pairwise ranking optimization.
"""

def create_pro_examples(nbests, fe):
    """Create train/test examples for PRO."""
    examples = []
    pro_fe = ProFE()
    for nbest in nbests:
        for s1, s2 in itertools.combinations(nbest.sentences, 2):
            features = pro_fe.extract(s1, s2, fe)
            class_ = 1 if s1.wer() < s2.wer() else -1
            example = Example(class_, features)
            examples.append(example)
    return examples


def nbest_pairs(nbests):
    for nbest in nbests:
        for s1, s2 in itertools.combinations(nbest.sentences, 2):
            yield (s1, s2)

