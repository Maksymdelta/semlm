"""
FEs return maps.  This is in part for compatibilty with sklearn.
"""

import itertools

from itertools import tee
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer


# Should this keep it's own list of features somwhere?

# Should this have its own "fit" function?  Takes an iterator
# over sentences as input?

class FE(object):
    """Generic feature extractor."""

    # Feature vectorizer--maps from features to ints
    vec = None

    def size(self):
        return len(self.vec.vocabulary_)

    def extract_ids(self, s):
        if not self.vec:
            raise Exception("Can't extract feature IDs without a vectorizer.")
        return self.vec.transform(self.extract(s))
        return feature_ids

    def fix(self, features):
        """`features` can be a mapping or an iterable over mappings?"""
        vec = DictVectorizer()
        vec.fit([features])
        self.vec = vec

class CompoundFE(FE):
    fes = []
    
    def __init__(self, fes):
        self.fes = fes

    # How much does something like this help/hurt performance?
    # def extract(self, s):
    #     return itertools.chain.from_iterable(map(lambda x: x.extract(s), self.fes))
    def extract(self, s):
        features = {}
        for fe in self.fes:
            features.update(fe.extract(s))
        return features

        
class UnigramFE(FE):
    """Unigram feature extractor.  Values can be either binary (is the word
    present or not) or counts (how many times does the feature occur)."""

    binary = False

    def __init__(self, binary=False):
        self.binary = binary

    def extract(self, s):
        """Returns a map."""
        features = defaultdict(int)
        for word in s.words:
            if self.binary:
                features[word] = 1
            else:
                features[word] += 1
        return features


class BigramFE(FE):
    binary = False

    def __init__(self, binary=False):
        self.binary = binary

    def extract(self, s):
        """Returns a map."""
        features = defaultdict(int)
        a, b = tee(s.words)
        next(b, None)
        for x, y in zip(a, b):
            feat = ' '.join([x,y])
            if self.binary:
                features[feat] = 1
            else:
                features[feat] += 1
        return features

class TrigramFE(FE):
    binary = False

    def __init__(self, binary=False):
        self.binary = binary

    def extract(self, s):
        """Returns a map."""
        features = defaultdict(int)
        a, b, c = tee(s.words, 3)
        next(b, None)
        next(c, None)
        next(c, None)
        for x, y, z in zip(a, b, c):
            feat = ' '.join([x,y,z])
            if self.binary:
                features[feat] = 1
            else:
                features[feat] += 1
        return features








    
# class NgramFE(FE):
    # I'm not sure if a completely general implementation is possible...?
    # def extract(self, s):
    #     """Returns a map."""
    #     features = defaultdict(int)
    #     n = 3
    #     iterators = tee(s.words, n)
    #     print(iterators)
    #     for i in range(2, n):
    #         for j in range(i):
    #             next(iterators[i], None)
    #     print(iterators)
    #     for ngram in zip(iterators):
    #         print(ngram)
    #         if self.binary:
    #             features[ngram] = 1
    #         else:
    #             features[ngram] += 1
    #     return features


    
