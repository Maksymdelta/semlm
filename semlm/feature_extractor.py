"""
FEs return maps.  This is in part for compatibilty with sklearn.
"""

from itertools import tee
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer


# Should this keep it's own list of features somwhere?

# Should this have its own "fit" function?  Takes an iterator
# over sentences as input?

class FE(object):
    """Generic feature extractor."""

    # Feature vectorizer--maps from features to ints
    def __init__(self):
        """Set the vectorizer to None."""
        self.vec = None

    def extract(self, s):
        """An abstract method for extracting features (non-integer)."""
        raise NotImplementedError()

    def size(self):
        """The size of the featurizer's vocabulary (i.e. how many features
        this FE could produce)."""
        return len(self.vec.vocabulary_)

    def extract_ids(self, s):
        """Given a sentence object return a sequence of feature IDs for that
        sentence."""
        if not self.vec:
            raise Exception("Can't extract feature IDs without a vectorizer.")
        return self.vec.transform(self.extract(s))

    def fix(self, features):
        """`features` can be a mapping or an iterable over mappings?"""
        vec = DictVectorizer()
        vec.fit([features])
        self.vec = vec

class CompoundFE(FE):
    """A compound feature extractor, contains multiple feature extractors and
    returns the set of features produced by all of them."""

    def __init__(self, fes):
        """Initialize with a sequence of feature extractors."""
        super(CompoundFE, self).__init__()
        self.fes = fes

    # How much does something like this help/hurt performance?
    # def extract(self, s):
    #     return itertools.chain.from_iterable(map(lambda x: x.extract(s), self.fes))
    def extract(self, s):
        """Extract (non-int) features from sentence `s`."""
        features = {}
        for fe in self.fes:
            features.update(fe.extract(s))
        return features

class UnigramFE(FE):
    """Unigram feature extractor.  Values can be either binary (is the word
    present or not) or counts (how many times does the feature occur)."""

    def __init__(self, binary=False):
        super(UnigramFE, self).__init__()
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
    """Bigram feature extractor."""

    def __init__(self, binary=False):
        super(BigramFE, self).__init__()
        self.binary = binary

    def extract(self, s):
        """Returns a map."""
        features = defaultdict(int)
        a, b = tee(s.words)
        next(b, None)
        for x, y in zip(a, b):
            feat = ' '.join([x, y])
            if self.binary:
                features[feat] = 1
            else:
                features[feat] += 1
        return features

class TrigramFE(FE):
    """Trigram feature extractor."""

    def __init__(self, binary=False):
        super(TrigramFE, self).__init__()
        self.binary = binary

    def extract(self, s):
        """Returns a map."""
        features = defaultdict(int)
        a, b, c = tee(s.words, 3)
        next(b, None)
        next(c, None)
        next(c, None)
        for x, y, z in zip(a, b, c):
            feat = ' '.join([x, y, z])
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
