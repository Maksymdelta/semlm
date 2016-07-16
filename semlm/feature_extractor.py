from collections import defaultdict

"""
FEs return maps.  This is in part for compatibilty with sklearn.
"""

class FE():
    """Generic feature extractor."""
    pass

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


class ProFE(FE):
    """Takes two sentences and a feature extractor to build examples."""
    
    def extract(self, s1, s2, fe):
        class_ = 1 if s1.wer() < s2.wer() else -1
        features = defaultdict(int)
        for k, v in fe.extract(s1).items():
            features[k] += v * class_
        for k, v in fe.extract(s2).items():
            features[k] += v * (- class_)
        for k, v in list(features.items()):
            if v == 0:
                features.pop(k)
        return features
