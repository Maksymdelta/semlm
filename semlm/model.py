"""
Classes for representing language models.
"""

class LM(object):
    """Represents a language model."""
    pass

class WSLM(LM):
    """The way I'm going to use this, the scores will be added
    to a baseline LM score (coming from the lattice/nbest).

    Model requires:
        vec - Feature vectorizer
        fe - feature extractor
        params - sklearn model/parameters
    """

    def __init__(self, vec, fe, params, lmwt=14):
        self.vec = vec
        self.fe = fe
        self.params = params
        self.lmwt = lmwt

    # TODO: Is there anything dumb and slow going on here?
    # Don't have to recompute this if the parameter vector hasn't changed...
    # But what's a good place to save the values?
    def score(self, s):
        """Compute the model's score for the given sentence."""
        fv = s.feature_vector
        product = fv.dot(self.params.T)[0, 0]
        return s.score(lmwt=self.lmwt) + product

    def print_feature_weights(self, max_=None, threshold=None):
        """Print up to `max_` feature weights, for weights above the given `threshold`."""
        print('Feature weights:')
        feature_weights = []
        for i in range(len(self.vec.get_feature_names())):
            name = self.vec.get_feature_names()[i]
            val = self.params[0, i]
            if not threshold or abs(val) >= threshold:
                feature_weights.append((name, val))

        items = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
        if max_:
            items = items[:max_]

        for name, val in items:
            print('{:20} {:>8.2f}'.format(str(name), val))
