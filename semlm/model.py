from semlm.features import sent_to_fv

"""
Classes for representing our models.
"""

class lm(object):
    pass

class wslm(lm):
    """The way I'm going to use this, the scores will be added
    to a baseline LM score (coming from the lattice/nbest).

    Model requires:
        vec - Feature vectorizer
        fe - feature extractor
        params - sklearn model/parameters
    """

    vec = None
    fe = None
    params = None

    def __init__(self, vec, fe, params):
        self.vec = vec
        self.fe = fe
        self.params = params

    def score(self, s):
        # Extract features -- re-doing this everytime which may be wasteful...
        fv = sent_to_fv(s, self.fe, self.vec)
        # Compute a score
        product = fv.dot(self.params.T)[0][0]
        return s.score() + product
