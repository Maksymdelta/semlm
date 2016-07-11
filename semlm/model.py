


from semlm.features import sent_to_fv


class lm():
    pass


# The way I'm going to use this, the scores will be added
# to a baseline LM score (coming from the lattice/nbest)

class wslm(lm):

    # Feature vectorizer, feature extractor, sklearn model/parameters
    vec = None
    fe = None
    params = None

    def __init__(self, vec, fe, params):
        self.vec = vec
        self.fe = fe
        self.params = params

    def score(self, s):
        # Extract features
        fv = sent_to_fv(s, self.fe, self.vec)
        # Compute a score
        product = fv.dot(self.params.T)[0][0]
        return s.score() + product
