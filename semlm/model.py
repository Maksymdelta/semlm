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
        fv = s.feature_vector
        product = fv.dot(self.params.T)[0,0]
        return s.score() + product


    def print_feature_weights(self):
        print('Feature weights:')
        feature_weights = []
        for i in range(len(self.vec.get_feature_names())):
            name = self.vec.get_feature_names()[i]
            val = self.params[0,i]
            feature_weights.append((name, val))
        for name, val in sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True):
            print('{:20} {:>8.2f}'.format(name, val))
            
            

    
