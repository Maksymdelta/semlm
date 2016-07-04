

from semlm.feature_extractor import UnigramFE


class lm():
    pass


# The way I'm going to use this, the scores will be added
# to a baseline LM score (coming from the lattice/nbest)

class wslm():

    # Feature vectorizer
    vec=None

    # sklearn model
    model=None
    def __init__(self):
        self.fe = UnigramFE()

    
    def score(s):
        # - Extract features
        # - Add up all the weights...
        
        


        
        pass

    
