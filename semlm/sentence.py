

    

class Sentence:
    def __init__(self, id_, words):
        self.id_ = id_
        self.words = words
        self.acoustic_score = None
        self.lm_score = None

    def __str__(self):
        return "<{} (ac:{}, lm:{} )>".format(self.words.lower(), self.acoustic_score, self.lm_score)
