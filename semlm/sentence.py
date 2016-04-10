

class Sentence:
    def __init__(self, id_, words):
        self.id_ = id_
        self.words = words
        self.acoustic_score = None
        self.lm_score = None

    def __str__(self):
        sentence_string = " ".join(self.words).lower()
        return "<{} (ac:{}, lm:{} )>".format(sentence_string,
                                             self.acoustic_score,
                                             self.lm_score)
