

class Sentence:
    def __init__(self, id_, words, acscore=None, lmscore=None):
        self.id_ = id_
        self.words = words
        self.acscore = acscore
        self.lmscore = lmscore

    def __str__(self):
        sentence_string = " ".join(self.words).lower()
        return "<{} (ac: {:,.2f}, lm: {:,.2f} )>".format(sentence_string,
                                             self.acscore,
                                             self.lmscore)
