

class Sentence:

    def __init__(self, id_, words, acscore=None, lmscore=None):
        self.id_ = id_
        self.words = words
        self.acscore = acscore
        self.lmscore = lmscore
        self.eval_ = None

    def __str__(self):
        sentence_str = " ".join(self.words).lower()
        print_str = '{:8,.2f}  {:5.2f}{} -- {}'
        if self.eval_:
            eval_str = ' {:5.0%}'.format(self.eval_.wer())
        else:
            eval_str = ''
        print_str = print_str.format(self.acscore,
                                     self.lmscore,
                                     eval_str,
                                     sentence_str)
        return print_str
