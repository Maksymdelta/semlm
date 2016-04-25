

class Sentence:
    """Represents a sentence or utterance."""

    def __init__(self, id_, words, acscore=None, lmscore=None):
        """Constructor.  ID and words are required."""
        self.id_ = id_
        self.words = words
        self.acscore = acscore
        self.lmscore = lmscore
        self.eval_ = None

    def __str__(self):
        """Returns a string representation of this object."""
        sentence_str = ' '.join(self.words).lower()
        print_str = '{:8,.2f}  {:5.2f}{} -- {}'
        if self.eval_:
            eval_str = ' {:5.0%}'.format(self.eval_.wer())
        else:
            eval_str = ''
        print_str = print_str.format(self.acscore if self.acscore else 0.0,
                                     self.lmscore if self.lmscore else 0.0,
                                     eval_str,
                                     sentence_str)
        return print_str

    def wer(self):
        return self.eval_.wer()
