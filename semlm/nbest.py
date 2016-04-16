import sys

from semlm.sentence import Sentence


class NBest:

    sentences = None
    id_ = None

    def __init__(self, id_, sentences):
        assert(sentences is not None)
        assert(len(sentences) > 0)
        self.id_ = id_
        self.sentences = sentences
   
    def print_(self):
        out = sys.stdout
        print("ID: {}".format(self.id_))
        for i, s in enumerate(self.sentences):
            print("{:<3d} {:8,.2f}  {:5.2f} -- {}".format(i, s.acscore,
                                                         s.lmscore,
                                                         ' '.join(s.words).lower()),
                  file=out)
