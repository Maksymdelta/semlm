import sys

from semlm.sentence import Sentence


class NBest:

    sentences = None
    id_ = None

    def __init__(self, entries):
        id_ = entries[0][0]
        assert(all(map(lambda x: x[0] == id_, entries))) # Make sure all the IDs match
        self.id_ = id_
        assert(all(map(lambda x: x[0] == x[1][1]-1, enumerate(entries))))
        sentences = []
        for entry in entries:
            id_ = entry[0]
            text = entry[2]
            s = Sentence(id_, text)
            sentences.append(s)
        self.sentences = sentences


    def print_(self):
        out = sys.stdout
        print("ID: {}".format(self.id_))
        for i, s in enumerate(self.sentences):
            print("{:<3d} {:8.2f}  {:5.2f} -- {}".format(i, s.acoustic_score, s.lm_score, s.words.lower()), file=out)
