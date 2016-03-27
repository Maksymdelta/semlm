
from semlm.sentence import Sentence


class NBest:

    sentences = None

    def __init__(self, id_, entries):
        entries.sort(key=lambda x: x.i)
        sentences = []
        for entry in entries:
            s = Sentence(entry.id_, entry.words)
            sentences.append(s)
        self.sentences = sentences
