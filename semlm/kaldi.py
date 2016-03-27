# import semlm.nbest
# import semlm.sentence
from collections import OrderedDict
from semlm.nbest import NBest


class NbestFileEntry():
    def __init__(self, line):
        line = line.strip()
        tokens = line.split()
        id_ = tokens[0]
        self.words = tokens[1:]
        self.id_, self.i = id_.rsplit('-', 1)
        self.i = int(self.i)


def read_nbest_file(filename):
    entries = read_nbest_file_entries(filename)
    nbests = nbest_entries_to_nbests(entries)
    print(nbests)


def read_nbest_file_entries(filename):
    "Read a Kaldi n-best file."
    entries = []
    with open(filename) as f:
        for line in f:
            entry = NbestFileEntry(line)
            entries.append(entry)
    return entries


def nbest_entries_to_nbests(entries):
    id_map = OrderedDict()
    nbests = []
    for entry in entries:
        id_map[entry.id_] = id_map.get(entry.id_, []) + [entry]
    for id_, entries in id_map.items():
        nbest = NBest(id_, entries)
        nbests.append(nbest)
    return nbests
