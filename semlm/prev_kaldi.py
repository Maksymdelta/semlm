from collections import OrderedDict


def nbest_entries_to_nbests(entries):
    id_map = OrderedDict()
    nbests = []
    for entry in entries:
        id_map[entry.id_] = id_map.get(entry.id_, []) + [entry]
    for id_, entries in id_map.items():
        nbest = NBest(id_, entries)
        nbests.append(nbest)
    return nbests
