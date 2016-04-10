# import semlm.nbest
# import semlm.sentence

from semlm.nbest import NBest

def read_nbests(filename):
    "Read a Kaldi n-best file."
    nbest = []
    current_id = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            id_rank, word_string =  line.split(maxsplit=1)
            id_, rank = id_rank.rsplit('-', maxsplit=1)
            if current_id == None: current_id = id_
            if id_ != current_id:
                yield(NBest(nbest))
                nbest = []
                current_id = id_
            nbest.append((id_, int(rank), word_string))
        if len(nbest) > 0:
            yield(NBest(nbest))


def read_score_file(filename):
    "Read a Kaldi score file."
    scores = []
    current_id = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            id_rank, score_string =  line.split(maxsplit=1)
            id_, rank = id_rank.rsplit('-', maxsplit=1)
            if current_id == None: current_id = id_
            if id_ != current_id:
                yield(scores)
                scores = []
                current_id = id_
            scores.append((id_, int(rank), float(score_string)))
        yield(scores)

        
def read_nbest_files(nbest_filename, acscore_filename, lmscore_filename):

    for nbest, acscore, lmscore in zip(read_nbests(nbest_filename),
                                       read_score_file(acscore_filename),
                                       read_score_file(lmscore_filename)):
        assert(len(nbest.sentences) == len(acscore) == len(lmscore))
        for s, a, l in zip(nbest.sentences, acscore, lmscore):
            s.lm_score = l[2]
            s.acoustic_score = a[2]        
        yield nbest


def read_ref(filename):
    refs = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            id_, words =  line.split(maxsplit=1)
            refs[id_] = words
    return refs


        
