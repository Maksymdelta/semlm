from semlm.nbest import NBest
from semlm.sentence import Sentence


def read_transcript_table(f):
    trans_table = {}
    trans = read_transcript(f)
    for s in trans:
        trans_table[s.id_] = s
    return trans_table


def read_transcript(f):
    trans = []
    for line in f:
        line = line.strip()
        id_, words = line.split(maxsplit=1)
        s = Sentence(id_, words.split())
        trans.append(s)
    return trans


def read_nbest_file(f):
    "Read a Kaldi n-best file."
    nbest = []
    current_id = None
    prev_id = None
    while True:
        entry = read_nbest_entry_lines(f)  # this is a sentence.
        if not entry:
            break
        id_ = entry[0]
        id_ = id_.rsplit('-', maxsplit=1)[0]
        if prev_id is None: prev_id = id_
        if id_ != prev_id:
            yield NBest(prev_id, nbest)
            nbest = []
            prev_id = id_
        s = entry_lines_to_sentence(entry)
        nbest.append(s)

def read_nbest_entry_lines(f):
    entry_lines = []
    while True:
        line = f.readline()
        if line == '':
            return None
        if line == '\n':
            return entry_lines
        else:
            entry_lines.append(line.strip())


def entry_lines_to_sentence(lines):
    words = []
    lmscores = []
    acscores = []
    id_ = lines.pop(0)
    id_ = id_.rsplit('-', maxsplit=1)[0]
    for line in lines:
        tokens = line.split()
        if len(tokens) == 4:
            s1, s2, word, scores = tokens
            assert(int(s1) == int(s2) - 1)
            score_parts = scores.split(',')
            lmscores.append(float(score_parts[0]))
            acscores.append(float(score_parts[1]))           
            words.append(tokens[2])
    lmscore = sum(lmscores)
    acscore = sum(acscores)
    return Sentence(id_, words, lmscore=lmscore, acscore=acscore)

