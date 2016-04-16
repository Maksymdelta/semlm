
from editdistance.editdistance import edit_distance
from semlm.evaluation import Evaluation


def sentence_editdistance(s1, s2):
    distance, matches = edit_distance(s1.words, s2.words)
    return distance


def evaluate(ref_table, s):
    ref = ref_table[s.id_]
    distance, matches = edit_distance(ref.words, s.words)
    eval_ = Evaluation(len(ref.words), matches, distance)
    return eval_
