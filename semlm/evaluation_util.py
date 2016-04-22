from semlm.evaluation import Evaluation
from editdistance.editdistance import edit_distance
from asr_evaluation.asr_evaluation import print_diff as eval_print_diff
from editdistance.editdistance import SequenceMatcher


REFERENCES = {}


def sentence_editdistance(s1, s2):
    """Given two 'sentence' objects compute the edit distance and
    return the distance."""
    distance, matches = edit_distance(s1.words, s2.words)
    return distance


def evaluate(ref_table, s):
    """Given a sentence and a reference table, create and return an
    Evaluation object."""
    ref = ref_table[s.id_]
    distance, matches = edit_distance(ref.words, s.words)
    eval_ = Evaluation(len(ref.words), matches, distance)
    return eval_


def set_global_references(ref_file):
    """Given a reference file, read it into the global table wit the name
    REFERECENS."""
    REFERENCES = read_transcript_table(ref_file)


def get_global_reference(id_):
    """Look-up ASR references by id."""
    return REFERENCES.get(id_)


def print_diff(s1, s2):
    """Print a readable diff between two sentences."""
    a = s1.words
    b = s2.words
    sm = SequenceMatcher(a, b)
    eval_print_diff(sm, s1.words, s2.words)
