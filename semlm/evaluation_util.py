
from editdistance.editdistance import edit_distance


def evaluate(s1, s2):
    print('REF: ' + ' '.join(s1.words).lower())
    print('HYP: ' + ' '.join(s2.words).lower())
    distance, matches = edit_distance(s1.words, s2.words)
    print(distance)
