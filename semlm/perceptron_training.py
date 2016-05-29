

def generate_training_pairs(nbests):
    """Generate pairs of sentences where the pair can be used as a training example.
    This is a naive, non-scalable way to do this."""
    pairs=[]
    classifications = []
    for nbest in nbests:
        for s1 in nbest.sentences:
            for s2 in nbest.sentences:
                if s1 is not s2:
                    if s1.wer() != s2.wer():
                        pairs.append((s1, s2))
                        classifications.append(1 if s1.wer() < s2.wer() else -1)
    return pairs, classifications
