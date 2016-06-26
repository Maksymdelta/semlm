

# WIP

def pair_to_dict(pair):
    """Convert unigrams to dicts of features.  Valued at 1 and -1."""
    features = {}
    for word in pair[0].words:
        features[word] = 1
    for word in pair[1].words:
        features[word] = -1
    return features


def generate_training_pairs(nbests):
    """Generate pairs of sentences where the pair can be used as a training example.
    This is a naive, non-scalable way to do this."""
    pairs = []
    classifications = []
    for nbest in nbests:
        for s1 in nbest.sentences:
            for s2 in nbest.sentences:
                if s1 is not s2:
                    if s1.wer() != s2.wer():
                        pairs.append((s1, s2))
                        classifications.append(1 if s1.wer() < s2.wer() else -1)
    return pairs, classifications


