


def pair_to_dict(pair):
    """Convert unigrams to dicts of features.  Valued at 1 and -1."""
    features = {}
    for word in pair[0].words:
        features[word] = 1
    for word in pair[1].words:
        features[word] = -1
    return features
