import semlm.feature_extractor

# WIP


def sent_to_fv(sentence, fe, vec):
    """To turn it into something we can use with sklearn have to use
    vectorizer and call vec.transform()."""
    feats = fe.extract(sentence)
    feat_dicts = features_to_dict(feats)
    fv = vec.transform(feat_dicts)
    return fv

def features_to_dict(features, value=True):
    dict_ = {}
    for f in features:
        dict_[f] = value
    return dict_

def pair_to_dict(pair):
    """Convert unigrams to dicts of features.  Valued at 1 and -1."""
    fe = semlm.feature_extractor.UnigramFE()
    features = {}
    for f in fe.extract(pair[0]).keys():
        features[f] = 1
    for f in fe.extract(pair[1]).keys():
        features[f] = -1
    return features

def generate_training_pairs(nbests):
    """Generate pairs of sentences where the pair can be used as a training example.
    This is a naive, non-scalable way to do this.

    This returns two aligned lists.  The first is a list of tuples, pairs of sentence pairs.
    The second is a list of 'classifications', where 1 means the first of the pair has lower
    WER while -1 means the second has lower WER."""
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
