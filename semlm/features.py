import semlm.feature_extractor

"""
Helper functions related to feature extraction.
"""

def sent_to_fv(sentence, fe, vec):
    """To turn it into something we can use with sklearn have to use
    vectorizer and call vec.transform()."""
    feats = fe.extract(sentence)
    feat_dicts = features_to_dict(feats)
    fv = vec.transform(feat_dicts)
    return fv

def features_to_dict(features, value=True):
    """The sk-learn feature extractor (i.e. DictVectorizer) needs dicts as inputs,
    this converts a list of feature 'names/identities' to a dict mapping to the
    given value.  By default the value is 'True' which seems to get converted into
    the number 1."""
    dict_ = {}
    for f in features:
        dict_[f] = value
    return dict_
