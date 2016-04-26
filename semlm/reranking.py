
def rerank_nbests(nbests, func):
    for nbest in nbests:
        rerank_nbest(nbest, func)

def rerank_nbest(nbest, func):
    sentences = nbest.sentences
    sentences = sorted(sentences, key=func)
    nbest.sentences = sentences
