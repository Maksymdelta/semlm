



def update_vectorizer(examples, vec):
    # This might be faster without a loop?
    features = map(lambda x: x.features, examples)
    vec.fit(features)

    return None
