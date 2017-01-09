"""
Simple `numpy`-based implementation of Perceptron.
"""

from numpy.linalg import norm

def perceptron(pairs, model, print_progress=False, rate=0.01):
    """Train the given model on the given pairs of sentences."""
    counter = 0
    for pair in pairs:
        counter += 1
        perceptron_update(pair, model, rate=rate)
        if print_progress and counter % 10000 == 0:
            print('{} pairs...'.format(counter))
            print('Model - norm: {}'.format(norm(model.params)))

def perceptron_update(pair, model, rate=0.01):
    """Perform the perceptron update on the given model using the
    given pair of sentences."""
    s1, s2 = pair
    score1 = model.score(s1)
    score2 = model.score(s2)
    wer1 = s1.wer()
    wer2 = s2.wer()
    score_diff = score1 - score2
    wer_diff = wer1 - wer2
    # same decision?
    if score_diff * wer_diff >= 0:
        # do nothing, they had the same classification
        pass
    else:
        # they had different classifications, so update the model.
        s1_vector = s1.feature_vector * wer_diff * rate
        s2_vector = s2.feature_vector * wer_diff * rate
        new_params = model.params + s1_vector - s2_vector
        model.params = new_params
