from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

# Mostly for learning how sklearn works.

def print_feature_weights(model, vec):
    """Given a model and a vectorized example file, print all the feature weights.
    Printing procedure is a little different between classification and regression."""
    print('Feature weights:')
    try:
        # For perceptron, SGD, logistic regression, etc...
        # because these are class-conditional?
        for name, val in sorted(zip(vec.get_feature_names(), model.coef_[0]), key=lambda x: x[1], reverse=True):
            print('{:20} {:0.2f}'.format(name, val))
    except:
        # For linear regression, ridge, etc.
        for name, val in sorted(zip(vec.get_feature_names(), model.coef_), key=lambda x: x[1], reverse=True):
            print('{:20} {:0.2f}'.format(name, val))


def evaluate_model(model, data):
    """Given a model and some data, show the first 10 classifications and references,
    the accuracy, and the classification report."""
    feature_array, classifications = data
    predictions = model.predict(feature_array)
    print('First 10 predictions: {}'.format(predictions[:10]))
    print('First 10 classifications: {}'.format(classifications[:10]))
    print('Accuracy: {}'.format(model.score(feature_array, classifications)))
    print('Classification report:')
    print(classification_report(classifications, predictions, digits=4))


def examples_to_matrix(examples, vec):
    print('# of examples: {}'.format(len(examples)))
    # The vectorizer wants dicts as inputs
    dicts = list(map(lambda x: x.features, examples))
    classes = list(map(lambda x: x.class_, examples))
    # Convert the data into the vocabulary
    data = vec.transform(dicts)
    return data, classes
