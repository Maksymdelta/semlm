from sklearn.metrics import classification_report

def print_feature_weights(model, vec):
    print('Feature weights:')
    try:
        # For perceptron, SGD, logistic regression, etc.:
        for name, val in zip(vec.get_feature_names(), model.coef_[0]):
            print ('{:20} {:0.2f}'.format(name, val))
    except:
        # For linear regression, ridge, etc.
        for name, val in zip(vec.get_feature_names(), model.coef_):
            print ('{:20} {:0.2f}'.format(name, val))


def evaluate_model(model, data):
    feature_array, classifications = data
    predictions = model.predict(feature_array)
    print('First 10 predictions: {}'.format(predictions[:10]))
    print('First 10 classifications: {}'.format(classifications[:10]))
    print('Accuracy: {}'.format(model.score(feature_array, classifications)))
    print('Classification report:')
    print(classification_report(classifications, predictions, digits=4))
