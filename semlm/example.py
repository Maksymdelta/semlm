

# Now I need to convert these to feature vectors and class lists...

class Example():
    class_ = None
    features = None

    def __init__(self, class_, features):
        self.class_ = class_
        self.features = features

    # need the unicode version of this too...
    def __str__(self):
        str_ = ['<', 'class:', str(self.class_), 'features:' + str(self.features), '>']
        return ' '.join(str_)