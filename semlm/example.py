"""
Represents an 'example' a very generic object with a class and features,
for level machine learning purposes.
"""

class Example(object):
    """Represents an 'example' a very generic object with a class and features,
    for level machine learning purposes."""

    def __init__(self, class_, features):
        """Must initialize with a class and features."""
        self.class_ = class_
        self.features = features

    def __unicode__(self):
        """Print the class and features readably."""
        str_ = ['<', 'class:', str(self.class_), 'features:' + str(self.features), '>']
        return ' '.join(str_)
