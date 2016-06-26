

# WIP


class FeatureExtractor():
    pass

class UnigramFE(FeatureExtractor):

    def extract(self, s):
        return s.words.copy()
