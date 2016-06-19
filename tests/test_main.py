import unittest

from semlm.kaldi import read_nbest_file
from semlm.kaldi import read_transcript_table



class Testing(unittest.TestCase):

    nbest_file = "test/data/lat.1.nbest.txt"
    ref_file = "test/data/dev_clean.ref"
    
    def test_testing(self):
        self.assertEqual('foo'.upper(), 'FOO')


    def test_read_nbest(self):
        with open(self.nbest_file) as f:
            nbests = list(read_nbest_file(f))
            self.assertTrue(len(nbests) == 40)

    def test_read_transcript(self):
        with open(self.ref_file) as f:
            refs = read_transcript_table(f)
            self.assertTrue(len(refs) == 2703)

            
        # python ./bin/sklearn-test.py  ~/data/librispeech1/nbests/lat.1.nbest.txt ~/data/librispeech1/dev_clean.ref
        #  6717  python ./bin/test-fe.py  ~/data/librispeech1/nbests/lat.1.nbest.txt
