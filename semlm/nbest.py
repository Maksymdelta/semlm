import sys
from semlm.sentence import Sentence
from semlm.evaluation_util import get_global_reference
from semlm.evaluation_util import print_diff


class NBest:

    sentences = None
    id_ = None

    def __init__(self, id_, sentences):
        assert(sentences is not None)
        assert(len(sentences) > 0)
        self.id_ = id_
        self.sentences = sentences

    def __str__(self):
        # This might be relatively slow because of all the string concatenation
        print_str = ''
        best_rank = self.best_rank()
        print_str += 'ID: {} (#{} is best)\n'.format(self.id_, best_rank)
        for i, s in enumerate(self.sentences):
            print_str += '{:3d} '.format(i+1) + str(s)
            if best_rank == i:
                print_str += ' **'
            print_str += '\n'
        return print_str

    def best_rank(self):
        best_wer = float('inf')
        best_rank = None
        for i, s in enumerate(self.sentences):
            if s.eval_.wer() < best_wer:
                best_wer = s.eval_.wer()
                best_rank = i
        return best_rank

    def improveable(self):
        return (best_rank == 0)

    def print_ref_hyp_best(self):
        best_rank = self.best_rank()
        ref = get_global_reference(self.id_)
        hyp = self.sentences[0]
        best = self.sentences[best_rank]
        print_str = ''
        print_str += 'ID: {} (#{} is best)\n'.format(self.id_, best_rank)
        if get_global_reference(self.id_):
            print_str += '{:3} '.format('') + str(ref) + '\n'
        else:
            print_str += '    No reference found.\n'
        print_str += '{:3d} '.format(1) + str(hyp) + '\n'
        print_str += '{:3d} '.format(best_rank + 1) + str(best) + '\n'
        print(print_str)
        print_diff(best, hyp)
