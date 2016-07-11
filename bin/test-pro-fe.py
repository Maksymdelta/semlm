#!/usr/bin/env python3

import argparse
import colorama

from asr_tools.kaldi import read_nbest_file
from asr_tools.nbest_util import evaluate_nbests


from semlm.feature_extractor import UnigramFE
from semlm.sklearn import examples_to_matrix
from semlm.util import load_references
from semlm.pro import create_pro_examples

def parse_args():
    """Create an argparser and parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    parser.add_argument("ref_file", type=argparse.FileType('r'))
    return parser.parse_args()


def main():
    args = parse_args()
    colorama.init()
    # Read the n-best lists and references
    nbests = list(read_nbest_file(args.nbest_file))
    load_references(args.ref_file)
    evaluate_nbests(nbests)

    # Create train/test examples
    fe = UnigramFE()
    examples = create_pro_examples(nbests, fe)

    # Converts the Example objects to sk-learn objects (matrices)
    print('# of examples: {}'.format(len(examples)))
    data, vec = examples_to_matrix(examples)

    # Do a little printing:
    print('DATA:')
    print(data)
    print('Inverse transformed data (first 5):')
    # This is fairly slow
    print(vec.inverse_transform(data)[:5])


if __name__ == "__main__":
    main()
