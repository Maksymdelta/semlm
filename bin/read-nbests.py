#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_nbests, read_nbest_files, read_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file")
    parser.add_argument("ac_file")
    parser.add_argument("lm_file")
    parser.add_argument("ref_file")
    args = parser.parse_args()
    refs = read_ref(args.ref_file)
    nbests = read_nbest_files(args.nbest_file, args.ac_file, args.lm_file)

    for nbest in nbests:
        nbest.print_()


if __name__ == "__main__":
    main()
