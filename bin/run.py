#!/usr/bin/env python3

import argparse
from semlm.kaldi import read_nbest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    # nbest = read_nbest_file(args.file)
    nbest = read_nbests(args.file)


if __name__ == "__main__":
    main()
