import argparse

# Don't need this any more

from semlm.kaldi import read_nbest_file
from scipy.sparse import lil_matrix
from scipy import int8


# Let's just use the DictVectorizer() for now!

def main():
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("nbest_file", type=argparse.FileType('r'))
    # parser.add_argument("ref_file", type=argparse.FileType('r'))
    args = parser.parse_args()
    print(args)
    # colorama.init()
    # Read the n-best lists
    nbests = list(read_nbest_file(args.nbest_file))

    print(len(nbests))

    # matrix = csr_matrix((3,4), dtype=int8)
    matrix = lil_matrix((3, 4), dtype=int8)
    # print(matrix.toarray())
    # print(matrix[0][0])
    matrix[0, 0] = 1
    print(matrix[0, 0])
    print(matrix.toarray())
    print(matrix)

if __name__ == "__main__":
    main()
