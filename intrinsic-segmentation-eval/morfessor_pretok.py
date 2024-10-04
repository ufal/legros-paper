#!/usr/bin/env python3
"""Pretokenize text using Morfessor."""

import argparse
import morfessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Morfessor model file')
    parser.add_argument('text', help='Text file to pretokenize', type=argparse.FileType('r'))
    args = parser.parse_args()

    model = morfessor.MorfessorIO().read_binary_model_file(args.model)

    for line in args.text:
        line = line.strip()
        if not line:
            print()
            continue
        tokens = []
        for word in line.split():
            tokens.extend(model.viterbi_segment(word)[0])
        print(' '.join(tokens))


if __name__ == '__main__':
    main()
