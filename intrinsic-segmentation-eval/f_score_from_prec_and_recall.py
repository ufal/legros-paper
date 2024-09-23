#!/usr/bin/env python3

import argparse
import sys


def f_score(precision, recall):
    precision, recall = float(precision), float(recall)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('precision', type=argparse.FileType('r'), help='Precision')
    parser.add_argument('recall', type=argparse.FileType('r'), help='Recall')
    parser.add_argument('--header', action='store_true', help='Print header')
    args = parser.parse_args()

    if args.header:
        print("vocab_size,boundary_f1,boundary_f1_low,boundary_f1_high")
        args.precision.readline()
        args.recall.readline()

    for prec_line, recall_line in zip(args.precision, args.recall):
        vocab_s, mean_prec, low_prec, high_prec = prec_line.strip().split(',')
        vocab_s2, mean_recall, low_recall, high_recall = recall_line.strip().split(',')

        assert vocab_s == vocab_s2

        print(
            f"{vocab_s},{f_score(mean_prec, mean_recall)},"
            f"{f_score(low_prec, low_recall)},"
            f"{f_score(high_prec, high_recall)}")


if __name__ == '__main__':
    main()
