#!/usr/bin/env python3

import argparse
from collections import defaultdict
from functools import partial
import logging
import math
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Converts a list of bigrams to a Python dictionary')
    parser.add_argument('bigrams', type=str, help='A file containing a list of bigrams in plain text format')
    parser.add_argument('output', type=str, help='The output file to write the Python dictionary to')
    args = parser.parse_args()

    logging.info("Read the bigrams.")
    subwords = set()
    counts = defaultdict(lambda: defaultdict(int))
    with open(args.bigrams, 'r') as f:
        for line in f:
            tok1, tok2, count = line.split()
            count = int(count)
            if count == 0:
                print(line)
                exit()
            if tok1 == "<s>":
                tok1 = "###"
            counts[tok1][tok2] = count
            subwords.add(tok1)
            subwords.add(tok2)

    logging.info("Normalize counts.")
    norm_counts = {}
    for seg1, seg2_counts in counts.items():
        denominator = sum(seg2_counts.values()) + len(subwords)
        assert denominator > 0
        norm_counts[seg1] = defaultdict(partial(float, -math.log(denominator)))
        for seg2, count in seg2_counts.items():
            assert count > 0
            norm_counts[seg1][seg2] = math.log(count) - math.log(denominator)

    logging.info("Save the counts into a file.")
    with open(args.output, "wb") as f_save:
        pickle.dump(norm_counts, f_save)
    logging.info("Done.")


if __name__ == '__main__':
    main()
