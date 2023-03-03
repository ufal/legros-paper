#!/usr/bin/env python3

"""Get test sets from the Universal Segmentations dataset.

We want to have common words, but the most frequent one, so we N words which
are in the middle with respect to the proportion of corpus that they represent.
In other words, we take N words, such that words that more frequent and less
frequent are approximately the proportion of the corpus.
"""

import argparse
from collections import defaultdict
import logging
import random
random.seed(1348)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("vocabulary", type=argparse.FileType("r"))
    parser.add_argument("segmentation", type=argparse.FileType("r"))
    parser.add_argument("--skip-first", type=int, default=10000)
    parser.add_argument("--min-freq", type=int, default=10)
    parser.add_argument(
        "--count", type=int, default=10000,
        help="Size of the final test set.")
    args = parser.parse_args()

    allowed_words = set()

    for i, line in enumerate(args.vocabulary):
        if i < args.skip_first:
            continue
        word, count_str = line.strip().split("\t")
        if len(word) < 2:
            continue
        count = int(count_str)
        if count < args.min_freq:
            break
        allowed_words.add(word)
    args.vocabulary.close()

    used_words = set()
    segmentation_pool = []
    for line in args.segmentation:
        tokens = line.strip().split("\t")
        word = tokens[0].lower()
        segmentation = tokens[3].lower()
        if not segmentation:
            continue
        if word in allowed_words and word not in used_words:
            used_words.add(word)
            format_seg = segmentation.replace(" + ", " ")
            segmentation_pool.append(f"{word.lower()}\t{format_seg.lower()}")

    if len(segmentation_pool) < args.count:
        segmentations = segmentation_pool
    else:
        segmentations = random.sample(segmentation_pool, args.count)
    for item in segmentations:
        print(item)

    logging.info("Done.")


if __name__ == "__main__":
    main()
