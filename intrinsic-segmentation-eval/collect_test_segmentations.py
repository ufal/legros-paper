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


def load_segmentations_from_file(handle, segmentation_dict):
    for line in handle:
        tokens = line.strip().split("\t")
        word = tokens[0].lower()
        segmentation = tokens[3].lower()
        if not segmentation:
            continue
        segments = segmentation.lower().split(" + ")
        assert "".join(segments) == word
        index = 0
        for seg in segments:
            index += len(seg)
            segmentation_dict[word].add(index)


def segmentation_from_indices(word, indices):
    prev_start = 0
    segments = []
    for i, char in enumerate(word):
        if i in indices:
            if i in indices:
                segments.append(word[prev_start:i])
                prev_start = i
    segments.append(word[prev_start:])
    return segments


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("vocabulary", type=argparse.FileType("r"))
    parser.add_argument("segmentations", type=argparse.FileType("r"), nargs="+")
    parser.add_argument("--base-count", type=int, default=1000)
    parser.add_argument("--min-freq", type=int, default=10)
    args = parser.parse_args()

    logging.info("Load vocabulary: %s.", args.vocabulary)
    vocab = []
    for line in args.vocabulary:
        word, count_str = line.strip().split("\t")
        if len(word) < 2:
            continue
        count = int(count_str)
        if count < args.min_freq:
            break
        vocab.append(word)
    args.vocabulary.close()
    logging.info("Vocabulary size: %d.", len(vocab))

    segmentations = defaultdict(set)
    for seg_file in args.segmentations:
        logging.info("Load segmentations: %s", seg_file)
        load_segmentations_from_file(seg_file, segmentations)
    logging.info("Total %d segmentations.", len(segmentations))

    test_set = []
    vocab_tenth = len(vocab) // 10
    for decile, start in enumerate(range(0, len(vocab), vocab_tenth)):
        if decile > 9:
            break
        decile_count = args.base_count // (decile + 1)
        logging.info("Sample decile %d -> %d words.", decile + 1, decile_count)
        vocab_part = set(vocab[start:start + vocab_tenth])

        decile_segmentations = [
            (k, segmentation_from_indices(k, v))
            for k, v in segmentations.items()
            if k in vocab_part]

        if len(decile_segmentations) <= decile_count:
            test_set.extend(decile_segmentations)
        else:
            test_set.extend(
                random.sample(decile_segmentations, decile_count))

    for word, seg in test_set:
        print(f"{word}\t{' '.join(seg)}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
