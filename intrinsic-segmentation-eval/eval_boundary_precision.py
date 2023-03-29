#!/usr/bin/env python3

"""Evaluation of the precision on morphological segmentation boundaries.

Small parts of the code are copied from the SIGMORPHON 2022 Morpheme
Segmentation Shared Task Evaluation Script.
"""

import argparse

import numpy as np
from scipy.stats import bootstrap


SEP = " "


def read_tsv(path):
    # tsv without header, format: word TAB segments
    entries = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Line {i} of {path} does not have 2 columns")

            word, segments = fields

            seg_list = segments.split(SEP)
            if "".join(seg_list) != word:
                raise ValueError(f"Segments '{segments}' do not construct the word '{word}'")

            entries.append(segments)
    return entries


def get_boundary_indices(segments):
    boundaries = []
    j = 0

    for i in range(len(segments)):
        if segments[i] == SEP:
            boundaries.append(j)
        else:
            # only increment i when not on boundary
            j += 1

    len_after_sep_rem = len("".join(segments.split(SEP)))
    assert j == len_after_sep_rem, f"{segments}, {j}, {len_after_sep_rem}, {str(boundaries)}"

    return boundaries


def main():
    parser = argparse.ArgumentParser(description='Measure precision on morphological segmentation boundaries')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    args = parser.parse_args()

    gold = read_tsv(args.gold)
    guess = read_tsv(args.guess)

    if len(gold) != len(guess):
        raise ValueError("gold and guess tsvs do not have the same number of entries")

    # we want to compute precision on the breaks, not on the morphemes, since we'd like to
    # use this for subword segmentation for downstream tasks, so whole words are OK.

    precisions = []

    for i, (ref, hyp) in enumerate(zip(gold, guess)):
        ref_breaks = get_boundary_indices(ref)
        hyp_breaks = get_boundary_indices(hyp)

        boundaries_made = 1 + len(hyp_breaks)
        correct_boundaries_made = 1
        for br in ref_breaks:
            if br in hyp_breaks:
                correct_boundaries_made += 1

        precision = correct_boundaries_made / boundaries_made * 100
        precisions.append(precision)

    stat = bootstrap((precisions,), np.mean)
    mean_prec = np.mean(precisions)

    print(f"{mean_prec:.3f},{stat.confidence_interval.low:.3f},{stat.confidence_interval.high:.3f}")


if __name__ == "__main__":
    main()
