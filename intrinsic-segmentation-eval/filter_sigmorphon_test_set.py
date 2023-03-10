#!/usr/bin/env python3

"""Get morphs from the SIGMORPHON 2022 test set.

For most languages there morphemes instead of morph in the SIMORPHON test sets.
This scripts tries to map the morphemes back to the surface form.
"""

import argparse
import logging
import sys


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def longest_common_substring(a, b):
    """Find longest common substring between two strings A and B."""
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(a), 0, -1):
        for j in range(len(a) - i + 1):
            if a[j:j + i] in b:
                return a[j:j + i]
    return ''


def try_with_lcs(word, segments):
    new_segments = []
    rest_word = word
    for seg in segments:
        if not rest_word.startswith(seg):
            seg = longest_common_substring(rest_word, seg)

        if rest_word.startswith(seg):
            new_segments.append(seg)
            rest_word = rest_word[len(seg):]
        else:
            break

        non_alpha_idx = None
        for i, c in enumerate(rest_word):
            if not c.isalpha():
                non_alpha_idx = i
            else:
                break

        if non_alpha_idx is not None:
            new_segments.append(rest_word[:i])
            rest_word = rest_word[i:]
    return new_segments


def format(word, segments):
    return f"{word}\t{' '.join(segments)}"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input", type=argparse.FileType("r"), nargs="?",
        default=sys.stdin)
    args = parser.parse_args()
    logging.info("Read segmentations from %s.", args.input)

    total_number = 0
    skipped = 0
    for line in args.input:
        total_number += 1
        tokens = line.strip().split("\t")
        word = tokens[0].lower()
        segments = tokens[1].lower().split(" @@")

        new_segments = try_with_lcs(word, segments)

        if "".join(new_segments) == word:
            print(format(word, new_segments))
            continue

        reverse_word = word[::-1]
        reverse_segments = [seg[::-1] for seg in segments[::-1]]
        reversed_new_segments = try_with_lcs(
            reverse_word, reverse_segments)
        reversed_new_segments = [seg[::-1] for seg in reversed_new_segments[::-1]]

        if "".join(reversed_new_segments) == word:
            print(format(word, reversed_new_segments))
            continue
        if "".join(new_segments + reversed_new_segments) == word:
            print(format(word, new_segments + reversed_new_segments))
            continue

        left_len = sum([len(s) for s in new_segments])
        right_len = sum([len(s) for s in reversed_new_segments])
        segment_candidate = word[left_len:-right_len]

        if len(segment_candidate) < 3:
            print(format(
                word, new_segments + [segment_candidate] + reversed_new_segments))
            continue

        skipped =+ 1

    logging.info(
        "Skipped %d of %d => %.1f", skipped, total_number,
        100 * skipped / total_number)
    logging.info("Done.")


if __name__ == "__main__":
    main()
