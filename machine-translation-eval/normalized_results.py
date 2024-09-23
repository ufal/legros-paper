#!/usr/bin/env python3

import argparse
from collections import defaultdict

import numpy as np


VOCAB_SIZES = ["4k", "8k", "16k"]
PRETOK_TYPES = ["words", "morf"]
VOCAB_TYPES = ["bpe", "unigram"]
INFERENCE = ["original", "embedding"]


HEADER = r"""\begin{tabular}{lll cccc}
\toprule
\multicolumn{3}{l}{\multirow{2}{*}{Tokenization}} & \multicolumn{3}{c}{Vocabulary} & \mulrow{2}{*}{Avg.} \\ \cmidrule{4-6}
& & & 4k & 8k & 16k & \\ \midrule
"""
FOOTER = r"\end{tabular}"


ROW_STARTS = r"""\multirow{4}{*}{\rotatebox[origin=c]{90}{Word-like}} & \multirow{2}{*}{\rotatebox[origin=c]{90}{BPE}} & Original & 
& & Emb. + Bigr. & 
& \multirow{2}{*}{\rotatebox[origin=c]{90}{Unigram}} & Original & 
& & Emb. + Bigr. & 
\multirow{4}{*}{\rotatebox[origin=c]{90}{Morfessor}} & \multirow{2}{*}{\rotatebox[origin=c]{90}{BPE}} & Original & 
& & Emb. + Bigr. & 
& \multirow{2}{*}{\rotatebox[origin=c]{90}{Unigram}} & Original & 
& & Emb. + Bigr. & 
""".split("\n")

ROW_ENDS = r"""\\
\\ \cmidrule{4-7}
\\
\\ \midrule
\\
\\ \cmidrule{4-7}
\\
\\ \bottomrule
""".split("\n")


def main():
    parser = argparse.ArgumentParser(description='Normalize results')
    parser.add_argument('results', type=str, help='Results file')
    args = parser.parse_args()

    lng_pair_results = defaultdict(dict)

    with open(args.results) as f:
        for line in f:
            epx_id, value_str = line.strip().split(",")
            value = float(value_str)

            src_lng, tgt_lng, pretok, vocab_type, vocab_size, segm_type = epx_id.split("-")

            lng_pair = f"{src_lng}-{tgt_lng}"
            experiment = f"{pretok}-{vocab_type}-{vocab_size}-{segm_type}"

            lng_pair_results[lng_pair][experiment] = value

    norm_lng_pair_results = defaultdict(lambda: dict)
    for lng_pair, results in lng_pair_results.items():
        table = np.zeros((len(PRETOK_TYPES) * len(VOCAB_TYPES) * len(INFERENCE), len(VOCAB_SIZES)))
        mean = np.mean(list(results.values()))
        std = np.std(list(results.values()))
        for i, pretok in enumerate(PRETOK_TYPES):
            for j, vocab_type in enumerate(VOCAB_TYPES):
                for k, inference in enumerate(INFERENCE):
                    for l, vocab_size in enumerate(VOCAB_SIZES):
                        experiment = f"{pretok}-{vocab_type}-{vocab_size}-{inference}"
                        if experiment in results:
                            res = results[experiment]
                        else:
                            res = mean
                        table[i * len(VOCAB_TYPES) * len(INFERENCE) + j * len(INFERENCE) + k, l] = res
        table = (table - mean) #/ std
        norm_lng_pair_results[lng_pair] = table

    mean_overall = np.stack([table for table in norm_lng_pair_results.values()]).mean(axis=0)

    print(HEADER)
    line = 0
    for pretok in PRETOK_TYPES:
        for vocab_type in VOCAB_TYPES:
            for inference in INFERENCE:
                values = [f"{val:.2f}" for val in mean_overall[line]]
                mean = f"{mean_overall[line].mean():.2f}"
                print(ROW_STARTS[line] + " & ".join(values) + f" & {mean} " + ROW_ENDS[line])
                line += 1
    print(FOOTER)

if __name__ == '__main__':
    main()
