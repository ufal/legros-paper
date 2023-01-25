#!/bin/bash

#SBATCH -J collect-bigrams
#SBATCH --cpus-per-task=2
#SBATCH --mem=240G

set -ex

#python3 scripts/distill_count_based_bigram_model.py \
#    data/czeng.cs.lc.fasttext ${DIR}/subwords.last \
#    ${DIR}/subword_embeddings.last.txt \
#    data/czeng.cs.lc.vocab.200k \
#    ${DIR}/bigram_stats

#cut -f1 evaluation/ces.word.dev.tsv | \
#    python -m neuralpiece.segment_vocab_with_bigram_stats ${DIR}/bigram_stats | \
#    sed -e 's/ / @@/' | \
#    paste <(cut -f1 evaluation/ces.word.dev.tsv) - \
#        > ${DIR}/ces.word.dev.bigram_counts.tsv

python3 scripts/distill_neural_bigram_model.py \
    data/czeng.cs.lc.fasttext ${DIR}/subwords.last \
    ${DIR}/subword_embeddings.last.txt \
    data/czeng.cs.lc.vocab.200k \
    ${DIR}/bigram_estimator.numpy

cut -f1 evaluation/ces.word.dev.tsv | \
    python -m neuralpiece.segment_vocab_with_bigram_stats ${DIR}/bigram_estimator.numpy --model-type neural | \
    sed -e 's/ / @@/' | \
    paste <(cut -f1 evaluation/ces.word.dev.tsv) - \
        > ${DIR}/ces.word.dev.bigram_neural.tsv
