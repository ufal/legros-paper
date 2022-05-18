#!/usr/bin/bash

set -ex

CORPUS=$1
VOCAB=$2

TMP=$(mktemp experiments/tmp.XXXXX)

# shuffle corpus and split it
shuf $CORPUS | head -n 1000000 | split -l 1000 --numeric-suffixes=1 -a 5 - ${TMP}.split.

MAX_FILES=$(ls ${TMP}.split.* | wc -l)

qsub -t 1-${MAX_FILES} -N sample_bigrams -e ${TMP}.bigrams.0.log -o ${TMP}.bigrams.0 -sync y qsub_array_sample_bigrams.sh $VOCAB uniform ${TMP}.split

python -m neuralpiece.train_estimator $VOCAB ${TMP}.bigrams.0 ${TMP}.model.0

for I in {1..10}; do
    rm ${TMP}.split.*
    shuf $CORPUS | head -n 200000 | split -l 200 --numeric-suffixes=1 -a 5 - ${TMP}.split.
    MAX_FILES=$(ls ${TMP}.split.* | wc -l)

    qsub -t 1-${MAX_FILES} -N sample_bigrams -e ${TMP}.bigrams.${I}.log -o ${TMP}.bigrams.${I} -sync y qsub_array_sample_bigrams.sh $VOCAB ${TMP}.model.$((I - 1)) ${TMP}.split

    python -m neuralpiece.train_estimator $VOCAB ${TMP}.bigrams.${I} ${TMP}.model.${I} --load-estimator ${TMP}.model.$((I - 1))
done

rm ${TMP}.split.*
