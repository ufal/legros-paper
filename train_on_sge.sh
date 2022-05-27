#!/usr/bin/bash

set -ex

CORPUS=$1
VALID=$2
SEED_VOCAB=$3
TARGET_VOCAB_SIZE=$4

TMP=$(mktemp experiments/XXXXX)

VOCAB_CYCLE_STEPS=5 # Number of times that we sample bigrams
VOCAB=${TMP}.vocab.0

cp ${SEED_VOCAB} ${VOCAB}

CYCLE=0
while true; do
    VOCAB_SIZE=$(wc -l < $VOCAB)

    # shuffle corpus and split it
    shuf $CORPUS | head -n 200000 | split -l 400 --numeric-suffixes=1 -a 5 - ${TMP}.split.

    MAX_FILES=$(ls ${TMP}.split.* | wc -l)

    qsub -t 1-${MAX_FILES} -N sample_bigrams -pe smp 1 -e ${TMP}.bigrams.${CYCLE}.0.log -o ${TMP}.bigrams.${CYCLE}.0 -sync y \
        qsub_array_sample_bigrams.sh $VOCAB uniform ${TMP}.split

    python -m neuralpiece.train_estimator $VOCAB ${TMP}.bigrams.${CYCLE}.0 ${TMP}.model.${CYCLE}.0 2>&1 | tee ${TMP}.model.${CYCLE}.0.log
    python -m neuralpiece.tokenize $VOCAB ${TMP}.model.${CYCLE}.0.numpy $VALID | tee ${TMP}.valid.${CYCLE}.0

    for I in `seq 1 ${VOCAB_CYCLE_STEPS}`; do
        rm ${TMP}.split.*
        shuf $CORPUS | head -n 50000 | split -l 80 --numeric-suffixes=1 -a 5 - ${TMP}.split.
        MAX_FILES=$(ls ${TMP}.split.* | wc -l)

        qsub -t 1-${MAX_FILES} -N sample_bigrams -e ${TMP}.bigrams.${CYCLE}.${I}.log -o ${TMP}.bigrams.${CYCLE}.${I} -sync y \
            qsub_array_sample_bigrams.sh $VOCAB ${TMP}.model.${CYCLE}.$((I - 1)).numpy ${TMP}.split

        python -m neuralpiece.train_estimator \
            $VOCAB ${TMP}.bigrams.${I} ${TMP}.model.${I} \
            --load-estimator ${TMP}.model.$((I - 1)) \
            --learning-rate 5e-6 2>&1 | tee ${TMP}.model.${I}.log
        python -m neuralpiece.tokenize $VOCAB ${TMP}.model.${I}.numpy $VALID | tee ${TMP}.valid.${I}
    done

    rm ${TMP}.split.*

    if [ $VOCAB_SIZE -eq $TARGET_VOCAB_SIZE ]; then
        break
    fi

    NEW_VOCAB_SIZE=$(echo "print(max(${TARGET_VOCAB_SIZE}, int(0.95 * ${VOCAB_SIZE})))")
    python -m neuralpiece.reduce_vocab ${VOCAB} ${TMP}.model.${VOCAB_CYCLE_STEPS} > ${TMP}.new_vocab

    CYCLE=$(( CYCLE + 1 ))
    VOCAB=${TMP}.vocab.${CYCLE}
    mv ${TMP}.new_vocab ${VOCAB}
done
