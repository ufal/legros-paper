#!/bin/bash

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

for LNG in cs de en es fi fr hu it mn ru; do
    VOCAB=${LNG}/fasttext.vocab
    for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80 96 128 160 192; do
        EXP_DIR=${LNG}/experiments/from_sp${SIZE}k
        mkdir -p ${EXP_DIR}
        cat ${VOCAB} | \
        ${SPM_HOME}/build/src/spm_encode --model ${LNG}/spm/spm.${SIZE}.model | \
        sed 's/▁//g' | paste -d' ' ${VOCAB} - > ${EXP_DIR}/init.allowed

        EXP_DIR=${LNG}/experiments/from_bpe${SIZE}k
        mkdir -p ${EXP_DIR}
        cat ${VOCAB} | \
        subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
        sed 's/@@ / /g' | paste ${VOCAB} - > ${EXP_DIR}/init.allowed
    done
done
