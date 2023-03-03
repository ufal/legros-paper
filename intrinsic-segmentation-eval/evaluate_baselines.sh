#!/bin/bash

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

for LNG in cs de en es fi fr hu it mn ru; do
    mkdir -p ${LNG}/baselines
    TEST_FILE=${LNG}/test_set.tsv

    for SIZE in 1 2 4 8 16 32 64 96 128 160 192; do
        cut -f1 ${TEST_FILE} | \
        ${SPM_HOME}/build/src/spm_encode --model ${LNG}/spm/spm.${SIZE}.model | \
        sed 's/▁//g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/test.spm${SIZE}.tsv

        cut -f1 ${TEST_FILE} | \
        subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
        sed 's/@@ / /g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/test.bpe${SIZE}.tsv
    done

    cut -f1 ${TEST_FILE} | morfessor-segment - -l ${LNG}/morfessor/model.bin | paste <(cut -f1 ${TEST_FILE}) - >  ${LNG}/baselines/test.morfessor.tsv
done


for LNG in cs de en es fi fr hu it mn ru; do
    TEST_FILE=${LNG}/test_set.tsv
    for TYPE in bpe spm; do
        LOG_FILE=${LNG}/baselines/${TYPE}.baseline
        echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}
        for SIZE in 1 2 4 8 16 32 64 96 128 160 192; do
            echo -n ${SIZE}000, >> ${LOG_FILE}
            python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/test.${TYPE}${SIZE}.tsv >> ${LOG_FILE}
        done
    done

    (
    LOG_FILE=${LNG}/baselines/morfessor.baseline
    echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}

    cut -f1 ${LNG}/plaintext/${LNG}.lc.txt.vocab | morfessor-segment - -l ${LNG}/morfessor/model.bin | tr ' ' '\n' | sort -u | wc -l | tr '\n' ',' >> ${LOG_FILE}
    python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/test.morfessor.tsv >> ${LOG_FILE}
    ) &
done
