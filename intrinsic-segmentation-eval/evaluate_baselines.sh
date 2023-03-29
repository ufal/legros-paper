#!/bin/bash

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

# 1. SEGMEENT THE WORDS USING TRAINED BASELINE MODELS

#for LNG in cs de en es fi fr hu it mn ru; do
#    mkdir -p ${LNG}/baselines
#    for SET in test sigmorphon; do
#        TEST_FILE=${LNG}/${SET}_set.tsv
#
#        if [ ! -e ${TEST_FILE} ]; then
#            continue
#        fi
#
#        #for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80 96 128 160 192; do
#        for SIZE in 24 40 48 56 72 80; do
#            cut -f1 ${TEST_FILE} | \
#            ${SPM_HOME}/build/src/spm_encode --model ${LNG}/spm/spm.${SIZE}.model | \
#            sed 's/▁//g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.spm${SIZE}.tsv
#
#            cut -f1 ${TEST_FILE} | \
#            subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
#            sed 's/@@ / /g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.bpe${SIZE}.tsv
#        done
#
#       # cut -f1 ${TEST_FILE} | morfessor-segment - -l ${LNG}/morfessor/model.bin | paste <(cut -f1 ${TEST_FILE}) - >  ${LNG}/baselines/${SET}.morfessor.tsv
#    done
#done

# 2. EVALUATE SP and BPE (because it is fast)

for LNG in cs de en es fi fr hu it mn ru; do
    for SET in test sigmorphon; do
        TEST_FILE=${LNG}/${SET}_set.tsv

        if [ ! -e ${TEST_FILE} ]; then continue; fi
        for TYPE in bpe spm; do
            LOG_FILE=${LNG}/baselines/${SET}.${TYPE}.baseline
            echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}
            for SIZE in 1 2 4 8 16 24 32 40 48 56 64 72 80 96 128; do
                echo -n ${SIZE}000, >> ${LOG_FILE}
                python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/${SET}.${TYPE}${SIZE}.tsv >> ${LOG_FILE}
            done
        done
    done
done

# 3. EVALUATE MORFESSOR (becase it is slow)

#for LNG in cs de en es fi fr hu it mn ru; do
#for LNG in en es fi fr hu it mn ru; do
#    for SET in test sigmorphon; do
#        TEST_FILE=${LNG}/${SET}_set.tsv
#        LOG_FILE=${LNG}/baselines/test.morfessor.baseline
#        SIGMORPHON_LOG_FILE=${LNG}/baselines/sigmorphon.morfessor.baseline
#        echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}
#
#        cut -f1 ${LNG}/plaintext/${LNG}.lc.txt.vocab | morfessor-segment - -l ${LNG}/morfessor/model.bin | tr ' ' '\n' | sort -u | wc -l | tr '\n' ',' >> ${LOG_FILE}
#        cp ${LOG_FILE} ${SIGMORPHON_LOG_FILE}
#        for SET in test sigmorphon; do
#            TEST_FILE=${LNG}/${SET}_set.tsv
#            if [ ! -e ${TEST_FILE} ]; then continue; fi
#            python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/${SET}.morfessor.tsv >> ${LNG}/baselines/${SET}.morfessor.baseline
#        done
#    done
#done
