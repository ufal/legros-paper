#!/bin/bash

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

# 1. SEGMENT THE WORDS USING TRAINED BASELINE MODELS

for LNG in cs en es fr hu it mn ru; do
    mkdir -p ${LNG}/baselines
    for SET in test sigmorphon; do
        TEST_FILE=${LNG}/${SET}_set.tsv

        if [ ! -e ${TEST_FILE} ]; then
            continue
        fi

        #cut -f1 ${TEST_FILE} | morfessor-segment - -l ${LNG}/morfessor/model.bin | paste <(cut -f1 ${TEST_FILE}) - >  ${LNG}/baselines/${SET}.morfessor.tsv

        for SIZE in 1 2 4 8 16 24 32 40 48 56 72 64 80 96 128; do
            #MODEL=${LNG}/spm/morf-spm.${SIZE}.model
            #if [ -e ${MODEL} ]; then
            #    cut -f1 ${TEST_FILE} | \
            #    ${SPM_HOME}/build/src/spm_encode --model ${LNG}/spm/spm.${SIZE}.model | \
            #    sed 's/▁//g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.spm${SIZE}.tsv
            #fi

            MODEL=${LNG}/spm/morf-spm.${SIZE}.model
            if [ -e ${MODEL} ]; then
                cut -f2 ${LNG}/baselines/${SET}.morfessor.tsv | \
                ${SPM_HOME}/build/src/spm_encode --model ${MODEL} | \
                sed 's/▁//g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.morf-spm${SIZE}.tsv
            else
                echo "SKIPPING ${MODEL}"
            fi

            #cut -f1 ${TEST_FILE} | \
            #subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
            #sed 's/@@ / /g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.bpe${SIZE}.tsv

            #cut -f2 ${LNG}/baselines/${SET}.morfessor.tsv | \
            #subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.morf-bpe${SIZE} | \
            #sed 's/@@ / /g' | paste <(cut -f1 ${TEST_FILE}) - > ${LNG}/baselines/${SET}.morf-bpe${SIZE}.tsv
        done

    done
done

# 2. EVALUATE SP and BPE (because it is fast)

for LNG in cs en es fr hu it mn ru; do
    #for SET in test sigmorphon; do
    for SET in sigmorphon; do
        TEST_FILE=${LNG}/${SET}_set.tsv

        if [ ! -e ${TEST_FILE} ]; then continue; fi
        #for TYPE in bpe spm morf-bpe morf-spm; do
        for TYPE in morf-spm ; do
            LOG_FILE=${LNG}/baselines/${SET}.${TYPE}.score
            RECALL_FILE=${LNG}/baselines/${SET}.${TYPE}.recall
            FSCORE_FILE=${LNG}/baselines/${SET}.${TYPE}.fscore
            echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}
            echo vocab_size,boundary_recall,confidence_low,confidence_high > ${RECALL_FILE}
            echo vocab_size,boundary_fscore,confidence_low,confidence_high > ${FSCORE_FILE}
            for SIZE in 1 2 4 8 16 24 32 40 48 56 64 72 80 96; do
                GUESS_FILE=${LNG}/baselines/${SET}.${TYPE}${SIZE}.tsv
                if [ ! -e ${GUESS_FILE} ]; then continue; fi
                echo -n ${SIZE}000, >> ${LOG_FILE}
                echo -n ${SIZE}000, >> ${RECALL_FILE}
                python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${GUESS_FILE} >> ${LOG_FILE}
                python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${GUESS_FILE} --use-recall >> ${RECALL_FILE}
            done
            python3 f_score_from_prec_and_recall.py ${LOG_FILE} ${RECALL_FILE} --header > ${FSCORE_FILE}
        done
    done
done
exit

# 3. EVALUATE MORFESSOR (becase it is slow)

#for LNG in cs de en es fi fr hu it mn ru; do
#for LNG in en es fi fr hu it mn ru; do
for LNG in cs de; do
    for SET in test sigmorphon; do
        VOCAB=$(cut -f1 ${LNG}/plaintext/${LNG}.lc.txt.vocab | morfessor-segment - -l ${LNG}/morfessor/model.bin | tr ' ' '\n' | sort -u | wc -l | tr '\n' ',')
        for SET in test sigmorphon; do
            TEST_FILE=${LNG}/${SET}_set.tsv
            LOG_FILE=${LNG}/baselines/${SET}.morfessor.score
            RECALL_FILE=${LNG}/baselines/${SET}.morfessor.recall
            FSCORE_FILE=${LNG}/baselines/${SET}.morfessor.fscore
            echo vocab_size,boundary_prec,confidence_low,confidence_high > ${LOG_FILE}
            echo vocab_size,boundary_recall,confidence_low,confidence_high > ${RECALL_FILE}
            echo -n ${VOCAB} >> ${LOG_FILE}
            echo -n ${VOCAB} >> ${RECALL_FILE}
            if [ ! -e ${TEST_FILE} ]; then continue; fi
            python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/${SET}.morfessor.tsv >> ${LOG_FILE}
            python3 eval_boundary_precision.py --gold ${TEST_FILE} --guess ${LNG}/baselines/${SET}.morfessor.tsv --use-recall >> ${RECALL_FILE}
            python3 f_score_from_prec_and_recall.py ${LNG}/baselines/${SET}.morfessor.score ${LNG}/baselines/${SET}.morfessor.recall --header > ${FSCORE_FILE}
        done
    done
done
