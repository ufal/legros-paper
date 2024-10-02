#!/bin/bash

set -ex

SPM_HOME=../3rd_party/sentencepiece

for LNG in cs en es fr hu it mn ru; do
    VOCAB=${LNG}/fasttext.vocab

    MORFESSOR_DIR=${LNG}/experiments/from_morfessor
    #mkdir -p ${MORFESSOR_DIR}
    #cat ${VOCAB} | \
    #    morfessor-segment - -l ${LNG}/morfessor/model.bin | paste ${VOCAB} - > ${MORFESSOR_DIR}/init.allowed

    for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80 96 128; do
        #EXP_DIR=${LNG}/experiments/from_sp${SIZE}k
        #mkdir -p ${EXP_DIR}
        #cat ${VOCAB} | \
        #${SPM_HOME}/build/src/spm_encode --model ${LNG}/spm/spm.${SIZE}.model | \
        #sed 's/▁//g' | paste -d' ' ${VOCAB} - > ${EXP_DIR}/init.allowed

        EXP_DIR=${LNG}/experiments/from_morf-sp${SIZE}k
        MODEL=${LNG}/spm/morf-spm.${SIZE}.model
        if [ -e ${MODEL} ]; then
            mkdir -p ${EXP_DIR}
            cut -f2 ${MORFESSOR_DIR}/init.allowed | \
            ${SPM_HOME}/build/src/spm_encode --model ${MODEL} | \
            sed 's/▁//g' | paste -d' ' ${VOCAB} - > ${EXP_DIR}/init.allowed
        fi

        #EXP_DIR=${LNG}/experiments/from_bpe${SIZE}k
        #mkdir -p ${EXP_DIR}
        #cat ${VOCAB} | \
        #subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
        #sed 's/@@ / /g' | paste ${VOCAB} - > ${EXP_DIR}/init.allowed

        #EXP_DIR=${LNG}/experiments/from_morf-bpe${SIZE}k
        #mkdir -p ${EXP_DIR}
        #cut -f2 ${MORFESSOR_DIR}/init.allowed | \
        #subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.morf-bpe${SIZE} | \
        #sed 's/@@ / /g' | paste ${VOCAB} - > ${EXP_DIR}/init.allowed
    done
done
