#!/usr/bin/bash

#SBATCH -J train-smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -o log/smp-train-%A.out

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

if [ -z ${MORFESSOR} ]; then
    INPUT=${LNG}/plaintext/${LNG}.lc.txt
    OUTPUT=${LNG}/spm/spm.${SIZE}
    SENTECESIZE=5000000
else
    INPUT=${LNG}/plaintext/${LNG}.lc.morfessor.txt
    OUTPUT=${LNG}/spm/morf-spm.${SIZE}
    SENTECESIZE=1000000000
fi

if [ -e ${OUTPUT}.model ]; then
    exit
fi

${SPM_HOME}/build/src/spm_train \
    --input ${INPUT} \
    --model_prefix ${OUTPUT} \
    --vocab_size ${SIZE}000 \
    --num_threads 16 \
    --input_sentence_size=${SENTECESIZE} \
    --train_extremely_large_corpus=true \
    --shuffle_input_sentence=true
