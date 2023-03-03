#!/usr/bin/bash

#SBATCH -J train-smp
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -o logs/smp-train-%A.out

set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece

${SPM_HOME}/build/src/spm_train \
    --input ${DATA} \
    --model_prefix ${OUTPUT}/spm.${SIZE} \
    --vocab_size ${SIZE}000 \
    --num_threads 4 \
    --input_sentence_size=5000000 \
    --shuffle_input_sentence=true
