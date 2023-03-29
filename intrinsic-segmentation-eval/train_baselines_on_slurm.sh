#!/bin/bash

set -ex

for LNG in cs de en es fi fr hu it mn ru; do
    for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80 96 128 160 192; do
        echo $LNG - $SIZE
        sbatch --export="ALL,DATA=${LNG}/plaintext/${LNG}.lc.txt,SIZE=${SIZE},OUTPUT=${LNG}/spm" submit_spm_train.sh
        sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_bpe_train.sh
    done
    sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_morfessor.sh
done

