#!/bin/bash

set -ex

for LNG in cs de en es fi fr hu it mn ru; do
    for SIZE in 1 2 4 8 16 32 64 96 128 160 192; do
        echo $LNG - $SIZE
        sbatch --export="ALL,DATA=${LNG}/plaintext/${LNG}.lc.txt,SIZE=${SIZE},OUTPUT=${LNG}/spm" submit_spm_train.sh
        sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_bpe_train.sh
    done
    sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_morfessor.sh
done

for LNG in hu; do
    for SIZE in 1 2 4 8 16 32 64 96 128 160 192; do
        sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_bpe_train.sh
    done
done
