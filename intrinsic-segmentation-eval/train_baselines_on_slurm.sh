#!/bin/bash

set -ex

#for LNG in cs de en es fi fr hu it mn ru; do
#    for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80 96 128 160 192; do
#        echo $LNG - $SIZE
#        sbatch --export="ALL,DATA=${LNG}/plaintext/${LNG}.lc.txt,SIZE=${SIZE},OUTPUT=${LNG}/spm" submit_spm_train.sh
#        sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_bpe_train.sh
#    done
#    sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE}" submit_morfessor.sh
#done
#
#while squeue -u $USER | grep -q "train-morfessor:"; do
#    echo "Waiting for morfessor to finish"
#    sleep 60
#done

for LNG in cs en es fr hu it mn ru; do
    for SIZE in 1 2 4 8 16 24 32 40 48 56 64 72 80 96 128; do
        echo $LNG - $SIZE
        sbatch --export="ALL,LNG=${LNG},SIZE=${SIZE},MORFESSOR=1" submit_spm_train.sh
        #sbatch --export "ALL,LNG=${LNG},SIZE=${SIZE},MORFESSOR=1" submit_bpe_train.sh
    done
done
