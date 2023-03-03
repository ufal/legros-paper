#!/usr/bin/bash

#SBATCH -J train-morfessor
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -o logs/morfessor-train-%A.out

if [ -z ${LNG} ]; then
    echo LNG must be defined. > /dev/stderr
    exit 1
fi

set -ex

mkdir -p ${LNG}/morfessor
morfessor-train --encoding=UTF-8 --traindata-list --logfile=${LNG}/morfessor/training.log --save ${LNG}/morfessor/model.bin -d ones ${LNG}/plaintext/${LNG}.lc.txt.vocab

cut -f1 ${LNG}/test_set.tsv | morfessor-segment - -l ${LNG}/morfessor/model.bin | paste <(cut -f1 ${LNG}/test_set.tsv ) - >  ${LNG}/morfessor/test_set_segmented.tsv
morfessor-segment ${LNG}/fasttext.vocab -l ${LNG}/morfessor/model.bin | paste -d' ' ${LNG}/fasttext.vocab - > ${LNG}/morfessor/experiment.init

