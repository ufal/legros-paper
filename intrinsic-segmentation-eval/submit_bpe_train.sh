#!/usr/bin/bash

#SBATCH -J train-bpe
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH -o log/bpe-train-%A.out

set -ex

if [ -z ${LNG} ]; then
    echo LNG must be define. > /dev/stderr
    exit 1
fi

if [ -z ${SIZE} ]; then
    echo SIZE must be define. > /dev/stderr
    exit 1
fi


if [ -z ${MORFESSOR} ]; then
    INPUT=${LNG}/plaintext/${LNG}.lc.txt
    OUTPUT=${LNG}/bpe/${LNG}.bpe${SIZE}
else
    INPUT=${LNG}/plaintext/${LNG}.lc.morfessor.txt
    OUTPUT=${LNG}/bpe/${LNG}.morf-bpe${SIZE}
fi


subword-nmt learn-bpe -i ${INPUT} -o ${OUTPUT} -s ${SIZE}000
