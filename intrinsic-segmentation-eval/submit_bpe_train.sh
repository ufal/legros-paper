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

subword-nmt learn-bpe -i ${LNG}/plaintext/${LNG}.lc.txt -o ${LNG}/bpe/${LNG}.bpe${SIZE} -s ${SIZE}000

