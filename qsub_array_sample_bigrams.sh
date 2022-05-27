#!/bin/bash

#$ -V -cwd
#$ -l hostname=!(iridium|hydra*)
#$ -pe smp 2

VOCAB=$1
MODEL=$2
PREFIX=$3

ID=$(printf "%05d" $SGE_TASK_ID)
FILE=${PREFIX}.${ID}

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m neuralpiece.sample_bigrams $VOCAB $MODEL $FILE
