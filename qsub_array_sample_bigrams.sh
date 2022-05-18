#!/bin/bash

#$ -V -cwd
#$ -l hostname=!(hydra*)

VOCAB=$1
MODEL=$2
PREFIX=$3

ID=$(printf "%05d" $SGE_TASK_ID)
FILE=${PREFIX}.${ID}

python -m neuralpiece.sample_bigrams $VOCAB $MODEL $FILE
