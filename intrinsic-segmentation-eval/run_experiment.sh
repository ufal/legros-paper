#!/bin/bash

#SBATCH -J embedding-tokenizer
#SBATCH --cpus-per-task=60
#SBATCH --mem=200G

set -ex

CPP_HOME=/lnet/troja/projects/neuralpiece/cpp-implementation

while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--language)
            LNG=$2
            shift
            shift
            ;;
        -s|--size)
            SP_SIZE=$2
            shift
            shift
            ;;
        -i|--init)
            INIT_TYPE=$2
            shift
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [ ${INIT_TYPE} != bpe ] && [ ${INIT_TYPE} != sp ]; then
    echo "Initialization type (-i --init) must be 'sp' or 'bpe'." > /dev/stderr
fi

DATA=../../plaintext/${LNG}.lc.txt
FASTTEXT=../../fasttext
OUT=${LNG}/experiments/from_${INIT_TYPE}${SP_SIZE}k
INIT_ALLOWED=init.allowed


if [ ! -d ${OUT} ]; then
    echo ${OUT} must be an existing directory. > /dev/stderr
    exit 1
fi

if [ ! -f ${OUT}/${INIT_ALLOWED} ]; then
    echo \"${INIT_ALLOWED}\" should be a file with initial segmentation. > /dev/stderr
    exit 1
fi

# ##################################
# TRAIN USING CPP PROGRAM
# ##################################

cd ${OUT}
${CPP_HOME}/build/train_subword_embeddings \
    ${FASTTEXT}.txt ${DATA} \
    --fastext-output-pseudoinverse ${FASTTEXT}.out_inv.txt \
    --allowed-substrings ${INIT_ALLOWED} \
    --epochs 20
cd -

# ##################################
# EXTRACT BIGRAM STATS AND EVALUTE
# ##################################

source ../env/bin/activate
python3 ../scripts/distill_count_based_bigram_model.py \
    ${OUT}/${FASTTEXT} ${OUT}/subwords.19 \
    ${OUT}/subword_embeddings.19 \
    ${LNG}/plaintext/${LNG}.lc.txt.vocab.200k \
    ${OUT}/bigram_stats

