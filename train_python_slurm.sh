#!/bin/bash

set -ex

SEGMENT_SHARDS=16
EMBEDDING_TYPE=fasttext

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA=$2
            shift
            shift
            ;;
        -o|--out-dir)
            OUT=$2
            shift
            shift
            ;;
        -f|--fasttext)
            FASTTEXT=$2
            shift
            shift
            ;;
        --shards)
            SEGMENT_SHARDS=$2
            shift
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [ -z ${DATA} ]; then
    echo "Data (-d|--data) need to be specified." > /dev/stderr
    exit 1
fi

if [ -z ${OUT} ]; then
    echo "Output dir (-o|--out-dir) need to be specified." > /dev/stderr
    exit 1
fi

if [ -z ${FASTTEXT} ]; then
    echo "FastText model (-f|--fasttext) need to be specified." > /dev/stderr
    exit 1
fi

INIT_ALLOWED=${OUT}/allowed.init
INIT_SUBWORDS=${OUT}/vocab


if [ ! -d ${OUT} ]; then
    echo ${OUT} must be an existing directory. > /dev/stderr
    exit 1
fi

if [ ! -f ${INIT_SUBWORDS} ]; then
    echo \"${INIT_SUBWORDS}\" should be a file with initial subword vocabulary. > /dev/stderr
    exit 1
fi

# TODO if initial embeddings provided, it should compute the segmentation first

if [ ! -f ${INIT_ALLOWED} ]; then
    echo \"${INIT_ALLOWED}\" should be a file with initial segmentation. > /dev/stderr
    exit 1
fi

cp ${INIT_SUBWORDS} ${OUT}/subwords.00
#split -l 80000 -d -a 2 ${INIT_SUBWORDS} ${OUT}/subwords.00.
#SUBWORD_SHARDS=$(ls ${OUT}/subwords.00.?? | wc -l)
#
#sbatch \
#    --export "SUBWORDS=${OUT}/subwords.00,VOCAB=${DATA}.${EMBEDDING_TYPE}.vocab,OUT_EMB_INV_MATRIX=${DATA}.${EMBEDDING_TYPE}.out_inv.txt,DATA=${DATA},OUTPUT=${OUT}/subword_embeddings.00,ALLOWED=${INIT_ALLOWED}" \
#    --array 0-$(( SUBWORD_SHARDS - 1 )) \
#    --output ${OUT}/subwords.00.log \
#    --wait \
#    submit_subword_embeddings.sh
#
#cat ${OUT}/subword_embeddings.00.?? > ${OUT}/subword_embeddings.00.txt
#rm ${OUT}/subword_embeddings.00.??

python3 scripts/fasttext_for_subwords.py ${DATA}.${EMBEDDING_TYPE} ${INIT_SUBWORDS} ${OUT}/subword_embeddings.00.txt

PREV_EPOCH=00
for EPOCH in {01..30}; do
    sbatch \
        --export "ALL,FASTTEXT=${FASTTEXT},SUBWORD_VOCAB=${OUT}/subwords.${PREV_EPOCH},SUBWORD_EMBEDDINGS=${OUT}/subword_embeddings.${PREV_EPOCH}.txt,OUTPUT_PREFIX=${OUT}/segmentations.${EPOCH},WORD_VOCAB=${DATA}.${EMBEDDING_TYPE}.vocab,NUM_SHARDS=${SEGMENT_SHARDS}" \
        --array 1-${SEGMENT_SHARDS} \
        --output ${OUT}/log_segmentations.${EPOCH}.log \
        --wait \
        submit_vocab_segmentation.sh

    cat ${OUT}/segmentations.${EPOCH}.??? > ${OUT}/segmentations.${EPOCH}
    rm ${OUT}/segmentations.${EPOCH}.???

    tr ' ' '\n' < ${OUT}/segmentations.${EPOCH} | sort -u > ${OUT}/subwords.${EPOCH}
    paste -d' ' ${DATA}.${EMBEDDING_TYPE}.vocab ${OUT}/segmentations.${EPOCH} > ${OUT}/allowed_subwords.${EPOCH}

    if [ -e ${OUT}/allowed_subwords.${PREV_EPOCH} ]; then
        if `cmp --silent ${OUT}/allowed_subwords.${PREV_EPOCH} ${OUT}/allowed_subwords.${EPOCH}`; then
            break
        fi
    fi

    split -l 80000 -d -a 2 ${OUT}/subwords.${EPOCH} ${OUT}/subwords.${EPOCH}.
    SUBWORD_SHARDS=$(ls ${OUT}/subwords.${EPOCH}.?? | wc -l)

    sbatch \
        --export "SUBWORDS=${OUT}/subwords.${EPOCH},VOCAB=${DATA}.${EMBEDDING_TYPE}.vocab,OUT_EMB_INV_MATRIX=${DATA}.${EMBEDDING_TYPE}.out_inv.txt,DATA=${DATA},OUTPUT=${OUT}/subword_embeddings.${EPOCH},ALLOWED=${OUT}/allowed_subwords.${EPOCH}" \
        --array 0-$(( SUBWORD_SHARDS - 1 )) \
        --output ${OUT}/log_subword_embeddings.${EPOCH}.log \
        --wait \
        submit_subword_embeddings.sh

    cat ${OUT}/subword_embeddings.${EPOCH}.?? > ${OUT}/subword_embeddings.${EPOCH}.txt
    rm ${OUT}/subword_embeddings.${EPOCH}.?? ${OUT}/subwords.${EPOCH}.??

    PREV_EPOCH=$EPOCH
done

cd ${OUT}
ln -s $(realpath subwords.${PREV_EPOCH}) subwords.last
ln -s $(realpath subword_embeddings.${PREV_EPOCH}.txt) subword_embeddings.last.txt
cd -

python3 -m neuralpiece.score_subword_redundancy ${OUT}/{subwords.last,subword_embeddings.last.txt} > ${OUT}/subword_redundancy



# THIS WAS THE EVALUATION ON DEV DATA
# cut -f1 evaluation/ces.word.dev.tsv | python3 -m neuralpiece.segment_vocab_with_subword_embeddings data/czeng.cs.lc.${EMBEDDING_TYPE} oracle/subwords oracle/subword_embeddings.txt | sed -e 's/ / @@/g' | paste <(cut -f1 evaluation/ces.word.dev.tsv) - > evaluation/baselines/with_training_data_sum.csv
