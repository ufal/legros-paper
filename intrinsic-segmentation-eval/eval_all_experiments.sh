#!/bin/bash

set -ex

for LNG in en es fi fr hu it mn ru cs; do
    FASTTEXT=${LNG}/fasttext
    for SIZE in 1 2 4 8 16 24 32 40 48 56 64 72 80 96 128 ; do
        #for TYPE in bpe sp; do
        for TYPE in morf-bpe morf-sp; do
            OUT=${LNG}/experiments/from_${TYPE}${SIZE}k

            if [ -e ${OUT}/subword_embeddings.19 ]; then
                #for SET in test sigmorphon; do
                for SET in sigmorphon; do
                    EVAL_DATA=${LNG}/${SET}_set.tsv
                    if [ ! -e ${EVAL_DATA} ]; then continue; fi

                    # 1. EVALUATE EMBEDDING-BASED MODEL
                    SCORE_FILE=${OUT}/${SET}.embedding-based.score
                    RECALL_FILE=${OUT}/${SET}.embedding-based.recall
                    FSCORE_FILE=${OUT}/${SET}.embedding-based.fscore
                    OUTPUT_FILE=${OUT}/${SET}_output_embedding-based.tsv

                    if [ -e ${SCORE_FILE} ] && [ ${SCORE_FILE} -nt ${OUT}/subword_embeddings.19 ]; then
                        continue
                    fi

                    wc -l < ${OUT}/subwords.19 | tr '\n' ',' > ${SCORE_FILE}
                    cp ${SCORE_FILE} ${RECALL_FILE}
                    cut -f1 ${EVAL_DATA} | \
                        python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
                            ${FASTTEXT} ${OUT}/{subwords.19,subword_embeddings.19} | \
                        paste <(cut -f1 ${EVAL_DATA}) - > ${OUTPUT_FILE}
                    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} >> ${SCORE_FILE}
                    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} --use-recall >> ${RECALL_FILE}
                    python3 f_score_from_prec_and_recall.py ${SCORE_FILE} ${RECALL_FILE} > ${FSCORE_FILE}

                    #2. EVALUATE BIGRAM-BASED MODEL

                    if [ ! -e ${OUT}/bigram_stats ]; then continue; fi

                    SCORE_FILE=${OUT}/bigrams.score
                    RECALL_FILE=${OUT}/bigrams.recall
                    FSCORE_FILE=${OUT}/bigrams.fscore
                    OUTPUT_FILE=${OUT}/test_output_bigrams.tsv
                    wc -l < ${OUT}/subwords.19 | tr '\n' ',' > ${SCORE_FILE}
                    cp ${SCORE_FILE} ${RECALL_FILE}
                    cut -f1 ${EVAL_DATA} | \
                        python3 -m neuralpiece.segment_vocab_with_bigram_stats --greedy ${OUT}/bigram_stats | \
                        paste <(cut -f1 ${EVAL_DATA}) - > ${OUTPUT_FILE}
                    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} >> ${SCORE_FILE}
                    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} --use-recall >> ${RECALL_FILE}
                    python3 f_score_from_prec_and_recall.py ${SCORE_FILE} ${RECALL_FILE} > ${FSCORE_FILE}
                done
            fi
        done

        # Test morfessor-based bigram models without embeddings
        #for TYPE in morf-bpe morf-sp; do
        #    EVAL_DATA=${LNG}/sigmorphon_set.tsv
        #    OUT=${LNG}/experiments/from_${TYPE}${SIZE}k
        #    if [ ! -e ${OUT}/init.allowed ]; then continue; fi

        #    # Compute bigram stats if necessary
        #    if [ ! -e ${OUT}/init_bigram_stats ] || [ ${OUT}/init_bigram_stats -ot ${OUT}/init.allowed ]; then
        #        python3 ../subword-segmentation/scripts/distill_count_based_bigram_model_from_allowed_init.py \
        #            ${OUT}/init.allowed \
        #            ${LNG}/plaintext/${LNG}.lc.txt.vocab.200k \
        #            ${OUT}/init_bigram_stats
        #    fi

        #    SCORE_FILE=${OUT}/init_bigrams.score
        #    RECALL_FILE=${OUT}/init_bigrams.recall
        #    FSCORE_FILE=${OUT}/init_bigrams.fscore
        #    OUTPUT_FILE=${OUT}/test_output_init_bigrams.tsv
        #    wc -l < ${OUT}/subwords.19 | tr '\n' ',' > ${SCORE_FILE}
        #    cp ${SCORE_FILE} ${RECALL_FILE}
        #    cut -f1 ${EVAL_DATA} | \
        #        python3 -m neuralpiece.segment_vocab_with_bigram_stats ${OUT}/init_bigram_stats | \
        #        paste <(cut -f1 ${EVAL_DATA}) - > ${OUTPUT_FILE}
        #    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} >> ${SCORE_FILE}
        #    python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} --use-recall >> ${RECALL_FILE}
        #    python3 f_score_from_prec_and_recall.py ${SCORE_FILE} ${RECALL_FILE} > ${FSCORE_FILE}


        #done
    done
    continue

    OUT=${LNG}/experiments/from_morfessor
    for SET in sigmorphon; do
        EVAL_DATA=${LNG}/${SET}_set.tsv
        if [ ! -e ${EVAL_DATA} ]; then continue; fi

        # 1. EVALUATE EMBEDDING-BASED MODEL
        SCORE_FILE=${OUT}/${SET}.embedding-based.score
        RECALL_FILE=${OUT}/${SET}.embedding-based.recall
        FSCORE_FILE=${OUT}/${SET}.embedding-based.fscore
        OUTPUT_FILE=${OUT}/${SET}_output_embedding-based.tsv

        #if [ -e ${SCORE_FILE} ] && [ ${SCORE_FILE} -nt ${OUT}/subword_embeddings.19 ]; then
        #    continue
        #fi

        wc -l < ${OUT}/subwords.19 | tr '\n' ',' > ${SCORE_FILE}
        cp ${SCORE_FILE} ${RECALL_FILE}
        cut -f1 ${EVAL_DATA} | \
            python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
                ${FASTTEXT} ${OUT}/{subwords.19,subword_embeddings.19} | \
            paste <(cut -f1 ${EVAL_DATA}) - > ${OUTPUT_FILE}
        python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} >> ${SCORE_FILE}
        python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} --use-recall >> ${RECALL_FILE}
        python3 f_score_from_prec_and_recall.py ${SCORE_FILE} ${RECALL_FILE} > ${FSCORE_FILE}

        #2. EVALUATE BIGRAM-BASED MODEL

        if [ ! -e ${OUT}/bigram_stats ]; then continue; fi

        SCORE_FILE=${OUT}/bigrams.score
        RECALL_FILE=${OUT}/bigrams.recall
        FSCORE_FILE=${OUT}/bigrams.fscore
        OUTPUT_FILE=${OUT}/test_output_bigrams.tsv
        wc -l < ${OUT}/subwords.19 | tr '\n' ',' > ${SCORE_FILE}
        cp ${SCORE_FILE} ${RECALL_FILE}
        cut -f1 ${EVAL_DATA} | \
            python3 -m neuralpiece.segment_vocab_with_bigram_stats ${OUT}/bigram_stats | \
            paste <(cut -f1 ${EVAL_DATA}) - > ${OUTPUT_FILE}
        python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} >> ${SCORE_FILE}
        python3 eval_boundary_precision.py --gold ${EVAL_DATA} --guess ${OUTPUT_FILE} --use-recall >> ${RECALL_FILE}
        python3 f_score_from_prec_and_recall.py ${SCORE_FILE} ${RECALL_FILE} > ${FSCORE_FILE}
    done
done
