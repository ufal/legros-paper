#!/bin/bash

#set -ex

SPM_HOME=/lnet/troja/projects/neuralpiece/evaluation/sentencepiece
OURS_HOME=/lnet/troja/projects/neuralpiece/subword-segmentation/build

#for LNG in cs en es fr hu it mn ru; do
#    mkdir -p ${LNG}/renyi
#
#    for SIZE in 8 16 24 32 40 48 56 64; do
#        FASTTEXT=${LNG}/fasttext
#
#        # WORD PRE-TOKENIZATION
#
#        # Vanilla BPE and SPM
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.bpe${SIZE} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            ${SPM_HOME}/build/src/spm_encode --model=${LNG}/spm/spm.${SIZE}.model --output_format=piece | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_sp${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/spm${SIZE}k.txt
#
#        # Words Ours from embeddings
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
#                ${FASTTEXT} ${LNG}/experiments/from_bpe${SIZE}k/{subwords.19,subword_embeddings.19} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_emb_from_bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/emb_from_bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
#                ${FASTTEXT} ${LNG}/experiments/from_sp${SIZE}k/{subwords.19,subword_embeddings.19} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_emb_from_sp${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/emb_from_sp${SIZE}k.txt
#
#        # Words Ours Distilled
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            python3 -m neuralpiece.segment_vocab_with_bigram_stats --greedy ${LNG}/experiments/from_bpe${SIZE}k/bigram_stats | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_ours_from_bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/ours_from_bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.txt | \
#            python3 -m neuralpiece.segment_vocab_with_bigram_stats --greedy ${LNG}/experiments/from_sp${SIZE}k/bigram_stats | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_ours_from_sp${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/ours_from_sp${SIZE}k.txt
#
#        # MORFESSOR PRE-TOKENIZATION
#        # Vanilla BPE and SPM
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            subword-nmt apply-bpe -c ${LNG}/bpe/${LNG}.morf-bpe${SIZE} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/morf-bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            ${SPM_HOME}/build/src/spm_encode --model=${LNG}/spm/morf-spm.${SIZE}.model --output_format=piece | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_sp${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/morf-spm${SIZE}k.txt
#
#        # Morfessor Ours from embeddings
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
#                ${FASTTEXT} ${LNG}/experiments/from_morf-bpe${SIZE}k/{subwords.19,subword_embeddings.19} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_emb_from_morf-bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/emb_from_morf-bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            python3 -m neuralpiece.segment_vocab_with_subword_embeddings \
#                ${FASTTEXT} ${LNG}/experiments/from_morf-sp${SIZE}k/{subwords.19,subword_embeddings.19} | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_emb_from_morf-spm${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/emb_from_morf-sp${SIZE}k.txt
#
#        # Morfessor Ours from embeddings
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            python3 -m neuralpiece.segment_vocab_with_bigram_stats --greedy ${LNG}/experiments/from_morf-bpe${SIZE}k/bigram_stats | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_ours_from_morf-bpe${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/ours_from_morf-bpe${SIZE}k.txt
#
#        head -n 4000 ${LNG}/plaintext/${LNG}.lc.morfessor.txt | \
#            python3 -m neuralpiece.segment_vocab_with_bigram_stats --greedy ${LNG}/experiments/from_morf-sp${SIZE}k/bigram_stats | \
#            sed 's/  */ /g' | \
#            tee ${LNG}/renyi/output_ours_from_morf-sp${SIZE}k.txt | \
#            tokenization-scorer > ${LNG}/renyi/ours_from_morf-sp${SIZE}k.txt
#    done
#done

for LNG in cs en es fr hu it mn ru; do
    echo "${LNG},8k,16k,24k,32k,48k"
    for TYPE in bpe emb_from_bpe ours_from_bpe spm emb_from_sp ours_from_sp morf-bpe emb_from_morf-bpe ours_from_morf-bpe morf-spm emb_from_morf-sp ours_from_morf-sp; do
        echo -n "${TYPE},"
        for SIZE in 8 16 24 32 40 48; do
            FILE=${LNG}/renyi/${TYPE}${SIZE}k.txt
            if [ ! -f ${FILE} ]; then
                echo -n ","
            else
                cat ${FILE} | awk '{printf "%.4f,",$1}'
            fi
        done | sed 's/,$//'
        echo
    done
    echo
done
