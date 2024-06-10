#!/bin/bash

ARGS="-p gpu-troja,gpu-ms --mem=16G --gres=gpu:1"

for SEED in {1..10}; do
    for LNG in cs en es fr hu it ru; do
        LNG_HOME="../intrinsic-segmentation-eval/${LNG}"

        sbatch $ARGS --job-name=pos_${LNG}_words --output=logs/pos_${LNG}_words.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_bpe --output=logs/pos_${LNG}_bpe.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bpe --subword-model ${LNG_HOME}/bpe/${LNG}.bpe32 --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_sp --output=logs/pos_${LNG}_sp.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords sp --subword-model ${LNG_HOME}/spm/spm.32.model --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_bpe_ours --output=logs/pos_${LNG}_bpe_ours.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bigram --subword-model ${LNG_HOME}/experiments/from_bpe32k/bigram_stats --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_sp_ours --output=logs/pos_${LNG}_sp_ours.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bigram --subword-model ${LNG_HOME}/experiments/from_sp32k/bigram_stats --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_morf_bpe --output=logs/pos_${LNG}_morf_bpe.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bpe --subword-model ${LNG_HOME}/bpe/${LNG}.morf-bpe32 --morfessor ${LNG_HOME}/morfessor/model.bin --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_morf_sp --output=logs/pos_${LNG}_morf_sp.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords sp --subword-model ${LNG_HOME}/spm/morf-spm.32.model --morfessor ${LNG_HOME}/morfessor/model.bin --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_morf_bpe_ours --output=logs/pos_${LNG}_morf_bpe_ours.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bigram --subword-model ${LNG_HOME}/experiments/from_morf-bpe32k/bigram_stats --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_morf_sp_ours --output=logs/pos_${LNG}_morf_sp_ours.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords bigram --subword-model ${LNG_HOME}/experiments/from_morf-sp32k/bigram_stats --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_morf --output=logs/pos_${LNG}_morf.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --morfessor ${LNG_HOME}/morfessor/model.bin --seed ${SEED}"

        sbatch $ARGS --job-name=pos_${LNG}_char --output=logs/pos_${LNG}_char.${SEED}.out \
            --wrap="python3 pos_tagger.py ${LNG} --subwords char --seed ${SEED}"
    done
done
