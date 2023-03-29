#!/bin/bash

for LNG in cs de en es fi fr hu it mn ru; do
    for SIZE in 1 2 4 8 16 24 32 48 56 64 72 80; do
        for TYPE in bpe sp; do
            EXP_DIR=${LNG}/experiments2/from_${TYPE}${SIZE}k

            if [ ! -e ${EXP_DIR}/subword_embeddings.19 ]; then
                echo ${EXP_DIR}
                mkdir -p ${EXP_DIR}
                cp ${LNG}/experiments/from_${TYPE}${SIZE}k/init.allowed ${EXP_DIR}
                sbatch <<EOT
#!/bin/bash

#SBATCH -J embedding-tokenizer-${LNG}-${SIZE}-${TYPE}
#SBATCH --cpus-per-task=60
#SBATCH --mem=200G
#SBATCH -o log/ours-train-${LNG}-${TYPE}-${SIZE}-%A.out

bash run_experiment.sh -l ${LNG} -s ${SIZE} -i ${TYPE}
EOT
            fi
        done
    done
done
