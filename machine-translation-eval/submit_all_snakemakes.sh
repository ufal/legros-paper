#!/bin/bash

set -ex

for DIR in iwslt2017-{ar-en,de-en,en-fr,en-nl,en-ro,it-en,it-nl,ro-it,ro-nl} ; do
    cd ${DIR}
    snakemake --unlock
    cd ..
    sbatch --job-name=${DIR}-snakemake --output=${DIR}/snakemake-%j.log --wrap="cd ${DIR} && snakemake --profile ufal --rerun-incomplete"
done

