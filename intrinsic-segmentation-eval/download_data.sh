#!/bin/bash

set -ex
LNG=$1

mkdir -p ${LNG}/plaintext
cd ${LNG}/plaintext

SIZE=0

for YEAR in {2021..2007}; do
    wget https://data.statmt.org/news-crawl/${LNG}/news.${YEAR}.${LNG}.shuffled.deduped.gz
    YEAR_SIZE=$(zcat news.${YEAR}.${LNG}.shuffled.deduped.gz | wc -l)
    SIZE=$(( SIZE + YEAR_SIZE  ))
    if [ ${SIZE} -gt 50000000 ]; then
        break
    fi
done


zcat *.gz | shuf | head -n 50000000 | sacremoses -l ${LNG} tokenize -x | sed 's/[[:upper:]]*/\L&/g' > ${LNG}.lc.txt
rm *.gz

python3 ../../get_vocabulary.py ${LNG}.lc.txt --min-count 5 --num-threads 8 > ${LNG}.lc.txt.vocab

cd -

python3 ../scripts/train_fasttext.py ${LNG}/plaintext/${LNG}.lc.txt ${LNG}/fasttext
