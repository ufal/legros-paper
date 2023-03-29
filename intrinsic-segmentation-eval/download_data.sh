#!/bin/bash

set -ex
LNG=$1

mkdir -p ${LNG}/plaintext
cd ${LNG}/plaintext

# Mongolian is not NewsCrawl, let's use CC-100
if [ ${LNG} = "mn" ]; then
    wget https://data.statmt.org/cc-100/mn.txt.xz
    xzcat mn.txt.xz | sacremoses -l ${LNG} tokenize -x | sed 's/[[:upper:]]*/\L&/g' > ${LNG}.lc.txt
    rm mn.txt.xz
else
    SIZE=0

    # Download years in decreasing order, until we reach 50M sentences
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
fi

# Get vocabulary from the tokenized corpus
python3 ../../get_vocabulary.py ${LNG}.lc.txt --min-count 5 --num-threads 8 > ${LNG}.lc.txt.vocab
head -n 200000 ${LNG}.lc.txt.vocab > ${LNG}.lc.txt.vocab.200k

cd -

# Train fasttext
python3 ../scripts/train_fasttext.py ${LNG}/plaintext/${LNG}.lc.txt ${LNG}/fasttext
