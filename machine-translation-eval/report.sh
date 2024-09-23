#!/bin/bash

function report () {
    DIR=$1
    LNG1=$2
    LNG2=$3

    echo "$DIR: $LNG1 -> $LNG2"
    echo "---------------------------------------------========"
    echo -e "segmentation\t\t4k\t8k\t16k\tavg"
    echo "---------------------------------------------========"

    for TYPE in words-bpe words-unigram morf-bpe morf-unigram; do
        for INFERENCE in original embedding; do
            echo -ne "${TYPE} ${INFERENCE}\t"
            for SIZE in 4 8 16; do
                RESULT_FILE=${DIR}/models/${LNG1}-${LNG2}-${TYPE}-${SIZE}k-${INFERENCE}/test.chrf
                #echo $RESULT_FILE
                if [ ! -f ${RESULT_FILE} ]; then
                    echo -e "$RESULT_FILE\tFAILED" >> report.log
                    echo -ne "\033[0;31m----\033[0m\t"
                    continue
                fi

                # If the resuls is smaller than 30, it is probably an error, so we print it in red
                # and log it to the report.log file
                if [ $(cat $RESULT_FILE) \< 30 ]; then
                    echo -e "${RESULT_FILE}\t$(cat $RESULT_FILE)" >> report.log
                    echo -ne "\033[0;31m"
                fi
                # Print the result, one decimal, add leading space if smaller than 10
                cat $RESULT_FILE | awk '{printf "%02.1f", $1}' 2> /dev/null
                echo -ne "\033[0m \t"
            done
            for F in ${DIR}/models/${LNG1}-${LNG2}-${TYPE}-*k-${INFERENCE}/test.chrf; do cat $F; echo; done 2> /dev/null | awk '{sum+=$1} END {printf "%.1f", sum/NR}' 2> /dev/null
            echo
        done
    done
    echo
}

rm report.log

#report iku-eng iku eng
#report iku-eng eng iku
report iwslt2017-ar-en ara eng
report iwslt2017-ar-en eng ara
report iwslt2017-de-en deu eng
report iwslt2017-de-en eng deu
report iwslt2017-en-fr eng fra
report iwslt2017-en-fr fra eng
report iwslt2017-en-nl eng nld
report iwslt2017-en-nl nld eng
report iwslt2017-en-ro eng ron
report iwslt2017-en-ro ron eng
report iwslt2017-it-en ita eng
report iwslt2017-it-en eng ita
report iwslt2017-it-nl ita nld
report iwslt2017-it-nl nld ita
report iwslt2017-ro-it ron ita
report iwslt2017-ro-it ita ron
report iwslt2017-ro-nl ron nld
report iwslt2017-ro-nl nld ron
