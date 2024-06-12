#!/bin/bash

if [ "$1" == "mean" ]; then
    VALUE=mean
elif [ "$1" == "std" ]; then
    VALUE=std
else
    echo "Usage: $0 [mean|std]"
    exit 1
fi

for TYPE in words morf bpe bpe_ours sp sp_ours morf_bpe morf_bpe_ours morf_sp morf_sp_ours; do
    for LNG in cs en es fr hu it ru; do
            grep Accuracy logs/pos_${LNG}_${TYPE}.*.out | \
                sed -e "s/.*Accuracy: //;s/%.*//;" | \
                if [ $VALUE == "mean" ]; then awk '{ total += $1; count++ } END { print total/count }'; else awk '{ total += $1; square += $1*$1; count++ } END { print sqrt(square/count - (total/count)**2) }'; fi | tr '\n' ','
    done
    echo
done | sed -e 's/,$//'
