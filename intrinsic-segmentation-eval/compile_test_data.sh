#!/bin/bash

set -ex

if [ ! -e UniSegments-1.0-public ]; then
    curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4629{/UniSegments-1.0-public.tar.gz}
    tar zxvf UniSegments-1.0-public.tar.gz
    rm UniSegments-1.0-public.tar.gz
fi

prepare_test () {
    LNG=$1
    DATA=$2
    python3 collect_test_segmentations.py \
        ${LNG}/plaintext/${LNG}.lc.txt.vocab \
        UniSegments-1.0-public/data/${DATA}/UniSegments-1.0-${DATA}.useg \
           | grep -Pv '\t$' > ${LNG}/test_set.tsv
}


prepare_test cs ces-DeriNet
prepare_test de deu-DerivBaseDE
prepare_test en eng-MorphoLex
prepare_test es spa-MorphyNet
prepare_test fi fin-MorphyNet
prepare_test fr fra-Demonette
prepare_test hu hun-MorphyNet
#prepare_test it ita-DerIvaTario
prepare_test it ita-MorphyNet
prepare_test mn mon-MorphyNet --skip-first 5000 --min-freq 5
prepare_test ru rus-DerivBaseRU
