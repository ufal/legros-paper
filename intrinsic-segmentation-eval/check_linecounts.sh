

echo Experiments that failed to train
echo ---------------------------------
for EXP in {cs,en,es,fr,hu,it,mn,ru}/experiments/*/; do
    if [ ! -e ${EXP}/subwords.19 ]; then
        echo ${EXP}
    fi
done
echo


echo Experiments with incorrect line counts or failed to evaluate
echo -------------------------------------------------------------
for EXP in ??/experiments/*/; do
    if [ -e ${EXP}/subwords.19 ]; then
        VOCAB_COUNT=$(wc -l < ${EXP}/subwords.19)
        EMBEDDING_COUNT=$(wc -l < ${EXP}/subword_embeddings.19)
        if [ ${VOCAB_COUNT} != ${EMBEDDING_COUNT} ]; then
            echo "${EXP}      (vocab ${VOCAB_COUNT}, embeddings ${EMBEDDING_COUNT})"
            continue
        fi

        if [ -e ${EXP}/test.embedding-based.score ]; then
            TEST_COUNT=$(tr ',' '\n' < ${EXP}/test.embedding-based.score 2> /dev/null | wc -l 2> /dev/null)
        else
            TEST_COUNT=0
        fi
        if [ -e ${EXP}/sigmorphon.embedding-based.score ]; then
            SIGMORPHON_COUNT=$(tr ',' '\n' < ${EXP}/sigmorphon.embedding-based.score  2> /dev/null | wc -l 2> /dev/null) 2> /dev/null
        else
            SIGMORPHON_COUNT=0
        fi

        if [ ${EXP} == 'de/*' ] || [ ${EXP} == 'fi/*' ]; then
            SIGMORPHON_COUNT=4
        fi

        if [ ${TEST_COUNT} != 4 ] || [ ${SIGMORPHON_COUNT} != 4 ]; then
            if [ ${TEST_COUNT} != 4 ]; then TEST_STATUS=FAILED; else TEST_STATUS=OK; fi
            if [ ${SIGMORPHON_COUNT} != 4 ]; then SIGMORPHON_STATUS=FAILED; else SIGMORPHON_STATUS=OK; fi
            echo "${EXP}      (test ${TEST_STATUS} sigmorphon ${SIGMORPHON_STATUS})"
        fi


    fi
done
