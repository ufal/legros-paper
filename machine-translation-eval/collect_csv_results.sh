#!/bin/bash

METRIC=$1
#if [ -z "$METRIC" ] || [ "$METRIC" != "chrf" ] || [ "$METRIC" != "bleu" ]; then
#    echo "Usage: $0 <metric>"
#    exit 1
#fi

for RES_FILE in `find iwslt* -name "test.${METRIC}"`; do
    echo -n "$RES_FILE,"
    cat $RES_FILE
    echo
done | sed "s#^.*models/##;s#/test.${METRIC}##"
