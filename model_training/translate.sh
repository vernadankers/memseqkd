#!/bin/bash -v

###################################################################
# Translate with a trained model.

set -eu

DATADIR=$1
MODELDIR=$2
INPUT=$3
OUTPUT=$4
BEAM=$5
MODEL=$6
vocab=joint.model.spm

$MARIAN_BINARIES/marian-decoder \
    -m $MODELDIR/$MODEL \
    -v $DATADIR/$vocab $DATADIR/$vocab \
    -d 0 1 3 4 5 6 7 \
    --mini-batch 64 --maxi-batch 512 --maxi-batch-sort src \
    --max-length 1000 --max-length-crop \
    -w -4096 \
    -i $INPUT -o $OUTPUT -b $BEAM \
    --log-level TEXT=debug
