#!/bin/bash -v

###################################################################
# Translate with a trained model.

set -eu

MODELDIR=$1
WMTDIR=$4
vocab=${WMTDIR}/joint.model.spm
model=$5

$MARIAN_BINARIES/marian-decoder \
  -m $MODELDIR/$model \
  -v $vocab $vocab \
  -d $(seq 0 7) \
  --mini-batch 64 --maxi-batch 1000 --maxi-batch-sort src \
  --max-length 1000 --max-length-crop \
  -w -8096 \
  -b 1 -i $2 -o $3 --log-level TEXT=debug
