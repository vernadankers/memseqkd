#!/bin/bash
MARIAN_BINARIES="<path>"

langpair=$1
srclang=$2
trglang=$3
model_num=$4
steps=$5
seed=1111

wmt_dir="../wmt20/${langpair}"

$MARIAN_BINARIES/marian-scorer -m ${srclang}-${trglang}/model_${model_num}/model.iter${steps}.npz \
    -t $wmt_dir/train.$srclang "${wmt_dir}/train.$trglang" \
    --vocabs $wmt_dir/joint.model.spm \
    $wmt_dir/joint.model.spm \
    --log-level debug -o ${srclang}-${trglang}/model_${model_num}/train.scores --devices 0 1 2 3 4 5 6 7 \
    --workspace -4096 -n --mini-batch 64 --maxi-batch 1000 --word-scores \
    --max-length 1000 --max-length-crop