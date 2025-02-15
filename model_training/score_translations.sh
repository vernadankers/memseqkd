#!/bin/bash

langpair=$1
srclang=$2
trglang=$3

model_dir="/mnt/internship/memseqkd/model_zoo/${srclang}-${trglang}/teacher_seed=1111"
wmt_dir="/mnt/internship/memseqkd/wmt20/${langpair}"

$MARIAN_BINARIES/marian-scorer \
    -m $model_dir/model.iter100000.npz \
    -t ${wmt_dir}/train.${srclang} ${model_dir}/kd.beam1.${trglang} \
    --vocabs ${wmt_dir}/joint.model.spm \
             ${wmt_dir}/joint.model.spm \
    --log-level debug -o ${model_dir}/train.scores --devices 4 5 \
    --workspace -4096 -n --mini-batch 32 --maxi-batch 32 --word-scores \
    --max-length 1000 --max-length-crop
