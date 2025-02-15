#!/bin/bash

langpair=$1
srclang=$2
trglang=$3
bucket=$4

python analyze_buckets.py --model_zoo_path ../model_zoo/${srclang}-${trglang} \
    --wmt_path ../wmt20/${langpair} --srclang $srclang --trglang $trglang --bucket $bucket --device 0