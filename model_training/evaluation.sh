#!/bin/bash

export MARIAN_BINARIES=/mnt/internship/utils/marian_binaries_updated
root="/mnt/internship/memseqkd"
langpair=$1
srclang=$2
trglang=$3
test=$4

for folder in teacher_seed=1111
do
    echo "Translate ${folder}"
    bash translate.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/${folder}" \
        "${root}/wmt20/${langpair}/${test}.${srclang}" \
        "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" 5 "model.iter300000.npz"
done

for folder in student_b=1_seed=1111 student_b=5_seed=1111 student_b=10_seed=1111
do
    echo "Translate ${folder}"
    bash translate.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/${folder}" \
        "${root}/wmt20/${langpair}/${test}.${srclang}" \
        "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" 5 "model.iter100000.npz"
done


for folder in teacher_seed=1111 student_b=1_seed=1111 student_b=5_seed=1111 student_b=10_seed=1111
do
    echo "Score ${folder}"
    #sacrebleu "${root}/wmt20/${langpair}/${test}.${trglang}" -i "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" -m bleu -b -w 2
    #sacrebleu "${root}/wmt20/${langpair}/${test}.${trglang}" -i "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" -m chrf -b -w 2
    #sacrebleu "${root}/wmt20/${langpair}/${test}.${trglang}" -i "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" -m ter -b -w 2
    comet-score -s "${root}/wmt20/${langpair}/${test}.${srclang}" -t "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" -r "${root}/wmt20/${langpair}/${test}.${trglang}" --model /mnt/internship/comet-22/wmt20-comet-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/${langpair}/${test}.${srclang}" -t "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" -r "${root}/wmt20/${langpair}/${test}.${trglang}" --model /mnt/internship/comet-22/wmt22-comet-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/${langpair}/${test}.${srclang}" -t "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" --model /mnt/internship/comet-22/wmt20-comet-qe-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/${langpair}/${test}.${srclang}" -t "${root}/model_zoo/${srclang}-${trglang}/${folder}/${test}.${trglang}" --model /mnt/internship/comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt --quiet
done

#python compare.py "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=1111/kd.beam1.${trglang}" "${root}/model_zoo/${srclang}-${trglang}/student_b=1_seed=1111/kd.beam1.${trglang}"

