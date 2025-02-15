#!/bin/bash

export MARIAN_BINARIES=/mnt/internship/utils/marian_binaries_updated
root="/mnt/internship/memseqkd"
langpair=$1
srclang=$2
trglang=$3
beam=$4
seed=1111

# Construct the vocabulary, only run once per corpus, reuse afterwards
cat ${root}/wmt20/${langpair}/train.* > ${root}/wmt20/${langpair}/train.all
python train_spm.py "${root}/wmt20/${langpair}/train.all" 32000 "${root}/wmt20/${langpair}"
rm ${root}/wmt20/${langpair}/train.all

# Train the T-Large teacher model, 300k steps
bash train_teacher.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}" $seed $srclang $trglang "big"
bash translate.sh "${root}/wmt20" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}" "${root}/wmt20/train.en" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam1.de" 1
if [ "$srclang" == "en" ] && [ "$trglang" == "de" ]; then
    bash translate.sh "${root}/wmt20" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}" "${root}/wmt20/train.en" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam2.de" 2
    bash translate.sh "${root}/wmt20" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}" "${root}/wmt20/train.en" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam5.de" 5
    bash translate.sh "${root}/wmt20" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}" "${root}/wmt20/train.en" "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam10.de" 10
fi

# Train the T-Base student model, 100k steps
bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_b=${beam}_seed=${seed}" \
    "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam${beam}.${trglang}" $seed $srclang $trglang "base"
if [ "$srclang" == "en" ] && [ "$trglang" == "de" ]; then
    bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_b=${beam}_small_seed=${seed}" \
        "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam${beam}.${trglang}" $seed $srclang $trglang "small"
    bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_b=${beam}_big_seed=${seed}" \
        "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${seed}/kd.beam${beam}.${trglang}" $seed $srclang $trglang "big"
fi

# Train the T-Base baseline model, 100k steps
bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/baseline_seed=${seed}" \
    "${root}/wmt20/${langpair}/train.${trglang}" $seed $srclang $trglang "base"
