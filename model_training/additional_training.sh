#!/bin/bash

export MARIAN_BINARIES=/mnt/internship/utils/marian_binaries_updated
root="/mnt/internship/memseqkd"
langpair="en-de"
seed=1111
srclang="en"
trglang="de"

echo $1

#bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_${1}" \
#    "${root}/wmt20/${langpair}/train.${1}.${trglang}" $seed $srclang $trglang "base"
#bash train_student.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_b=1_big_seed=${seed}" \
#    "${root}/model_zoo/${srclang}-${trglang}/teacher_seed=${se
bash translate.sh "${root}/wmt20/${langpair}" "${root}/model_zoo/${srclang}-${trglang}/student_b=1_small_seed=${seed}" \
	    "${root}/wmt20/${langpair}/train.${srclang}" \
	        "${root}/model_zoo/${srclang}-${trglang}/student_b=1_small_seed=${seed}/kd.beam1.${trglang}" 1 "model.iter100000.npz"

