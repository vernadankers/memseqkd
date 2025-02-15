#!/bin/bash

export MARIAN_BINARIES=/mnt/internship/utils/marian_binaries_updated

root="/mnt/internship/Finding-Memo"
# # Model training
# bash train_teacher.sh "${root}/wmt20" "${root}/model_zoo/teacher_seed=1111" 1111
# bash train_student.sh "${root}/wmt20" "${root}/model_zoo/student_b=1_seed=1111" "${root}/model_zoo/teacher_seed=1111/kd.beam1.de" 1111
# bash train_student.sh "${root}/wmt20" "${root}/model_zoo/student_b=2_seed=1111" "${root}/model_zoo/teacher_seed=1111/kd.beam2.de" 1111
# bash train_student.sh "${root}/wmt20" "${root}/model_zoo/student_b=5_seed=1111" "${root}/model_zoo/teacher_seed=1111/kd.beam5.de" 1111
# bash train_student.sh "${root}/wmt20" "${root}/model_zoo/student_b=10_seed=1111" "${root}/model_zoo/teacher_seed=1111/kd.beam10.de" 1111
# bash train_baselines.sh "${root}/wmt20" "${root}/model_zoo/base100k_seed=1111" 100000 1111
# bash train_baselines.sh "${root}/wmt20" "${root}/model_zoo/base300k_seed=1111" 300000 1111

# # # Translate WMT20 test set
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/teacher_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/teacher_seed=1111/test.hyp" 5
bash translate.sh "${root}/wmt20" "${root}/model_zoo/student_b=2_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/student_b=2_seed=1111/test.hyp" 5
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/student_b=1_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/student_b=1_seed=1111/test.hyp" 5
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/student_b=5_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/student_b=5_seed=1111/test.hyp" 5
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/student_b=10_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/student_b=10_seed=1111/test.hyp" 5
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/teacher_baseline_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/teacher_baseline_seed=1111/test.hyp" 5
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/student_baseline_seed=1111" "${root}/wmt20/test.en" "${root}/model_zoo/student_baseline_seed=1111/test.hyp" 5

# Score WMT20 translations
for folder in "student_b=2" #"student_b=5" "student_b=10" "student_baseline" "teacher_baseline"
do
    sacrebleu "${root}/wmt20/test.de" -i "${root}/model_zoo/${folder}_seed=1111/test.hyp" -m bleu -b -w 2
    sacrebleu "${root}/wmt20/test.de" -i "${root}/model_zoo/${folder}_seed=1111/test.hyp" -m chrf -b -w 2
    sacrebleu "${root}/wmt20/test.de" -i "${root}/model_zoo/${folder}_seed=1111/test.hyp" -m ter -b -w 2
    comet-score -s "${root}/wmt20/test.en" -t "${root}/model_zoo/${folder}_seed=1111/test.hyp" -r "${root}/wmt20/test.de" --model /mnt/internship/comet-22/wmt20-comet-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/test.en" -t "${root}/model_zoo/${folder}_seed=1111/test.hyp" -r "${root}/wmt20/test.de" --model /mnt/internship/comet-22/wmt22-comet-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/test.en" -t "${root}/model_zoo/${folder}_seed=1111/test.hyp" -r "${root}/wmt20/test.de" --model /mnt/internship/comet-22/wmt20-comet-qe-da/checkpoints/model.ckpt --quiet
    comet-score -s "${root}/wmt20/test.en" -t "${root}/model_zoo/${folder}_seed=1111/test.hyp" -r "${root}/wmt20/test.de" --model /mnt/internship/comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt --quiet
done

# # Generate KD corpora
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/teacher_seed=1111" "${root}/wmt20/train.en" "${root}/model_zoo/teacher_seed=1111/kd.beam1.de" 1
# bash translate.sh "${root}/wmt20" "${root}/model_zoo/teacher_seed=1111" "${root}/wmt20/train.en" "${root}/model_zoo/teacher_seed=1111/kd.beam5.de" 5

# bash translate.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/wmt20/train.0-10000000.en" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111/kd.beam1.0-10000000.de" 1
# bash translate.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/wmt20/train.10000000-20000000.en" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111/kd.beam1.10000000-20000000.de" 1
# bash translate2.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/wmt20/train.20000000-30000000.en" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111/kd.beam1.20000000-30000000.de" 1
# bash translate2.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/wmt20/train.30000000-40000000.en" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111/kd.beam1.30000000-40000000.de" 1
# bash translate2.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/wmt20/train.40000000-50000000.en" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111/kd.beam1.40000000-50000000.de" 1

#bash train_student.sh "/mnt/internship/Finding-Memo/wmt20" "/mnt/internship/Finding-Memo/model_zoo/student_b=10_seed=1111" "/mnt/internship/Finding-Memo/model_zoo/teacher_seed=1111/kd.beam2.de" 1111
