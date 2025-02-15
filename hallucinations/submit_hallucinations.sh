#!/bin/sh

langpair=$1
srclang=$2
trglang=$3

for corpus in "pulpo-L_pulpo-L.beam1" "commoncrawl_commoncrawl.beam1" "train_kd.beam1"
do
    var1=${corpus%_*}
    var2=${corpus#*_}
    echo $var1
    echo $var2
    sbatch -o logs/oschal_${srclang}${trglang}_${var1}_base.out --exclude=buccleuch hallucinations.sh $langpair $srclang $trglang $var1 $var2 base "teacher_seed=1111 student_b=1_seed=1111 baseline_seed=1111"
    sbatch -o logs/oschal_${srclang}${trglang}_${var1}_adaptiveseqkd.out --exclude=buccleuch hallucinations.sh $langpair $srclang $trglang $var1 $var2 adaptiveseqkd "teacher_seed=1111_finetuned_hq teacher_seed=1111_finetuned_random student_b=1_seed=1111_finetuned_random student_b=1_seed=1111_finetuned_hq"
    if [ "$srclang" == "en" ] && [ "$trglang" == "de" ]; then
        sbatch -o logs/oschal_${srclang}${trglang}_${var1}_beam.out --exclude=buccleuch hallucinations.sh $langpair $srclang $trglang $var1 $var2 beam "student_b=1_seed=1111 student_b=2_seed=1111 student_b=5_seed=1111 student_b=10_seed=1111 student_b=5_seed=1111_finetuned_hq"
        sbatch -o logs/oschal_${srclang}${trglang}_${var1}_size.out --exclude=buccleuch hallucinations.sh $langpair $srclang $trglang $var1 $var2 size "student_b=1_small_seed=1111 student_b=1_big_seed=1111"
    fi
    if [ "$srclang" == "fr" ] && [ "$trglang" == "de" ]; then
        sbatch -o logs/oschal_${srclang}${trglang}_${var1}_beam.out --exclude=buccleuch hallucinations.sh $langpair $srclang $trglang $var1 $var2 beam "student_b=1_seed=1111 student_b=2_seed=1111 student_b=5_seed=1111 student_b=10_seed=1111"
    fi
done
