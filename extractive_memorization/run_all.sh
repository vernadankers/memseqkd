#!/bin/bash
langpair=$1
srclang=$2
trglang=$3

bash extract_memorized.sh $langpair $srclang $trglang teacher_seed=1111 corpus 300000
bash extract_memorized.sh $langpair $srclang $trglang baseline_seed=1111 corpus 100000
bash extract_memorized.sh $langpair $srclang $trglang student_b=1_seed=1111 corpus 100000
bash extract_memorized.sh $langpair $srclang $trglang student_b=1_seed=1111 teacher 100000 1
