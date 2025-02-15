#!/bin/bash
langpair=$1
srclang=$2
trglang=$3
model=$4
mode=$5
iter=$6
root="../"
wmt_dir="${root}/wmt20/${langpair}"
model_dir="${root}/model_zoo/${srclang}-${trglang}/${model}"

if [ "$mode" = 'corpus' ]; then
    em_dir="${model_dir}/em_corpus2"
    mkdir -p $em_dir
    # Based on Reference, extract which sentences are exactly replicated
    python extract_memorized.py --trg ${wmt_dir}/train.${trglang} --hyp ${model_dir}/kd.beam1.${trglang} --src ${wmt_dir}/train.${srclang} --out ${em_dir}/memorized --srclang $srclang --trglang $trglang
else
    teacher="${root}/model_zoo/${srclang}-${trglang}/teacher_seed=1111"
    em_dir="${model_dir}/em_teacher2"
    beam=$7
    mkdir -p $em_dir
    # Based on Reference, extract which sentences are exactly replicated
    python extract_memorized.py --trg ${teacher}/kd.beam${beam}.${trglang} --hyp $model_dir/kd.beam1.${trglang} --src ${wmt_dir}/train.${srclang} --out ${em_dir}/memorized --srclang $srclang --trglang $trglang
fi

# Get the prefixes of those samples and translate prefixes
python generate_prefixes.py $em_dir/memorized > $em_dir/prefixes
cat $em_dir/prefixes | cut -f2 > $em_dir/prefixes.src
#bash translate.sh $model_dir $em_dir/prefixes.src $em_dir/translations $wmt_dir model.iter${6}.npz
python translate_pymarian.py --source $em_dir/prefixes.src --target $em_dir/translations --wmt $wmt_dir --model ${model_dir}/model.iter${iter}.npz
echo "Extracted prefixes and translations"

# Observe how quickly the prefixes generate the memorized translations
python parse_memorized.py $em_dir/prefixes $em_dir/translations > $em_dir/extractively_memorised.tsv
