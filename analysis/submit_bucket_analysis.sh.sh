#!/bin/bash

for bucket in low_confidence high_confidence comet22_0.8-1 comet22_0.6-0.8 comet22_0.4-0.6 comet22_0.2-0.4 comet22_0-0.2 cm_0-0.2 cm_0.2-0.3 cm_0.3-0.4  #random cm_0.4-1 cm_bottomleft cm_topright
do
    for langs in "en-de_en-de" "en-de_de-en" "pl-en_pl-en" "pl-en_en-pl" "fr-de_fr-de"
    do
        langpair=${langs%_*}
        srctrg=${langs#*_}
        src=${srctrg%-*}
        trg=${srctrg#*-}

        # echo $langpair
        # echo $src
        # echo $trg

        sbatch --exclude=buccleuch -o logs/${src}-${trg}_${bucket}.out --gres=gpu:1 -t 24:00:00 -p "MandI-Standard" analyze_buckets.sh $langpair $src $trg $bucket
    done
done