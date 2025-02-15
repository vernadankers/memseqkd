data_dir=$1
model_dir=$2
seed=$3
srclang=$4
trglang=$5

if [ "$6" = 'base' ]; then
    depth=6
    embdim=512
    ffndim=2048
    heads=8
    workspace=-2048
elif [ "$6" = 'big' ]; then
    depth=6
    embdim=1024
    ffndim=4096
    heads=16
    workspace=-4096
elif [ "$6" = 'small' ]; then
    depth=6
    embdim=256
    ffndim=1024
    heads=8
    workspace=-2048
fi

mkdir -p ${model_dir}
$MARIAN_BINARIES/marian \
    --model $model_dir/model.npz --type transformer \
    --train-sets $data_dir/train.${srclang} $data_dir/train.${trglang} \
    --max-length 256 \
    --vocabs $data_dir/joint.model.spm $data_dir/joint.model.spm \
    --mini-batch-fit true --mini-batch-fit-step 5 --workspace -4096 --maxi-batch 1000 --mini-batch 1000 --mini-batch-warmup 4000 --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 10 \
    --valid-metrics cross-entropy perplexity bleu chrf \
    --valid-sets $data_dir/valid.${srclang} $data_dir/valid.${trglang} \
    --valid-mini-batch 64 \
    --beam-size 5 \
    --log $model_dir/train.log --valid-log $model_dir/valid.log \
    --enc-depth $depth --dec-depth $depth \
    --transformer-heads $heads \
    --dim-emb $embdim \
    --transformer-ffn-activation relu \
    --transformer-dim-ffn $ffndim \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 --transformer-dropout-attention 0.1 --transformer-dropout-ffn 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0002 --optimizer adam --cost-type ce-sum --lr-warmup 8000 --lr-decay-inv-sqrt 8000 --lr-report \
	    --optimizer-params 0.9 0.98 1e-09 --clip-norm 0 \
	        --after-batches 300000 \
		    --tied-embeddings \
		        --devices 0 1 2 3 4 5 6 7 --sync-sgd --seed $seed \
			    --exponential-smoothing 1e-4

