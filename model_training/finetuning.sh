model_dir="../model_zoo/en-de/teacher_seed=1111_finetuned-${1}-${2}"
mkdir $model_dir
base_model_dir="../model_zoo/en-de/teacher_seed=1111"
data_dir="../wmt20/en-de"

cp $base_model_dir/model.npz $model_dir/
cp $base_model_dir/model.npz.yml $model_dir/
cp $base_model_dir/model.npz.decoder.yml $model_dir/

# Data directory should have the ft* and val* files present
echo "Starting Finetuning"
$MARIAN_BINARIES/marian \
    --model $model_dir/model.npz --type transformer \
    --shuffle batches \
    --no-restore-corpus \
    --train-sets $data_dir/train.finetuning-${1}-${2}.en $data_dir/train.finetuning-${1}-${2}.de \
    --workspace -4096 \
    --early-stopping 20 \
    --after-epochs 1 \
    --max-length 256 \
    --vocabs $data_dir/joint.model.spm $data_dir/joint.model.spm \
    --valid-freq 200 --save-freq 200 --disp-freq 10 \
    --valid-metrics cross-entropy perplexity bleu chrf \
    --valid-sets $data_dir/valid.en $data_dir/valid.de \
    --valid-mini-batch 64 \
    --beam-size 1 \
    --log $model_dir/train.log --valid-log $model_dir/valid.log \
    --enc-depth 6 --dec-depth 6 \
    --transformer-heads 16 \
    --dim-emb 1024 \
    --transformer-ffn-activation relu \
    --transformer-dim-ffn 4096 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 --transformer-dropout-attention 0.1 --transformer-dropout-ffn 0.1 --label-smoothing 0.1 \
    --learn-rate 3.2660e-05 --optimizer adam --cost-type ce-sum --lr-report \
    --optimizer-params 0.9 0.98 1e-09 --clip-norm 0.0 \
    --tied-embeddings \
    --devices 0 1 2 3 4 5 6 7 --sync-sgd --seed 1111
