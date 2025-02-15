scriptdir="$( dirname -- "$BASH_SOURCE"; )";
echo "Calling Install from ${scriptdir}..."
cd /mnt/internship/memseqkd/
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="~/miniconda3/bin:$PATH"
source ~/.bashrc
conda init bash
conda info --envs
source /home/aiscuser/.bashrc
conda init
##############################################################
conda create -n exp2 python=3.8
conda activate exp2
pip install numpy matplotlib sacrebleu emoji spacy fasttext
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install sentencepiece sacrebleu sacremoses matplotlib sklearn unbabel-comet faiss-gpu sentence_transformers
cd $scriptdir
sudo apt-get install nano unzip icdiff
pip install sentencepiece transformers tqdm
#source /home/viraunak/miniconda3/bin/activate
source ~/miniconda3/bin/activate
##############################################################
source /home/aiscuser/.bashrc
conda init
conda activate exp2
pip install tqdm matplotlib torch==1.12.1 sacrebleu==2.2.1 sentencepiece==0.1.97 plotly seaborn
pip install 'gem-metrics[heavy] @ git+https://github.com/GEM-benchmark/GEM-metrics.git'
pip install fasttext
cd /mnt/internship/comet-22
pip install poetry
poetry install
