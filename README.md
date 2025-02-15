# memseqkd

#### Set up

1. Set up the conda env using `./install.sh`
2. Install [Comet](https://github.com/Unbabel/COMET), version 1.2.0
3. Install [Marian](https://marian-nmt.github.io/quickstart/), version v1.12.16 b61755b65
4. Obtain WMT20 and CommonCrawl data from https://www.statmt.org/wmt20/translation-task.html, and Pulpo data from https://huggingface.co/datasets/linhd-postdata/pulpo.


#### Model training & evaluation

1. Run the following bash script, where the `<langpair>` is the WMT20 language pair (one of `en-de`, `pl-en`, `fr-de`), and `<srclang>` and `<trglang>` are the direction in which we're training. For `en-de` only, additional sizes will be trained. `<beam>` is used to generate the KD corpus:
   ```./model_training/training_pipeline.sh <langpair> <srclang> <trglang> <beam>```
2. Generate translations and evaluate their quality, afterwards, where `<input>` is one of `test`, `pulpo`, `commoncrawl`:
    ```./model_training/evaluation.sh <langpair> <srclang> <trglang> <input>```

#### Additional experiments and visualization

- For the computation of Extractive Memorization, consider the dedicated readme in `extractive_memorization`;
- For the computation of hallucinations, consider the dedicated readme in `hallucinations`;
- For the approximation of counterfactual memorization, consider the dedicated readme in `counterfactual_memorization`;
- For the subgroup analysis and overall visualization functionalities, consider the dedicated readme in `analysis`.

