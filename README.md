# memseqkd

Code for [Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](https://arxiv.org/abs/2502.01491).

Accepted at [ACL'25 (Main Conference)](https://2025.aclweb.org/).

> Abstract: In this work, we explore how instance-level memorization in the teacher Neural Machine Translation (NMT) model gets inherited by the student model in sequence-level knowledge distillation (SeqKD). We find that despite not directly seeing the original training data, students memorize more than baseline models (models of the same size, trained on the original data) -- 3.4% for exact matches and 57% for extractive memorization -- and show increased hallucination rates. Further, under this SeqKD setting, we also characterize how students behave on specific training data subgroups, such as subgroups with low quality and specific counterfactual memorization (CM) scores, and find that students exhibit amplified denoising on low-quality subgroups. Finally, we propose a modification to SeqKD named Adaptive-SeqKD, which intervenes in SeqKD to reduce memorization and hallucinations. Overall, we recommend caution when applying SeqKD: students inherit both their teachers' superior performance and their fault modes, thereby requiring active monitoring.

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

#### Citation

If you find our code useful in your research, please cite our paper:
```bibtex
@inproceedings{dankers-etal-2025-mem,
    title = "Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation",
    author = "Dankers, Verna and Raunak, Vikas,
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics"
}
```

##### Acknowledgement

Our repo leverages components from [Finding-Memo](https://github.com/vyraun/Finding-Memo), [Marian-NMT](https://github.com/marian-nmt/marian), [COMET](https://github.com/Unbabel/COMET) and [Curious Case of Hallucinations](https://github.com/vyraun/hallucinations) projects. Thank you!
