from collections import defaultdict
import json
import math
import argparse
import numpy as np
import tqdm


def compute_cm(models, trg_file, srclang, trglang):
    """
    Compute approximated CM scores.
    Args:
        - models: list of models ids to use 0 is the original one,
                  1 - 11 are all seeds trained within the cm folder
        - trg_file: file with the WMT targets
        - srclang: one of en | de | fr | pl
        - trglang: one of en | de | fr | pl
    """
    train = defaultdict(lambda: [math.nan] * len(models))
    test = defaultdict(lambda: [math.nan] * len(models))
    indices = set()
    for j, model_num in enumerate(models):
        print(f"Processing model {model_num}")
        if model_num != 0:
            indices = set(
                json.load(
                    open(f"{srclang}-{trglang}/indices.json")
                )[str(model_num)]["train"])
        print("Loaded indices...")

        # For model_num 0, take the original teacher from the model zoo
        if model_num == 0:
            fn = f"../model_zoo/{srclang}-{trglang}/teacher_seed=1111/train.scores"
        else:
            fn = f"{srclang}-{trglang}/model_{model_num}/train.scores"

        # if example i was in the train indices for this model seed,
        # add the probability (=geom mean of word probs) to the train dict
        with open(fn, encoding='utf-8') as f_probs:
            for i, prob in tqdm.tqdm(enumerate(f_probs)):
                prob = prob.split()[0]
                train[i][j] = 2**float(prob) if model_num == 0 or i in indices else math.nan
                test[i][j] = math.nan if model_num == 0 or i in indices else 2**float(prob)

    with open(trg_file, encoding='utf-8') as f_trg,\
         open(f"{srclang}-{trglang}/cm_prob.tsv", 'w') as f:
        for i, _ in tqdm.tqdm(enumerate(f_trg)):
            train[i] = np.nanmean(train[i])
            test[i] = np.nanmean(test[i])
            f.write(f"{train[i]:.3f}\t{test[i]:.3f}\t{train[i] - test[i]:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trg_file", type=str, default="../wmt20/en-de/train.de")
    parser.add_argument("--models", type=int, nargs='+', required=True)
    parser.add_argument("--srclang", type=str)
    parser.add_argument("--trglang", type=str)
    args = parser.parse_args()

    compute_cm(args.models, args.trg_file, args.srclang, args.trglang)
