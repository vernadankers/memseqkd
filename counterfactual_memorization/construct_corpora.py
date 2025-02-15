import argparse
import os
import random
import json
import numpy as np
import tqdm


def prepare_corpora(src_file: str, overwrite: bool, srclang: str, trglang: str, P: int):
    """
    For 10 seeds, distribute P% of the datapoints such that each seed
    sees ~50% of that data as training data and ~50% as testing data.
    Ensure each datapoint appears 5 times as training index and
    5 times as testing index.
    Stores dict of indices to file in json format (`indices.json`).
    Args:
        - src_file: filename of the training corpus, source side
        - overwrite: whether to overwrite existing json indices file
        - srclang: one of en | de | fr | pl 
        - trglang: one of en | de | fr | pl
        - P: how much of the data to include in the CM computation
    """
    random.seed(0)
    json_file = f"{srclang}-{trglang}/indices.json"
    if not os.path.exists(json_file) or overwrite:
        # Some examples are train-only, split others 50/50 between train & test
        seeds = list(range(1, 11))
        indices = {seed: {"train": set(), "test": set()} for seed in seeds}
        for i in tqdm.tqdm(range(len(open(src_file).readlines())),
                        desc="Distributing data over train/test per seed..."):
            if random.random() > P / 100:
                for s in seeds:
                    indices[s]["train"].add(i)
            else:
                random.shuffle(seeds)
                for s in seeds[:5]:
                    indices[s]["train"].add(i)
                for s in seeds[5:]:
                    indices[s]["test"].add(i)
        indices = {seed: {"train": sorted(list(indices[seed]["train"])),
                        "test": sorted(list(indices[seed]["test"]))}
                        for seed in indices}

        # Store to file in JSON format
        with open(json_file, "w") as outfile:
            json.dump(indices, outfile)
    else:
        print("File exists, not overwriting it.")


def construct_corpus(seed: int, src_file: str, trg_file: str, srclang: str, trglang: str):
    """
    Construct train / test files based on indices stored to
    file for the given seed.
    Store under `{folder}/tmp_corpus/train.XX`
    Args:
        - seed: int indicating the seed to use
        - src_file: original WMT source seqs file
        - trg_file: original WMT target seqs file
        - srclang: one of en | de | fr | pl
        - trglang: one of en | de | fr | pl
    """
    indices = json.load(open(f"{srclang}-{trglang}/indices.json"))
    assert str(seed) in indices, f"Seed {seed} not available..."
    train = set(indices[str(seed)]["train"])
    with open(src_file, encoding='utf-8') as f_src, \
         open(trg_file, encoding='utf-8') as f_trg, \
         open(f"{srclang}-{trglang}/tmp_corpus/train.{seed}.src", 'w',
              encoding='utf-8') as f_src_out, \
         open(f"{srclang}-{trglang}/tmp_corpus/train.{seed}.trg", 'w',
              encoding='utf-8') as f_trg_out:
        for i, (s, t) in enumerate(zip(f_src, f_trg)):
            if i in train:
                f_src_out.write(s)
                f_trg_out.write(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--src_file", type=str, default="../wmt20/en-de/train.en")
    parser.add_argument("--trg_file", type=str, default="../wmt20/en-de/train.de")
    parser.add_argument("--srclang", type=str, default="en")
    parser.add_argument("--trglang", type=str, default="de")
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    np.random.seed(0)
    random.seed(0)

    assert os.path.exists(args.src_file), f"{args.src_file} doesn't exist..."
    assert os.path.exists(args.trg_file), f"{args.trg_file} doesn't exist..."

    if args.prepare:
        prepare_corpora(args.src_file, args.overwrite, args.srclang,
                        args.trglang, args.percentage)
    else:
        construct_corpus(args.seed, args.src_file, args.trg_file,
                         args.srclang, args.trglang)
