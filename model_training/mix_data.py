import copy
import argparse
import numpy as np
import tqdm
import random
from collections import Counter


def mix_data(langpair, srclang, trglang, wmt_path, model_zoo_path):
    trgs = open(f"{wmt_path}/{langpair}/train.{trglang}", encoding="utf-8").readlines()
    teacher_trgs = open(f"{model_zoo_path}/{srclang}-{trglang}/teacher_seed=1111/kd.beam1.{trglang}", encoding="utf-8").readlines()
    comet = open(f"{wmt_path}/{langpair}/comet-qe_{srclang}-{trglang}.tsv", encoding="utf-8").readlines()[1:]

    length = len(trgs)
    n = int(length * 0.9)

    scores = Counter({i: float(c.split('\t')[-1].strip()) for i, c in enumerate(comet)})
    teacher_indices, _ = zip(*scores.most_common()[n:])
    teacher_indices = set(teacher_indices)

    mixmatch_targets = copy.deepcopy(trgs)
    for i in range(len(trgs)):
        if i in teacher_indices:
            mixmatch_targets[i] = teacher_trgs[i]
    with open(f"{wmt_path}/{langpair}/train.mix-match-teacher.{trglang}", 'w', encoding="utf-8") as f:
        for l in mixmatch_targets:
            f.write(l)

    random_indices = set([i for i in range(len(trgs)) if random.random() < 0.101][:length - n])
    randomised_targets = copy.deepcopy(trgs)
    for i in range(len(trgs)):
        if i in random_indices:
            randomised_targets[i] = teacher_trgs[i]
    with open(f"{wmt_path}/{langpair}/train.mix-match-random.{trglang}", 'w', encoding="utf-8") as f:
        for l in randomised_targets:
            f.write(l)


def finetuning_data(ch_threshold, co_threshold):
    confidence = [float(x.split()[0]) for x in open("../model_zoo/en-de/teacher_seed=1111/kd.beam1.de.scores").readlines()[1:]]
    chrf =  [float(x) for x in open("../model_zoo/en-de/teacher_seed=1111/chrf_.txt").readlines()]
    srcs = [x for x in open("../wmt20/en-de/train.en", encoding="utf-8").readlines()]
    trgs = [x for x in open("../wmt20/en-de/train.de", encoding="utf-8").readlines()]
    a, b = 0, 0
    lengths = []
    indices = []
    for i, (s, ch, co) in tqdm.tqdm(enumerate(zip(srcs, chrf, confidence))):
        co = 2 ** co
        if ch > ch_threshold and co > co_threshold and len(s.split()) > 5:
            a += 1
            lengths.append(len(s.split()))
            indices.append(i)
        b += 1
    print(a/b, np.mean(lengths), np.std(lengths))
    random.shuffle(indices)
    indices = indices[:200000]
    with open(f"../wmt20/en-de/train.finetuning-{ch_threshold}-{co_threshold}.en", 'w', encoding="utf-8") as f_en, \
         open(f"../wmt20/en-de/train.finetuning-{ch_threshold}-{co_threshold}.de", 'w', encoding="utf-8") as f_de:
        for i in indices:
            f_en.write(srcs[i])
            f_de.write(trgs[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmt_path", default="/mnt/internship/memseqkd/wmt20/")
    parser.add_argument("--model_zoo_path", default="/mnt/internship/memseqkd/model_zoo/")
    parser.add_argument("--langpair", type=str)
    parser.add_argument("--srclang", type=str)
    parser.add_argument("--trglang", type=str)
    parser.add_argument("--models", type=str,
        default=["teacher_seed=1111", "student_b=1_seed=1111",
        "baseline_seed=1111"], nargs="+")
    args = parser.parse_args()
    #mix_data(args.langpair, args.srclang, args.trglang, args.wmt_path, args.model_zoo_path)

    for ch_threshold in [85, 90, 95]:
        for co_threshold in [0.85, 0.9, 0.95]:
            finetuning_data(ch_threshold, co_threshold)
