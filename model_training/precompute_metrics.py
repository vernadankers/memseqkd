from comet import download_model, load_from_checkpoint
from comet.models.regression.referenceless import ReferencelessRegression
from sacrebleu import CHRF
import argparse
import evaluate
import random
import tqdm
import numpy as np
import os


def get_chrf(wmt_path, model_path, trglang):
    chrf = CHRF()
    with open(f"{wmt_path}/train.{trglang}", encoding="utf-8") as f_trg, \
         open(f"{model_path}/teacher_seed=1111/kd.beam1.{trglang}", encoding="utf-8") as f_hyp, \
         open(f"{model_path}/teacher_seed=1111/chrf_.txt", 'w') as f_out:
        for trg, hyp in tqdm.tqdm(zip(f_trg, f_hyp)):
            f_out.write(f"{chrf.sentence_score(hyp, [trg]).score}\n")


def get_perplexity(wmt_path, srclang, trglang):
    perplexity = evaluate.load("perplexity", module_type="metric")
    bsz = 1000
    with open(wmt_path + f"/train.{srclang}") as f, open(f"{wmt_path}/train.{srclang}.perplexity", 'w') as f_out:
        batch = []
        for i, l in enumerate(f):
            if len(batch) == bsz:
                if random.random() < 0.05:
                    try:
                        ids, input_texts = zip(*batch)
                        results = perplexity.compute(model_id='ai-forever/mGPT',
                                                    add_start_token=False,
                                                    predictions=input_texts)['perplexities']
                        for j, x in zip(ids, results):
                            f_out.write(str(x) + "\n")
                    except:
                        for j in batch:
                            f_out.write("-\n")
                else:
                    for j in batch:
                        f_out.write("-\n")
                batch = []
            else:
                batch.append((i, l.strip()))


def get_comet(wmt_path, srclang, trglang, start, stop):
    bsz = 1000
    model_wmt20_comet_qe = load_from_checkpoint("/mnt/internship/comet-22/wmt20-comet-qe-da/checkpoints/model.ckpt")
    model_wmt22_comet_qe = load_from_checkpoint("/mnt/internship/comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt")
    srcs = open(f"{wmt_path}/train.{srclang}", encoding='utf-8').readlines()[start:stop]
    trgs = open(f"{wmt_path}/train.{trglang}", encoding="utf-8").readlines()[start:stop]

    with open(f"{wmt_path}/comet_{srclang}-{trglang}_{start}-{stop}.tsv", 'w') as f_out:
        f_out.write("wmt20-comet-qe-da\twmt22-cometkiwi-da\n")
        batch_qe = []
        for i, (src, trg) in tqdm.tqdm(enumerate(zip(srcs, trgs))):    
            if len(batch_qe) == bsz:
                scores_wmt20_comet_qe = model_wmt20_comet_qe.predict(batch_qe, batch_size=8, gpus=1).scores
                scores_wmt22_comet_qe = model_wmt22_comet_qe.predict(batch_qe, batch_size=8, gpus=1).scores
                scores = zip(scores_wmt20_comet_qe, scores_wmt22_comet_qe)
                for score in scores:
                    score = [str(n) for n in score]
                    f_out.write('\t'.join(score) + "\n")
                batch_qe = []
            batch_qe.append({"src": src, "mt": trg})


def collect_comet(langpair, srclang, trglang):
    model_wmt20_comet_qe = load_from_checkpoint("/mnt/internship/comet-22/wmt20-comet-qe-da/checkpoints/model.ckpt")
    model_wmt22_comet_qe = load_from_checkpoint("/mnt/internship/comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt")
    srcs = open(f"../wmt20/{langpair}/train.{srclang}").readlines()
    trgs = open(f"../wmt20/{langpair}/train.{trglang}").readlines()
    num_examples = len(srcs)
    comet_scores = ['-\t-\n'] * num_examples

    for filename in os.listdir(f"../wmt20/{langpair}"):
        if f"_{srclang}-{trglang}_" in filename:
            [_, _, startstop] = filename.split('_')
            [start, stop] = startstop.replace(".tsv", "").split('-')
            start = int(start)
            for l in open(f"../wmt20/{langpair}/{filename}").readlines()[1:]:
                comet_scores[start] = l
                start += 1

    ids = []
    fill_in = []
    for i, (s, t) in enumerate(zip(srcs, trgs)):
        if comet_scores[i] == '-\t-\n':
            ids.append(i)
            fill_in.append({"src": s, "mt": t})
            
    cq20 = model_wmt20_comet_qe.predict(fill_in, gpus=1).scores
    cq22 = model_wmt22_comet_qe.predict(fill_in, gpus=1).scores

    for i, l in enumerate(comet_scores):
        if l == '-\t-\n':
            assert ids.pop(0) == i
            comet_scores[i] = f"{cq20.pop(0)}\t{cq22.pop(0)}\n"

    with open(f"../wmt20/{langpair}/comet-qe_{srclang}-{trglang}.tsv", 'w') as f:
        f.write("wmt20-comet-qe-da\twmt22-cometkiwi-da\n")
        for l in comet_scores:
            f.write(l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langpair")
    parser.add_argument("--srclang")
    parser.add_argument("--trglang")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #get_comet(
    #    f"/mnt/internship/memseqkd/wmt20/{args.langpair}",
    #    args.srclang, args.trglang, args.start, args.stop)

    #get_perplexity(f"/mnt/internship/memseqkd/wmt20/{args.srclang}-{args.trglang}", args.srclang, args.trglang)

    #collect_comet(args.langpair, args.srclang, args.trglang)

    get_chrf(
        f"/mnt/internship/memseqkd/wmt20/{args.langpair}",
        f"/mnt/internship/memseqkd/model_zoo/{args.srclang}-{args.trglang}",
        args.trglang)
