import argparse
import random
import pickle
import logging
from collections import Counter
from nltk import ngrams
from transformers import BasicTokenizer
from collections import Counter
from comet import load_from_checkpoint
import tqdm


# This one is just hashing the targets to get the target repeats for unique sources
class NaturalHallucination():
    def __init__(self, **kwargs) -> None:
        self.name = 'target_repeat'
        self.tokenizer = BasicTokenizer()
        self.repeat_size = 5        
        print("Running Detector {}".format(self.name))
        self.num_relevant = 0
        self.cases = []

    def reset(self):
        self.num_relevant = 0
        self.cases = []

    def hash_target(self, srcs, trgs):
        self.target_hash = {}
        for src, tgt in tqdm.tqdm(zip(srcs, trgs), desc="hashing..."):
            if tgt not in self.target_hash:
                self.target_hash[tgt] = [src]
            else:
                self.target_hash[tgt].append(src)
        return self.target_hash

    def filter(self, num, src, tgt, langpair, model) -> bool:
        self.num_relevant += 1
        if len(set(self.target_hash[tgt])) >= self.repeat_size:
            self.cases.append(num)
            return True
        return False

    def add_comet(self, subset):
        model_wmt22_comet_qe = load_from_checkpoint(
            "../../comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt")
        return model_wmt22_comet_qe.predict(subset, gpus=1).scores


class OscillatoryHallucination():
    def __init__(self, m=10, **kwargs) -> None:        
        self.tokenizer = BasicTokenizer()
        self.max_bigram_count = m
        self.max_length = 50
        self.difference = 4
        self.ngram_size = 2
        self.name = 'ngram'
        self.num_relevant = 0
        self.cases = []
        print("Running Detector {}".format(self.name))

    def reset(self):
        self.num_relevant = 0
        self.cases = []
        self.explanations = []

    def filter(self, num, source, target, langpair, model):
        if len(source.split()) < 2 or len(target.split()) < 2:
            return "too_short"

        src = self.tokenizer.tokenize(source)
        tgt = self.tokenizer.tokenize(target)
        if len(src) < 2 or len(tgt) < 2:
            return "too_short"

        self.num_relevant += 1
        src_bigrams = ngrams(src, self.ngram_size)
        src_max_bigram_count = Counter(src_bigrams).most_common(1)[0][1]
        tgt_bigrams = ngrams(tgt, self.ngram_size)
        tgt_max_bigram_count = Counter(tgt_bigrams).most_common(1)[0][1]

        if tgt_max_bigram_count > self.max_bigram_count and \
           (tgt_max_bigram_count - src_max_bigram_count) > self.difference and \
           len(src) < self.max_length:
            self.cases.append(num)
            self.explanations.append(
                " ".join(Counter(
                    ngrams(tgt, self.ngram_size)).most_common(1)[0][0]))
            return "hal"
        if tgt_max_bigram_count > self.max_bigram_count and \
           (tgt_max_bigram_count - src_max_bigram_count) > self.difference:
            self.cases.append(num)
            #print(" ".join(src) + "---" + " ".join(tgt))
            self.cases.append(num)
            self.explanations.append(
                " ".join(Counter(
                    ngrams(tgt, self.ngram_size)).most_common(1)[0][0]))
            return "long_hal"
        if len(src) >= self.max_length:
            return "long"
        return "other"


def score_oschal(m, langpair, srclang, trglang, wmt_path, model_zoo_path, models,
                 source, target, setup, step_size):
    """
    Collect Oscillatory Hallucinations: for bigrams that appear more than n
    times, if they appear at least m times more than in the source,
    for sequences below length l, mark them as a hallucination.
    """
    log = logging.getLogger(__name__)
    osc_filter = OscillatoryHallucination(m)
    results = dict()
    for model in models:
        srcs = open(
            f"{wmt_path}/{langpair}/{source}.{srclang}",
            encoding='utf-8').readlines()
        trgs = open(
            f"{model_zoo_path}/{srclang}-{trglang}/{model}/{target}.{trglang}",
            encoding='utf-8').readlines()
        osc_filter.reset()
        osc_results = []
        for i in tqdm.tqdm(range(0, len(trgs), step_size)):
            if not srcs[i].strip() or not trgs[i].strip():
                osc_results.append("empty")
                continue
            osc_results.append(
                osc_filter.filter(i, srcs[i], trgs[i], f"{srclang}-{trglang}", model))
        counter = Counter(osc_results)
        org_hal = 100 * counter['hal'] / (counter['hal'] + \
            counter['long_hal'] + counter['long'] + counter['other'])
        short_hal = 100 * counter['hal'] / (counter['hal'] + \
            counter['other'])
        long_hal = 100 * (counter['hal'] + counter['long_hal']) / \
            (counter['long'] + counter['hal'] + counter['long_hal'] + \
            counter['other'])
        results[model] = {
            "results": osc_results,
            "cases": osc_filter.cases,
            "counter": counter,
            "org_hal": org_hal,
            "short_hal": short_hal,
            "long_hal": long_hal}
        log.info(counter['hal']/osc_filter.num_relevant)
        log.info(f"{srclang}-{trglang}-{source}-{model}: {org_hal:.5f} {short_hal:.5f} {long_hal:.5f}")
        pickle.dump(
        results,
        open(f"{srclang}-{trglang}/oschal_{setup}_{source}.pickle", 'wb'))


def score_nathal(langpair, srclang, trglang, wmt_path, model_zoo_path, models,
                 source, target, setup):
    """
    Collect Natural Hallucinations: if >=5 source sequences map to the
    same translation, mark it as a natural hallucination.
    Filter potential paraphrases using Comet-QE.
    """
    random.seed(0)
    log = logging.getLogger(__name__)
    log.info(f"NatHal {srclang}-{trglang}-{source}")
    nat_filter = NaturalHallucination()
    results = dict()
    for model in models:
        srcs = open(
            f"{wmt_path}/{langpair}/{source}.{srclang}",
            encoding='utf-8').readlines()
        trgs = open(
            f"{model_zoo_path}/{srclang}-{trglang}/{model}/{target}.{trglang}",
            encoding='utf-8').readlines()
        ids = range(len(srcs))
        mapping = {(s, t): i for s, t, i in zip(srcs, trgs, ids)}
        print("Starting")
        unique = list(set(list(zip(srcs, trgs))))
        print(len(srcs), len(unique))
        srcs, trgs = zip(*unique)
        nat_filter.hash_target(srcs, trgs)
        nat_filter.reset()

        # Pass examples through filter, collect subset of hits
        nat_results = []
        subset = []
        for s, t in tqdm.tqdm(zip(srcs, trgs)):
            i = mapping[s, t]
            if not s.strip() or not t.strip():
                nat_results.append("empty")
                continue
            result = nat_filter.filter(i, s, t, f"{srclang}-{trglang}", model)
            if result:
                subset.append({"src": s, "mt": t})
            nat_results.append(result)

        # Compute COMET-QE scores to filter hits
        com = nat_filter.add_comet(subset)
        assert len(nat_filter.cases) == len(com)

        # Filter hits based on 3 COMET-QE thresholds
        below_t3 = len([i for i, c in zip(nat_filter.cases, com) if c < 0.85])
        results[model] = {
            "results": nat_results,
            "num_relevant": nat_filter.num_relevant,
            "cases": nat_filter.cases,
            "comet": com,
            "score_0.85": below_t3/nat_filter.num_relevant * 100,
            }
        log.info(f"{srclang}-{trglang}-{source}-{model}: "+
                 f"{results[model]['score_0.85']}")
        pickle.dump(
            results,
            open(f"{srclang}-{trglang}/nathal_{setup}_{source}.pickle", 'wb'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmt_path", default="../wmt20/")
    parser.add_argument("--model_zoo_path", default="../model_zoo/")
    parser.add_argument("--langpair", type=str,
                        choices=["en-de", "de-en", "pl-en", "en-pl", "fr-de"],
                        help="Lang pair for WMT corpus")
    parser.add_argument("--srclang", type=str, choices=["en", "de", "pl", "fr"],
                        help="Source language for models trained")
    parser.add_argument("--trglang", type=str, choices=["en", "de", "pl", "fr"],
                        help="Target language for models trained")
    parser.add_argument("--source", type=str, required=True,
                        help="Filename of source sequences, excl source lang")
    parser.add_argument("--target", type=str, required=True,
                        help="Filename of target sequences, excl target lang")
    parser.add_argument("--models", type=str,
                        default=["teacher_seed=1111", "student_b=1_seed=1111",
                        "baseline_seed=1111"], nargs="+",
                        help="Model names to compute hallucinations for")
    parser.add_argument("--oschal", action="store_true",
                        help="Flag for computing oscillatory hallucinations")
    parser.add_argument("--nathal", action="store_true",
                        help="Flag for computing natural hallucinations")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--setup", type=str, required=True)
    parser.add_argument("--m", type=int, default=10)
    args = parser.parse_args()

    if args.oschal:
        score_oschal(
            args.m, args.langpair, args.srclang, args.trglang, args.wmt_path,
            args.model_zoo_path, args.models, args.source, args.target,
            args.setup, args.step_size)

    if args.nathal:
        score_nathal(
            args.langpair, args.srclang, args.trglang, args.wmt_path,
            args.model_zoo_path, args.models, args.source, args.target, args.setup)
