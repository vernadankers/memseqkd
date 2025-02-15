from collections import defaultdict, Counter
import argparse
import random
import os
import tqdm


class Bucketer():
    """
    Class with functions to create subgroups of examples.
    The general workflow is that each function compiles a list of ids,
    and stores those ids to file.
    """
    def __init__(self, model_zoo_path, wmt_path, srclang, trglang, models):
        self.model_zoo_path = model_zoo_path
        self.wmt_path = wmt_path
        self.srclang = srclang
        self.trglang = trglang
        self.models = models

    def comet(self):
        """
        Compile five groups based on Comet-QE scores of the WMT20 corpus.
        This function assumes precomputed comet-qe tsv files in the WMT folder.
        """
        print("Comet buckets...")
        comet_ids = defaultdict(list)
        path = f"{self.wmt_path}/comet-qe_{self.srclang}-{self.trglang}.tsv"
        assert os.path.exists(path), "Comet-QE tsv doesn't exist!"
        scores = open(path).readlines()[1:]
        for i, n in tqdm.tqdm(enumerate(scores)):
            _, com22 = n.strip().split("\t")
            if com22.strip() != '-':
                com22 = float(com22)
                if com22 < 0.2:
                    comet_ids["comet22_0-0.2"].append(i)
                elif com22 < 0.4:
                    comet_ids["comet22_0.2-0.4"].append(i)
                elif com22 < 0.6:
                    comet_ids["comet22_0.4-0.6"].append(i)
                elif com22 < 0.8:
                    comet_ids["comet22_0.6-0.8"].append(i)
                else:
                    comet_ids["comet22_0.8-1"].append(i)
        return comet_ids

    def cm(self):
        """
        Compile six groups based on counterfactual memorization scores. This
        function assumes precomputed CM scores in ../counterfactual_memorization
        """
        print("CM buckets...")
        buckets = defaultdict(list)
        path = f"../counterfactual_memorization/{self.srclang}-{self.trglang}/cm_prob.tsv"
        assert os.path.exists(path), f"CM scores ({path}) don't exist!"
        with open(path) as f:
            for i, l in enumerate(f):
                train, test, cm = l.strip().split()
                train = float(train)
                test = float(test)
                cm = float(cm)
                if cm < 0.2:
                    buckets["cm_0-0.2"].append(i)
                elif 0.2 <= cm < 0.3:
                    buckets["cm_0.2-0.3"].append(i)
                elif 0.3 <= cm < 0.4:
                    buckets["cm_0.3-0.4"].append(i)
                elif cm >= 0.4:
                    buckets["cm_0.4-1"].append(i)
                    
                if train <= 0.2 and test <= 0.2:
                    buckets["cm_bottomleft"].append(i)
                if train >= 0.8 and test >= 0.8:
                    buckets["cm_topright"].append(i)
        return buckets

    def randomized(self):
        """
        Compile one extra large random bucket with 50k random ids.
        """
        print("Random bucket...")
        random.seed(1)
        path = f"{self.wmt_path}/train.{args.srclang}"
        corpus_size = len(open(path).readlines())
        ids = []
        for i in tqdm.tqdm(range(corpus_size)):
            if random.random() > 0.9:
                ids.append(i)
        return {"random": random.sample(ids, 50000)}

    def confidence(self):
        """
        Create two buckets, with the top and bottom 10k examples based on
        the confidence with which the teacher itself generated examples.
        """
        confidence = Counter()
        scored_translations = open(
            f"{self.model_zoo_path}/teacher_seed=1111/kd.beam1.{self.trglang}.scores"
        ).readlines()
        for i, l in enumerate(scored_translations):
            prob = 2 ** float(l.split()[0])
            confidence[i] = prob
        confidence_ranked = confidence.most_common()
        highest, _ = zip(*confidence_ranked[:10000])
        lowest, _ = zip(*confidence_ranked[-10000:])
        return {"high_confidence": highest, "low_confidence": lowest}

    def write_to_file(self, buckets: dict):
        """
        Store the bucket simply by dumping the list in a txt file.
        Take 10k examples max, except for the random 50k bucket.
        Args:
            - buckets: a dict mapping bucket names to list of ids
        """
        random.seed(0)
        for key in buckets:
            if not os.path.exists(f"{args.srclang}-{args.trglang}"):
                os.mkdir(f"{args.srclang}-{args.trglang}")
            with open(f"{args.srclang}-{args.trglang}/{key}.txt", 'w') as f:
                if len(buckets[key]) != 50000:
                    buckets[key] = random.sample(
                        buckets[key],
                        min(len(buckets[key]), 10000)
                    )
                print(f"Writing {key} to file, {len(buckets[key])} instances")
                f.write(str(buckets[key]))

    def print_examples(self):
        """
        Create a tsv with examples.
        """
        ids = dict()
        model_preds = dict()
        srcs = open(f"{self.wmt_path}/train.{args.srclang}", encoding="utf-8").readlines()
        for m in self.models:
            model_preds[m] = open(self.model_zoo_path + f"/{m}/kd.beam1.{args.trglang}", encoding="utf-8").readlines()
            print(len(model_preds[m]))

        buckets = os.listdir(f"{self.srclang}-{self.trglang}/")
        print(buckets)
        with open(f"print_examples/bucket_showcase_{self.srclang}-{self.trglang}.tsv", 'w', encoding="utf-8") as f:
            for bucket in buckets:
                if ".txt" in bucket:
                    id_ = eval(open(f"{self.srclang}-{self.trglang}/{bucket}").read())
                    id_ = [i for i in id_ if len(srcs[i].split()) > 5]
                    for i in set(random.sample(id_, min(len(id_), 100))):
                        ids[i] = bucket.replace(".txt", "")
            subset = []
            with open(f"{self.wmt_path}/train.{args.trglang}", encoding="utf-8") as f_trg:
                for i, (src, trg) in tqdm.tqdm(enumerate(zip(srcs, f_trg))):
                    if i in ids:
                        src = src.replace('\t', ' ')
                        trg = trg.replace('\t', ' ')
                        str_ = f"{ids[i]}\t{src.strip()}\t{trg.strip()}"
                        for m in self.models:
                            str_ += '\t' + model_preds[m][i].strip().replace('\t', ' ')
                        subset.append(str_ + '\n')
            for line in sorted(subset):
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_zoo_path", type=str, default="../model_zoo/en-de")
    parser.add_argument("--wmt_path", type=str, default="../wmt20/en-de")
    parser.add_argument("--models", type=str, nargs='+')
    parser.add_argument("--srclang", type=str)
    parser.add_argument("--trglang", type=str)
    args = parser.parse_args()

    bucketer = Bucketer(args.model_zoo_path, args.wmt_path, args.srclang,
                        args.trglang, args.models)
    bucketer.write_to_file(bucketer.comet())
    bucketer.write_to_file(bucketer.confidence())
    bucketer.write_to_file(bucketer.cm())
    bucketer.write_to_file(bucketer.randomized())
    #bucketer.print_examples()
