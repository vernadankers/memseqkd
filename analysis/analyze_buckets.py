import random
import os
import argparse
import numpy as np
import tqdm
import gem_metrics
import pickle
from collections import Counter, defaultdict
from sacrebleu import CHRF, BLEU
from comet import load_from_checkpoint


class Analyzer():
    def __init__(self, model_zoo_path, wmt_path, teacher, students, srclang, trglang, **kwargs):
        self.wmt_path = wmt_path
        self.model_zoo_path = model_zoo_path
        self.teacher = teacher
        self.students = students
        self.srclang = srclang
        self.trglang = trglang
        self.comet20 = load_from_checkpoint(
            "../../comet-22/wmt20-comet-da/checkpoints/model.ckpt")
        self.comet22 = load_from_checkpoint(
            "../../comet-22/wmt22-comet-da/checkpoints/model.ckpt")
        self.comet20_qe = load_from_checkpoint(
            "../../comet-22/wmt20-comet-qe-da/checkpoints/model.ckpt")
        self.comet22_qe = load_from_checkpoint(
            "../../comet-22/wmt22-cometkiwi-da/checkpoints/model.ckpt")

    def compute_metrics(self, ids: list, student_path: str):
        """
        Compute quality, memorization and diversity metrics for a given list
        of WMT identifiers. Compute it for the specific student given.
        Note that the student could also be a teacher or the WMT corpus.
        Args:
            - ids: list of example identifiers (ints)
            - student_path: folder name of student, e.g. student_b=1_seed=1111
              note that it's not a complete path, just the student folder
        Returns:
            - dictionary mapping metric names to values
        """
        random.seed(0)
        metrics = dict()
        hyps, trgs, comet_examples, comet_examples_ref = [], [], [], []
        match_corpus, match_teacher = [], []
        teacher_path = f"{self.model_zoo_path}/{self.teacher}/kd.beam1.{self.trglang}"
        student_path = f"{self.model_zoo_path}/{student_path}/kd.beam1.{self.trglang}" \
            if not "wmt" in student_path else f"{student_path}/train.{self.trglang}"

        # Collect source, target, teacher and student predictions
        # in formats that will allow us to compute multiple metrics
        with open(f"{self.wmt_path}/train.{self.srclang}", encoding='utf-8') as f_src, \
            open(f"{self.wmt_path}/train.{self.trglang}", encoding='utf-8') as f_trg, \
            open(teacher_path, encoding='utf-8') as f_teacher, \
            open(student_path, encoding='utf-8') as f_hyp:
            for i, (src, trg, teach, hyp) in tqdm.tqdm(enumerate(zip(f_src, f_trg, f_teacher, f_hyp))):
                if i in ids:
                    hyps.append(hyp)
                    trgs.append(trg)
                    match_corpus.append(trg.strip() == hyp.strip())
                    match_teacher.append(teach.strip() == hyp.strip())
                    comet_examples.append({"src": src, "mt": hyp})
                    comet_examples_ref.append({"src": src, "ref": trg, 'mt': hyp})

        # Compute quality, memorization and diversity metrics
        preds = gem_metrics.texts.Predictions(hyps)
        lexical_diversity = gem_metrics.compute(preds, metrics_list=['msttr'])
        metrics["replicate_corpus"] = np.mean(match_corpus) * 100
        metrics["replicate_teacher"] = np.mean(match_teacher) * 100
        metrics["comet20-qe"] = self.comet20_qe.predict(
            comet_examples, batch_size=8, gpus=1).system_score
        metrics["comet22-qe"] = self.comet22_qe.predict(
            comet_examples, batch_size=8, gpus=1).system_score
        metrics["comet20"] = self.comet20.predict(
            comet_examples_ref, batch_size=8, gpus=1).system_score
        metrics["comet22"] = self.comet22.predict(
            comet_examples_ref, batch_size=8, gpus=1).system_score
        metrics["bleu"] = BLEU().corpus_score(hyps, [trgs]).score
        metrics["chrf"] = CHRF().corpus_score(hyps, [trgs]).score
        metrics["msttr"] = lexical_diversity["msttr-100"]
        return metrics

    def analyze(self, ids: list, bucket_name: str):
        """
        Run quality, memorization and diversity metrics for all students,
        for the given bucket.
        Args:
            - ids: list of indices for WMT corpus
            - bucket_name: name of the bucket to which the identifiers belong
        """
        ids = set(ids)

        print("corpus", bucket_name)
        corpus = self.compute_metrics(ids, self.wmt_path)

        print("teacher", bucket_name)
        teacher = self.compute_metrics(ids, self.teacher)
        student_results = []
        results = {"corpus": corpus, "teacher": teacher} 

        for student in self.students:
            print(student, bucket_name)
            res = self.compute_metrics(ids, student)
            student_results.append(res)
            results[student] = res
        pickle.dump(results, open(bucket_name, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmt_path", default="../wmt20/en-de")
    parser.add_argument("--model_zoo_path", default="../model_zoo/en-de")
    parser.add_argument("--teacher", type=str, default="teacher_seed=1111")
    parser.add_argument("--srclang", type=str)
    parser.add_argument("--trglang", type=str)
    parser.add_argument("--bucket", type=str, nargs='+',
                        help="Name of bucket to analyze, could be multiple")
    parser.add_argument("--students", type=str,
                        default=["student_b=1_seed=1111", "baseline_seed=1111"],
                        nargs="+",
                        help="Models to compute results for")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU num to use for Comet metrics")

    args = vars(parser.parse_args())
    analyzer = Analyzer(**args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["device"])

    for filename in args['bucket']:
        full_filename = f"{args['srclang']}-{args['trglang']}/{filename}"
        assert os.path.exists(full_filename + ".txt"), full_filename + ".txt"
        bucket_ids = set(eval(open(f"{full_filename}.txt").read()))
        print(filename, len(bucket_ids))
        analyzer.analyze(bucket_ids, f"{full_filename}.pickle")
