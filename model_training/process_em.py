import sys
import numpy as np
print(f"EM 0.75: {100*np.mean([float(l.split()[-1]) <= 0.75 for l in open(sys.argv[1]).readlines()]):.2f}")

# srclang = "en"
# trglang = "de"
# for l in open(
#     f"../model_zoo/{srclang}-{trglang}/student_b=1_seed=1111/em_corpus/extractively_memorised.tsv").readlines():
#     if float(l.split('\t')[-1]) < 0.75:
#         print(l)
#         #break
# corpus = set([
#     tuple(l.split('\t')[:2]) for l in open(
#         f"../model_zoo/{srclang}-{trglang}/student_b=1_seed=1111/em_corpus/extractively_memorised.tsv").readlines()
#     if float(l.split('\t')[-1]) < 0.75])
# teacher = set([
#     tuple(l.split('\t')[:2]) for l in open(
#         f"../model_zoo/{srclang}-{trglang}/student_b=1_seed=1111/em_teacher/extractively_memorised.tsv").readlines()
#     if float(l.split('\t')[-1]) < 0.75])
# print(len(corpus))
# print(len(teacher))
# print()