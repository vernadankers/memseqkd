import sys
import tqdm

f1 = sys.argv[1]
f2 = sys.argv[2]

equal = []
with open(f1, encoding="utf-8") as file1, \
     open(f2, encoding="utf-8") as file2:
    for l1, l2 in zip(file1, file2):
        equal.append(l1.strip() == l2.strip())

print(sum(equal), round(100 * sum(equal) / len(equal), 2))