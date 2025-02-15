import sys


f1 = open(sys.argv[1], "r")
f2 = open(sys.argv[2], "r")
source, translation = [], []

org_seqs = []
temp_src = []
temp_trans = []

for prefix, trans in zip(f1, f2):
    label, src, trg = prefix.split("\t")
    temp_src.append(src.strip())
    temp_trans.append(trans.strip())
    if label.strip() != "Prefix":
        # Append and Reset
        org_seqs.append((src, trg))
        source.append(temp_src)
        translation.append(temp_trans)
        temp_src, temp_trans = [], []

prefix_vals = []
for s, h in zip(source, translation):
    memorized = h[-1]
    final = s[-1]
    for x, y in zip(s, h):
        if y.strip() == memorized.strip():
            x = x.split()
            final = final.split()
            prefix_vals.append(len(x)/len(final))
            break

assert len(prefix_vals) == len(org_seqs)
for x, (s, t) in zip(prefix_vals, org_seqs):
    print(f"{s}\t{t.strip()}\t{x}")
