import tqdm, lzma, random
import sys

one_m = set()
lang = sys.argv[1]

with lzma.open(f"{lang}.xz", mode='rt') as f:
    for i, l in tqdm.tqdm(enumerate(f)):
        if random.random() < 0.0101:
            one_m.add(l)
        if i == 100000000:
            break

with open(f"commoncrawl.{lang}", 'w', encoding="utf-8") as f:
    for i, l in enumerate(list(one_m)):
        f.write(l)
        if i == 1000000:
            break
