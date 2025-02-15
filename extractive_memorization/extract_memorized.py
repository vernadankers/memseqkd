import tqdm
import argparse
from filters import basic_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--trg", type=str)
    parser.add_argument("--hyp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--srclang", type=str)
    parser.add_argument("--trglang", type=str)
    args = parser.parse_args()

    f1 = open(args.trg, "r", encoding="utf-8")
    f2 = open(args.hyp, "r", encoding="utf-8")
    f3 = open(args.src, "r", encoding="utf-8")

    with open(args.out, 'w', encoding="utf-8") as f:
        index = 1
        for t, o, s in tqdm.tqdm(zip(f1, f2, f3)):
            t = t.strip()
            o = o.strip()
            s = s.strip()
            if t == o and basic_filter(f"{s}\t{t}", args.srclang, args.trglang):
                f.write(t + "\t" + str(index) + "\t" + s + "\n")
            index += 1
