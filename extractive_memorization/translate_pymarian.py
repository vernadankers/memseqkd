import argparse
import tqdm
from pymarian import Translator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--wmt")
    parser.add_argument("--model_dir")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    print(f"{args.model_dir}/model.iter{args.checkpoint}.npz")
    model = Translator(
        models=f"{args.model_dir}",
        vocabs=[f"{args.wmt}/joint.model.spm", f"{args.wmt}/joint.model.spm"],
        devices=[0], workspace=8000
    )

    source_stream = open(args.source, 'r', encoding="utf-8")

    i = 0
    sources = []
    with open(args.target, 'w', encoding="utf-8") as f:
        for s in tqdm.tqdm(source_stream):
            i += 1
            sources.append(s.strip())
            if len(sources) > 100000:
                out = model.translate(sources, beam_size=1, quiet=True, mini_batch=128, devices=[0])
                for l in out:
                    f.write(l.strip() + "\n")
                    sources = []
