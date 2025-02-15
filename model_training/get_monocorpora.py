from datasets import load_dataset
import tqdm
import random

# # Covost speech dataset
# ds = load_dataset("fixie-ai/covost2", "fr_en")["train"]
# fr = []
# # for x in tqdm.tqdm(ds):
# #     fr.append(x["sentence"])
# with open("../wmt20/fr-de/covost-L.fr", 'w', encoding="utf-8") as f:
#     for l in tqdm.tqdm(ds):
#         try:
#             f.write(l["sentence"].strip() + "\n")
#         except:
#             print("Help")
#             continue
ds = load_dataset("fixie-ai/covost2", "en_de")["train"]
# with open("../wmt20/en-de/covost-L.en", 'w', encoding="utf-8") as f:
#     for l in tqdm.tqdm(ds):
#         try:
#             f.write(l["sentence"].strip() + "\n")
#         except:
#             print("Help")
#             continue
with open("../wmt20/en-de/covost-L.de", 'w', encoding="utf-8") as f:
    for l in tqdm.tqdm(ds):
        try:
            f.write(l["translation"].strip() + "\n")
        except:
            print("Help")
            continue

# # Poetry dataset
# ds = load_dataset("linhd-postdata/pulpo")["train"]
# en, de = [], []
# for x in tqdm.tqdm(ds):
#     if x["lang"] == "english":
#         en.append(x["text"])
#     elif x["lang"] == "german":
#         de.append(x["text"])
# with open("../wmt20/en-de/pulpo-L.en", 'w', encoding="utf-8") as f:
#     for l in tqdm.tqdm(random.sample(en, 1000000)):
#         f.write(l.strip() + "\n")
# with open("../wmt20/en-de/pulpo-L.de", 'w', encoding="utf-8") as f:
#     for l in tqdm.tqdm(random.sample(de, 1000000)):
#         f.write(l.strip() + "\n")