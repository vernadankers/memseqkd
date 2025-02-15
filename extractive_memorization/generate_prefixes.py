import tqdm
import sys

def generate_prefixes(input):
    input, target = input
    tokens = input.split()
    return [(" ".join(tokens[0:i]).strip(), target) for i in range(1, len(tokens)-1)]

def generate_prefixes_truncated(input):
    tokens = input.split()
    start = len(tokens)//3
    return [" ".join(tokens[0:i]).strip() for i in range(start + 1, len(tokens)-1)]


f1 = open(sys.argv[1], "r")
lines = []
for line in f1:
    target, index, source = line.strip().split("\t")
    if source != target and len(source.split()) >= 4:
        lines.append((source, target))

prefixes = []
for line in tqdm.tqdm(lines):
    line_prefixes = generate_prefixes(line)
    line_prefixes = [ "Prefix \t" + x + "\t" + y for x, y in line_prefixes ]
    prefixes += line_prefixes
    prefixes += [ "Final \t" + line[0] + "\t" + line[1] ]

for p in prefixes:
    print(p)
