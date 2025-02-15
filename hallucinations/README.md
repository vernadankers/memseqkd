### Hallucinations

The code has been set up to compute hallucinations for the base setup discussed in the main paper (T, S, B per language pair).
- To run all languages at once, simply modify `hallucinations.sh` to reflect your local slurm setup, and run `bash submit_hallucinations.sh`.
- To run hallucinations using one language pair, run:
  ```python hallucinations.py --wmt_path <wmt_basedir> --model_zoo_path <models_basedir> --langpair <langpair> --srclang <srclang> --trglang <trglang> --source <filename> --target <filename> --models <list> --setup```
  The `langpair` is one of en-de, pl-en, fr-de, and the individual source and target languages reflect the right direction.
  Indicate source and target filenames without their base dir and language extension, since they will be composed based on the other arguments.
  `models` is a list of models, again without their base dir.

The script store pickled files containing the results, and indices of the hallucinated examples, in this directory under a `<srclang>-<trglang>` folder.
