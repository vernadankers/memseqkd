### Visualization and subgroup analysis

Subgroup analyses
- We assume all language pairs have translations for the wmt20 corpora at this point (available under `memseqkd/model_zoo/<langpair>/kd.beam1.<trglang>`).
- Then create "buckets" with example ids using `bash create_buckets.sh`.
- Run analyses using `bash analyze_buckets.sh <langpair> <srclang> <trglang> <bucket_name>`, with bucket names:
    - `low_confidence`, `high_confidence`;
    - `comet_X-Y` with X-Y indicating buckets in 0.2 increments;
    - `cm_X-Y` with X-Y in \[0-0.2, 0.2-0.3, 0.3-0.4, 0.4-1\], `cm_bottomleft`, `cm_topright`;
    - `random`.
- Create visualizations with `subgroup_analyses.ipynb`.

For the remaining visualisations of results in the main paper and appendix C, D, use `visualize_results.ipynb`.