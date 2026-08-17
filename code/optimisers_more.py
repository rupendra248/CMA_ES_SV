"""Additional optimiser seeds (3..9) to bring the comparison to 10 seeds.

Seeds 0-2 already exist in out_optimisers.json; this appends the rest to a
separate file which stats_analysis.py merges.
"""
import corrected_pipeline as cp

cp.run(seeds=tuple(range(3, 10)), budget=30, tune_n=3000, mode="raw",
       out="out_optimisers_more.json")
