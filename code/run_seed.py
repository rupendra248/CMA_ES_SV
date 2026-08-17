"""Run all four optimisers for ONE seed, so seeds can be executed in parallel.

The runs are fully independent, and libsvm is single-threaded, so the wall-clock
win from one process per seed is close to linear in the number of cores used.
"""
import sys
import corrected_pipeline as cp

seed = int(sys.argv[1])
cp.run(seeds=(seed,), budget=30, tune_n=3000, mode="raw",
       out=f"out_opt_seed{seed}.json")
