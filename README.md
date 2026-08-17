# Evaluation-protocol effects in WiFi fingerprint localisation

Evaluation harness for an article currently under review. The manuscript, the
per-run results behind its tables, and the prototype survey data will be added
to this repository upon publication.

A fingerprint survey visits a small number of physical positions — reference
points — and records several measurements at each. Splitting the *measurements*
at random therefore does not split the *positions*, and the resulting accuracy
figure measures re-identification of already-surveyed positions rather than
localisation of new ones. This repository contains the audit that measures the
effect, the three evaluation protocols that avoid it, and every experiment
reported in the article.

## Quick start

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# UJIIndoorLoc, from the original UCI source
curl -L -o uji.zip https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip
unzip uji.zip -d uji          # -> uji/UJIndoorLoc/{trainingData,validationData}.csv

# XJTLUIndoorLoc, from its authors
git clone https://github.com/Carloslee96/Indoor-Localization.git xjtlu

./venv/bin/python code/characterise.py     # protocols A and B, with a kNN reference
./venv/bin/python code/protocol_c.py       # reference-point-disjoint protocol
./venv/bin/python code/density_experiment.py 10   # thinning experiment at 10 m
```

Use the **original UCI files**. Any redistributed copy whose coordinates have been
normalised to [0,1] will silently produce dimensionless errors — see below.

## The audit in one command

```python
import numpy as np
from sklearn.model_selection import train_test_split
from uji_data import load_uji

tr, _ = load_uji()
_, g = np.unique(tr[:, [520, 521, 522]], axis=0, return_inverse=True)   # reference point ids
i_tr, i_te = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42)
leaked = sum(1 for i in i_te if g[i] in set(g[i_tr]))
print(f"{leaked}/{len(i_te)} test samples sit at a position present in training")
# 3988/3988
```

The leak is predictable in advance. If a reference point contributes `c` captures
and a fraction `f` is withheld for testing, it is absent from training only if all
`c` of its captures land in the test fold, with probability `f**c`. For an 80/20
split, `c = 3` already leaks more than 99% of reference points.

## What the code contains

| file | purpose |
|---|---|
| `uji_data.py` | loading, feature construction, targets **in metres**, error metrics |
| `corrected_pipeline.py` | SVR fit/predict in metres, log-space search, four optimisers |
| `region_svr.py` | the region-conditioned localiser (classify building, then regress) |
| `characterise.py` | protocol A vs B, with a kNN reference |
| `protocol_c.py` | reference-point-disjoint protocol |
| `region_protocols.py` | the region-conditioned localiser under all protocols |
| `density_experiment.py` | reference-point thinning against a size-matched control |
| `density_xjtlu.py` | the same upon the second database |
| `xjtlu_protocols.py` | second database, including device transfer |
| `tune_region.py`, `opt_region.py` | hyperparameter selection, RP-disjoint throughout |
| `repeatability.py`, `stats_analysis.py` | dispersion across partitions, paired tests |
| `opt_region2.py`, `optimisers_more.py`, `run_seed.py` | additional seeds for the optimiser comparison |
| `grid_svr.py` | coarse grid locating the useful hyperparameter region |
| `reproduce_paper.py` | reproduces the defective figure of the superseded pipeline |
| `make_figure.py`, `make_pipeline_fig.py`, `make_graphical_abstract.py`, `replot.py` | every figure in the article |

The JSON output of every run reported in the article, and the manuscript itself
with its per-run appendix, will be deposited here upon publication.

## Four protocols

| protocol | what varies | what it measures |
|---|---|---|
| **A** random split | nothing | re-identification — 100% of test RPs appear in training |
| **B** device/time holdout | users, phones, months | deployment generalisation |
| **C** RP-disjoint split | position only | spatial generalisation |
| **D** device transfer | handset only, same RPs | device generalisation, isolated |

Report B and C. Report A only to quantify how much it inflates.

## Pitfalls this harness exists to avoid

1. **Scale the targets for fitting, then invert before scoring.** UTM coordinates
   reach 4.86e6 m, so an unscaled target will not fit; but if the inverse
   transform is omitted the reported error is dimensionless and looks one to three
   orders of magnitude too small.
2. **Never tune on the reported test set.** Hyperparameter selection here uses an
   inner partition that is reference-point-disjoint from the fitting fold and drawn
   entirely from training data.
3. **Do not per-column standardise RSS features.** Most access point columns are
   near-constant at "not detected"; scaling them to unit variance amplifies their
   rare detections and dominates the kernel distance (measured: 17.6 m vs 10.9 m).
4. **Check for boundary-pinned optima.** A hyperparameter that converges onto the
   edge of its search range means the range, not the optimiser, chose the result.
5. **RMSE >= MAE always.** A table violating it indicates a computational error.

## Reproduction

All seeds are fixed. Reproducing the full experimental section takes roughly four
hours on a ten-core workstation, most of it in the final refittings.

## Licence and citation

Please cite the article if you use this harness. The databases remain under the
licences of their original authors.
