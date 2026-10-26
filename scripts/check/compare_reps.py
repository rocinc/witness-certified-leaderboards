"""Compare the rows of a partial evaluator run with the same repetitions of the full released run.

Every repetition of scripts/eval/evaluate.py is seeded by (seed, rep), so the rows of repetitions lo..hi-1 of a
partial run must equal those rows of the full table.  Numeric columns are compared to 1e-9 relative tolerance
(the same code on another BLAS can differ in the last bits); every other column must be identical.
usage: python scripts/check/compare_reps.py <partial_raw.csv> <full_raw.csv>
"""
import sys

import numpy as np
import pandas as pd

part = pd.read_csv(sys.argv[1])
full = pd.read_csv(sys.argv[2])
assert list(part.columns) == list(full.columns), (list(part.columns), list(full.columns))
reps = sorted(part.rep.unique())
ref = full[full.rep.isin(reps)].reset_index(drop=True)
part = part.reset_index(drop=True)
assert len(part) == len(ref), (len(part), len(ref))
key = ["rep", "method", "budget"]
assert (part[key] == ref[key]).all().all()
num = [c for c in part.columns if c not in key and np.issubdtype(part[c].dtype, np.number)]
other = [c for c in part.columns if c not in key and c not in num]
assert (part[other] == ref[other]).all().all(), other
a, b = part[num].to_numpy(float), ref[num].to_numpy(float)
same = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
assert same.all(), part.loc[~same.all(1), key].to_string()
print(f"{sys.argv[1]}: {len(part)} rows (repetitions {reps}) match {sys.argv[2]}; "
      f"max |diff| {np.nanmax(np.abs(a - b)):.2e} over {len(num)} numeric columns")
