"""Assert that two result JSON files hold the same structure and the same values.

Floats are compared to 1e-9 relative tolerance (the same computation on another BLAS can differ in the last bits);
keys, list lengths, strings, integers and booleans must be identical.
usage: python scripts/check/compare_json.py <a.json> <b.json>
"""
import json
import math
import sys


def check(x, y, path):
    assert type(x) is type(y) or (isinstance(x, (int, float)) and isinstance(y, (int, float))), (path, x, y)
    if isinstance(x, dict):
        assert set(x) == set(y), (path, sorted(set(x) ^ set(y)))
        return sum(check(x[k], y[k], f"{path}/{k}") for k in x)
    if isinstance(x, list):
        assert len(x) == len(y), (path, len(x), len(y))
        return sum(check(u, v, f"{path}[{i}]") for i, (u, v) in enumerate(zip(x, y)))
    if isinstance(x, float) or isinstance(y, float):
        assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(x) and math.isnan(y)), (path, x, y)
        return 1
    assert x == y, (path, x, y)
    return 1


n = check(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), "")
print(f"{sys.argv[1]} matches {sys.argv[2]}: {n} values")
