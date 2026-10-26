"""Checks of the hidden-engine propensity DP (src/witness/mech.py) and the ${VAR} config loader (src/witness/config.py).
usage: PYTHONPATH=src python3 -m pytest -q tests
"""
import numpy as np
import pytest

from witness import config as C
from witness import mech as M


def test_hidden_propensity_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n_q, k_h = 3, 5
    qi = np.repeat(np.arange(n_q), [4, 7, 6])  # query 0 has fewer candidates than k_h
    n = len(qi)
    y = rng.integers(0, 3, n)
    slots, rank_h = M.engine_slots(qi, n_q, rng.random(n), k_h)
    assert (slots[0, 4:] == -1).all() and ((rank_h > 0).sum() == 4 + k_h + k_h)
    p = M.mark_probs(y, rank_h, {"p1": 0.5, "p2": 0.8, "p_err": 0.2, "decay": 0.85})
    n_mc = 200_000
    for stop_after in (1, 2):
        e = M.hidden_propensity(slots, p, stop_after)
        assert (e[rank_h == 0] == 0).all()
        hits = np.zeros(n)
        for _ in range(n_mc // 10_000):
            draws = [M.hidden_draw(rng, slots, p, stop_after, n) for _ in range(10_000)]
            hits += np.sum(draws, axis=0)
        e_mc = hits / n_mc
        se = np.sqrt(np.maximum(e * (1 - e), 1e-12) / n_mc)
        assert (np.abs(e_mc - e) <= 4 * se).all(), (stop_after, e, e_mc)
        # at most stop_after marks per query, so the propensities of a query sum to at most stop_after
        per_q = np.bincount(qi, weights=e, minlength=n_q)
        assert (per_q <= stop_after + 1e-12).all()


def test_hidden_propensity_exact_small_case():
    """One query, ranks 1..3 with mark probs p: e = p1, p2 (1 - p1), p3 (1 - p1)(1 - p2) for stop_after = 1."""
    p = np.array([0.3, 0.6, 0.5])
    slots = np.array([[0, 1, 2]])
    e1 = M.hidden_propensity(slots, p, 1)
    assert np.allclose(e1, [0.3, 0.6 * 0.7, 0.5 * 0.7 * 0.4], rtol=0, atol=1e-15)
    e2 = M.hidden_propensity(slots, p, 2)
    assert np.allclose(e2, [0.3, 0.6, 0.5 * (1 - 0.3 * 0.6)], rtol=0, atol=1e-15)


def test_load_config_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("WITNESS_ROOT", "/data/w")
    monkeypatch.setenv("RUN_TAG", "r1")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "out: ${WITNESS_ROOT}/out\n"
        "nested:\n"
        "  paths: ['${WITNESS_ROOT}/a', plain, '${WITNESS_ROOT}/${RUN_TAG}']\n"
        "  deeper: {x: '${RUN_TAG}_x', n: 3, f: 0.5, flag: true}\n"
        "lower: ${lower_case}\n",
        encoding="utf8",
    )
    cfg = C.load_config(path)
    assert cfg == {
        "out": "/data/w/out",
        "nested": {"paths": ["/data/w/a", "plain", "/data/w/r1"], "deeper": {"x": "r1_x", "n": 3, "f": 0.5,
                                                                             "flag": True}},
        "lower": "${lower_case}",  # the pattern names upper-case variables only
    }


def test_load_config_unset_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("WITNESS_TEST_UNSET", raising=False)
    path = tmp_path / "cfg.yaml"
    path.write_text("a:\n  - {b: '${WITNESS_TEST_UNSET}/x'}\n", encoding="utf8")
    with pytest.raises(KeyError):
        C.load_config(path)
