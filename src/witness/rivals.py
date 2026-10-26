"""Re-implementations of the two closest prior LLM-assisted IR evaluation methods, for the WITNESS evaluator.

1. Oosterhuis, Jagerman, Qin, Wang, Bendersky, "Reliable Confidence Intervals for Information Retrieval Evaluation
   Using Generative A.I.", KDD 2024 (arXiv 2407.02464), Sec. 4.2, Eqs. (8), (10), (11).  Official code:
   github.com/google-research/google-research @ 57183fd5, high_confidence_ir_eval_using_genai/methods.py::ppi
   (estimator and interval) and utils.py::run_method / datasets.py::sample (human labels = all documents of n
   queries drawn uniformly without replacement).
2. Takehi, Voorhees, Sakai, Soboroff, "LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?" (LARA),
   SIGIR 2025 (arXiv 2411.06877), Algorithm 1.  Official code: github.com/RikiyaT/LARA @ 482cb834,
   src/utils/graded_algo.py (LogisticOrdinalNet, LogisticOrdinalPredictor, ours_graded), the graded version that
   src/test-al-graded.py runs.

Symbols
  pi [n, 3]      LLM relevance distribution of item d over grades {0, 1, 2} (paper 1: P_hat(R=r|d,q); paper 2: pi_i)
  f [n], u [n]   predicted gain E_pi[u(R)] and true gain u(y) of one row block (u0 = 1[y=2], u1 = 1[y>=1])
  G [R, n]       row coefficients (pair contrasts g and per-system discounts), as in witness.core
  U_r(q)         per-query metric of row r: sum_{d in q} G[r, d] u(y_d);  U_hat_r(q) the same with f
  lab_q          the n human-labelled queries; Q the number of queries
  p_hat(y|pi)    LARA calibrator (cumulative-logit ordinal logistic regression on pi)
"""
import numpy as np
import torch
import torch.nn.functional as F


def vote_shares(J):
    """pi [n, 3]: share of the answering judges that gave each grade; an item no judge answered gets the uniform
    distribution (no information).  J [n_judges, n] in {-1 (refused), 0, 1, 2}."""
    ans = J >= 0
    cnt = np.stack([((J == k) & ans).sum(0) for k in range(3)], 1).astype(float)  # [n, 3]
    k = ans.sum(0)
    return np.where(k[:, None] > 0, cnt / np.maximum(k, 1)[:, None], 1.0 / 3.0)


def per_query(G, x, qi, n_q):
    """[R, Q] per-query sums sum_{d in q} G[r, d] x_d."""
    out = np.zeros((G.shape[0], n_q))
    np.add.at(out.T, qi, (G * x[None, :]).T)
    return out


def oosterhuis_ppi(G, f, u, qi, n_q, lab_q):
    """methods.py::ppi for every row of G.  Eq. (8): est = mean_{all q} U_hat(q) - mean_{q in lab_q} (U_hat(q) - U(q));
    Eqs. (10)-(11): var = Var_lab(U_hat - U) / n + Var_all(U_hat) / N, both with ddof = 1 (np.std(ddof=1)^2 in the
    official code).  A pair row is the paired per-query difference U_A(q) - U_B(q), so its PPI interval is the
    method applied to the per-query difference metric (the estimator is linear, so est_AB = est_A - est_B).
    u is read only on the items of the labelled queries.  Returns est [R], var [R]."""
    assert len(lab_q) >= 2 and len(np.unique(lab_q)) == len(lab_q)
    lab = np.isin(qi, lab_q)
    pred = per_query(G, f, qi, n_q)  # [R, Q]
    err = pred[:, lab_q] - per_query(G, np.where(lab, u, 0.0), qi, n_q)[:, lab_q]  # [R, n]
    est = pred.mean(1) - err.mean(1)
    var = err.var(1, ddof=1) / len(lab_q) + pred.var(1, ddof=1) / n_q
    return est, var


class LogisticOrdinalNet(torch.nn.Module):
    """graded_algo.py::LogisticOrdinalNet: one logistic regression on pi per cumulative threshold y >= k."""

    def __init__(self, n_classes):
        super().__init__()
        self.logits = torch.nn.Linear(n_classes, n_classes - 1)

    def forward(self, x):
        return self.logits(x)


def _from_ordinal(cum):
    """graded_algo.py::_from_ordinal, including its final softmax over the class probabilities (kept as in the
    official code; it is monotone, so argmax labels are unaffected, but it compresses the margins)."""
    probs = torch.cat([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1)
    return F.softmax(probs, dim=1)


def _to_ordinal(y, n_classes):
    return (torch.arange(1, n_classes)[None, :] <= y[:, None]).float()  # [b, n_classes - 1], target 1[y >= k]


def _loss(model, xb, yb, transition_w):
    """graded_algo.py::fit_batch loss: BCE(cumulative logits, 1[y >= k]) + transition_w * MSE(p_hat_learned, pi)."""
    logits = model(xb)
    loss = F.binary_cross_entropy_with_logits(logits, _to_ordinal(yb, xb.shape[1]))
    return loss + transition_w * F.mse_loss(_from_ordinal(torch.sigmoid(logits)), xb)


def lara(pi, y, budget, rng, mode, lr, batch, epochs, transition_w, diff_w, lbfgs_iter):
    """LARA (Algorithm 1, one assessor group, LARA(n=1)) on the items of pi [N, 3]; true grades y [N] are read only
    when bought.

    Selection in batches of `batch` (graded_algo.py::ours_graded): uncertainty = margin between the two largest
    p_hat + diff_w * ||p_hat - pi||_1; the `batch` smallest are labelled; then the calibrator is updated.
    mode = "code": exactly what the official graded code executes.  The calibrator takes `epochs` Adam steps (lr) on
      the new batch only; prediction is p_hat = (1 - c) pi + c p_learned with c = min(1, n_samples * conf_rate),
      and n_samples stays 0 because ours_graded calls fit_batch, never add_training_sample, so c = 0: p_hat = pi
      throughout and the remaining items get argmax pi.
    mode = "paper": Algorithm 1 as written (p_hat = pi before any label, then p_hat = the calibrator, "regressing the
      true relevance y on pi" with logistic regression): after every batch the official ordinal model and loss are
      refit on all labels bought so far, full batch, L-BFGS warm-started (the paper fixes neither optimiser nor
      schedule), and p_hat = p_learned.
    Ties (pi takes few distinct values here) are broken uniformly at random instead of by file order.
    Returns y_hat [N] (the human grade where labelled, argmax p_hat elsewhere; Algorithm 1 lines 16-19), labelled [N]."""
    N, n_classes = pi.shape
    assert 0 < budget <= N and mode in ("code", "paper")
    torch.manual_seed(int(rng.integers(2 ** 31)))
    model = LogisticOrdinalNet(n_classes)
    X = torch.tensor(pi, dtype=torch.float32)  # [N, n_classes]
    Y = torch.tensor(y, dtype=torch.long)
    labelled = np.zeros(N, bool)
    if mode == "code":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iter, line_search_fn="strong_wolfe")

    def predict_proba(idx):
        if mode == "code" or not labelled.any():
            return pi[idx]
        with torch.no_grad():
            return _from_ordinal(torch.sigmoid(model(X[idx]))).double().numpy()

    while labelled.sum() < budget:
        cand = np.flatnonzero(~labelled)
        p = predict_proba(cand)  # [n_cand, n_classes]
        ps = np.sort(p, axis=1)[:, ::-1]
        unc = ps[:, 0] - ps[:, 1] + diff_w * np.abs(p - pi[cand]).sum(1)
        n_sel = min(batch, budget - int(labelled.sum()))
        order = rng.permutation(len(cand))
        sel = cand[order[np.argsort(unc[order], kind="stable")[:n_sel]]]
        labelled[sel] = True
        if mode == "code":
            for _ in range(epochs):
                opt.zero_grad()
                loss = _loss(model, X[sel], Y[sel], transition_w)
                loss.backward()
                opt.step()
        else:
            lab = torch.tensor(np.flatnonzero(labelled))

            def closure():
                opt.zero_grad()
                loss = _loss(model, X[lab], Y[lab], transition_w)
                loss.backward()
                return loss

            assert torch.isfinite(opt.step(closure))
    y_hat = np.where(labelled, y, -1)
    rest = np.flatnonzero(~labelled)
    y_hat[rest] = predict_proba(rest).argmax(1)
    return y_hat, labelled
