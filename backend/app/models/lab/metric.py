"""Official Kelvins score, its baselines, and VIGIL-E ranking metrics.

Official score (lower is better):   L = MSE_HR / F2
    F2      - F-beta (beta = 2) of the high-risk classification over all
              scored events, where an event is high-risk iff its (true or
              predicted) final log10 risk >= -6.
    MSE_HR  - mean squared error of the predicted final risk over the
              events whose TRUE final risk is >= -6.

Latest-risk prediction (LRP) baseline, as defined by the organisers:
    r_hat = latest input risk if >= -6, else -6.001.

``clip_low`` applies the same "-6.001 for anything predicted below the line"
convention to a model's output. It never changes the classification and can
only reduce MSE_HR (a true high-risk event predicted at -6.001 is always at
least as close as one predicted lower) - it is the optimal low prediction
under this metric (Uriot et al. 2022) - so it is applied to every candidate.
"""
from __future__ import annotations

import numpy as np

from backend.app.models.lab.data import HIGH_RISK, LOW_PREDICTION


def clip_low(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.where(pred < HIGH_RISK, LOW_PREDICTION, pred)


def lrp(latest: np.ndarray) -> np.ndarray:
    latest = np.asarray(latest, dtype=float)
    return np.where(latest >= HIGH_RISK, latest, LOW_PREDICTION)


def f_beta(truth_pos: np.ndarray, pred_pos: np.ndarray, beta: float = 2.0) -> float:
    truth_pos = np.asarray(truth_pos, bool)
    pred_pos = np.asarray(pred_pos, bool)
    tp = float(np.sum(truth_pos & pred_pos))
    fp = float(np.sum(~truth_pos & pred_pos))
    fn = float(np.sum(truth_pos & ~pred_pos))
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom > 0 else 0.0


def official(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """{'L', 'mse_hr', 'f2', 'n', 'n_hr', 'precision', 'recall'}

    ``weights`` (per event) gives the importance-weighted score, e.g. at
    another population's class mix under label shift: weighted TP/FP/FN in
    F2 and a weighted mean in MSE_HR. Without weights the score is exactly
    the official one."""
    if weights is not None:
        return _official_weighted(pred, truth, np.asarray(weights, dtype=float))
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    tpos = truth >= HIGH_RISK
    ppos = pred >= HIGH_RISK
    f2 = f_beta(tpos, ppos, 2.0)
    mse_hr = float(np.mean((pred[tpos] - truth[tpos]) ** 2)) if tpos.any() else float("nan")
    tp = float(np.sum(tpos & ppos))
    prec = tp / max(1.0, float(ppos.sum()))
    rec = tp / max(1.0, float(tpos.sum()))
    L = mse_hr / f2 if f2 > 0 else float("inf")
    return {"L": L, "mse_hr": mse_hr, "f2": f2, "n": int(len(truth)),
            "n_hr": int(tpos.sum()), "precision": prec, "recall": rec}


def _official_weighted(pred, truth, w) -> dict:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    tpos = truth >= HIGH_RISK
    ppos = pred >= HIGH_RISK
    tp = float(w[tpos & ppos].sum())
    fp = float(w[~tpos & ppos].sum())
    fn = float(w[tpos & ~ppos].sum())
    denom = 5.0 * tp + 4.0 * fn + fp
    f2 = 5.0 * tp / denom if denom > 0 else 0.0
    wh = float(w[tpos].sum())
    mse_hr = float(np.sum(w[tpos] * (pred[tpos] - truth[tpos]) ** 2) / wh) if wh > 0 else float("nan")
    L = mse_hr / f2 if f2 > 0 else float("inf")
    return {"L": L, "mse_hr": mse_hr, "f2": f2, "n": int(len(truth)), "n_hr": int(tpos.sum()),
            "precision": tp / max(float(w[ppos].sum()), 1e-12),
            "recall": tp / max(wh, 1e-12)}


def label_shift_weights(is_hr: np.ndarray, target_prevalence: float) -> np.ndarray:
    """Importance weights that move a sample's high-risk prevalence to
    ``target_prevalence`` (label shift: P(x | class) unchanged)."""
    is_hr = np.asarray(is_hr, bool)
    p = float(is_hr.mean()) if len(is_hr) else 0.0
    if not 0.0 < p < 1.0:
        return np.ones(len(is_hr))
    return np.where(is_hr, target_prevalence / p, (1.0 - target_prevalence) / (1.0 - p))


# --- VIGIL-E: ranking of rare escalations ------------------------------------

def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision, tie-aware (sklearn: one step per distinct
    score, so the order of tied events never matters). NaN without positives."""
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(score, float)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(score, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float = 0.05) -> float:
    """Expected share of positives inside the top ``budget`` fraction of
    events, with ties at the cut broken at random (in expectation), so the
    result never depends on the order of tied events."""
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(budget * len(y))))
    thr = np.sort(s)[::-1][k - 1]
    above = s > thr
    tied = s == thr
    need = k - int(above.sum())
    frac = need / max(1, int(tied.sum()))
    return float((y[above].sum() + frac * y[tied].sum()) / y.sum())


def partial_auc(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.1) -> float:
    """Standardised partial ROC AUC over FPR in [0, max_fpr] (McClish),
    0.5 = random, 1 = perfect - focuses on the top of the ranking."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, score, max_fpr=max_fpr))


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(y, float)) ** 2))


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of logit(prob): slope 1 / intercept 0 = calibrated."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def early_warning(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None,
                  budget: float = 0.05) -> dict:
    out = {
        "n": int(len(y)), "n_pos": int(np.sum(y)),
        "pr_auc": average_precision(y, score),
        "roc_auc": roc_auc(y, score),
        "pauc_fpr10": partial_auc(y, score, 0.1),
        "recall_at_budget": recall_at_budget(y, score, budget),
        "budget": budget,
    }
    if prob is not None:
        out["brier"] = brier(y, prob)
        s, i = calibration_slope_intercept(y, prob)
        out["cal_slope"], out["cal_intercept"] = s, i
    return out
