"""Post-hoc checks added after a review of the paper (not pre-registered).

    coherence  P6: the low-end test counted by mission, which holds however the
               encounters of one mission depend on each other, and the crossings
               split by the chaser's object type and by ESA's dilution flag
    cv         P4, P7 and P8 in the folds and seeds of the registered runs:
               - the corrected repeated k-fold t-test for every contrast, in place
                 of a bootstrap over fixed out-of-fold predictions
               - P8 against an isotonic calibration of the latest risk
               - false-alarm thresholds set inside the training folds
               - the benchmark features with the chaser's covariance rotated into
                 the target's frame (A0R)

The sealed test stays spent: nothing here trains on the training file and scores
the official test as a held-out set. The registered functions are not changed.

    .venv/bin/python -m backend.app.models.lab.review coherence --out reports/paper
    .venv/bin/python -m backend.app.models.lab.review cv --out reports/paper
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import data, metric, physics, protocol
from backend.app.models.lab import round4 as R4
from backend.app.models.lab.data import HIGH_RISK, Event
from backend.app.models.lab.paper import DEFAULTS, _write


# --- statistics ------------------------------------------------------------------

def corrected_t(d, folds: int) -> dict:
    """Corrected repeated k-fold t-test (Nadeau and Bengio 2003; Bouckaert and
    Frank 2004) on per-fold differences ``d`` from repeats of ``folds``-fold CV:
    the variance of the mean is s^2 (1/n + n_test/n_train) with
    n_test/n_train = 1/(folds - 1), because the folds of one repeat share
    training data."""
    from scipy import stats

    d = np.asarray([x for x in d if np.isfinite(x)], float)
    n = len(d)
    if n < 2:
        return {"n_folds": n, "mean": float(d.mean()) if n else None, "ci95": [None, None], "p": None}
    mean, s2 = float(d.mean()), float(d.var(ddof=1))
    se = math.sqrt(s2 * (1.0 / n + 1.0 / (folds - 1)))
    q = float(stats.t.ppf(0.975, n - 1))
    if se == 0:
        return {"n_folds": n, "mean": mean, "ci95": [mean, mean], "p": 0.0 if mean else 1.0}
    t = mean / se
    return {"n_folds": n, "mean": mean, "ci95": [mean - q * se, mean + q * se],
            "t": t, "p": float(2.0 * stats.t.sf(abs(t), n - 1))}


def mission_bound(missions, bound, cross) -> dict:
    """The P6 low-end count by mission. A mission's chance of any crossing is at
    most the sum of its encounters' bounds (a union bound, valid whatever the
    dependence within the mission), so the expected number of missions with a
    crossing is at most the sum over missions of min(1, that sum). Chernoff then
    needs independence across missions only."""
    missions = np.asarray(missions)
    bound, cross = np.asarray(bound, float), np.asarray(cross, bool)
    per = {}
    for m in np.unique(missions):
        i = missions == m
        per[m] = (min(1.0, float(bound[i].sum())), bool(cross[i].any()), int(cross[i].sum()))
    mean = float(sum(v[0] for v in per.values()))
    k = int(sum(v[1] for v in per.values()))
    return {"n_missions": len(per), "missions_with_crossing": k, "coherent_max_missions": mean,
            "crossings_per_crossing_mission": sorted((v[2] for v in per.values() if v[1]), reverse=True),
            "p_chernoff_missions": R4.chernoff_p(k, mean)}


def isotonic_baseline(latest_tr, y_tr):
    """P(final >= -6) from the latest risk alone, isotonic (any monotone curve)."""
    from sklearn.isotonic import IsotonicRegression

    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    m.fit(np.asarray(latest_tr, float), (np.asarray(y_tr) >= HIGH_RISK).astype(float))
    return lambda lat: m.predict(np.asarray(lat, float))


def keep_threshold(scores_true_calls, keep: float = 0.95) -> float:
    """The highest threshold that keeps ``keep`` of the true calls at or above it."""
    s = np.sort(np.asarray(scores_true_calls, float))[::-1]
    return float(s[int(math.ceil(keep * len(s))) - 1])


# --- the chaser's covariance in the target's frame ---------------------------------

@contextmanager
def rotated_physics():
    """While active, the lab's physics features add the chaser's covariance after
    rotating it into the target's RTN frame (round4.chaser_to_target), where that
    rotation is defined; elsewhere they keep the unrotated sum."""
    orig = physics.covariance_rtn
    counts = {"rotated": 0, "unrotated": 0}

    def rot(row, prefix):
        c = orig(row, prefix)
        if prefix != "c" or c is None:
            return c
        m = R4.chaser_to_target(row)
        if m is None:
            counts["unrotated"] += 1
            return c
        counts["rotated"] += 1
        return m @ c @ m.T

    physics.covariance_rtn = rot
    try:
        yield counts
    finally:
        physics.covariance_rtn = orig


# --- P6 by mission, object type and dilution -------------------------------------

def coherence_review(test_events: list[Event], late: dict) -> dict:
    ids = [int(e.event_id.split(":")[1]) for e in test_events]
    r = np.array([e.latest_risk for e in test_events])
    y = np.array([e.target for e in test_events])
    matched = np.array([i in late for i in ids])
    cross = np.array([max([y_i] + [float(c["risk"]) for c in late.get(i, [])]) >= HIGH_RISK
                      for i, y_i in zip(ids, y)])
    bound = R4.coherent_bound(r)
    band = (r < HIGH_RISK) & matched & (r > -29.9) & (r < -8.0)
    last = [e.inputs[-1] for e in test_events]
    mission = np.array([str(row.get("mission_id")) for row in last])
    ctype = np.array([str(row.get("c_object_type") or "UNKNOWN").upper() for row in last])
    scaling = np.array([R4._num(row, "max_risk_scaling") for row in last])

    def group(m):
        k, mean = int(cross[m].sum()), float(bound[m].sum())
        return {"n": int(m.sum()), "observed": k, "coherent_max": mean,
                "ratio": (k / mean) if mean > 0 else None, "p_chernoff": R4.chernoff_p(k, mean),
                **mission_bound(mission[m], bound[m], cross[m])}

    primary = group(band)
    reg = json.loads(Path("reports/paper/coherence.json").read_text())["a_low_end"]["primary_any_later_cdm"]
    primary["reproduces_registered"] = (primary["n"] == reg["n"] and primary["observed"] == reg["observed"]
                                        and abs(primary["coherent_max"] - reg["coherent_max"]) < 1e-9)
    payload = ctype == "PAYLOAD"
    dilution = np.isfinite(scaling) & (scaling < 1.0)
    robust = np.isfinite(scaling) & (scaling >= 1.0)
    by_type = {t: group(band & (ctype == t)) for t in sorted(set(ctype[band]))}
    split = {"chaser_payload": group(band & payload), "chaser_not_payload": group(band & ~payload),
             "dilution_region": group(band & dilution), "outside_dilution": group(band & robust),
             "scaling_missing": group(band & ~np.isfinite(scaling))}
    crossing = band & cross
    table = {f"{'payload' if p else 'not_payload'}|{'dilution' if d else 'outside'}":
             int(np.sum(crossing & (payload == p) & (dilution == d))) for p in (True, False) for d in (True, False)}
    return {"analysis": "Post hoc (review): P6's low-end test by mission, chaser object type and "
                        "ESA's dilution flag (max_risk_scaling < 1: shrinking the covariance raises Pc)",
            "band": "latest log10 Pc between -29.9 and -8, matched to a raw sequence",
            "primary": primary, "by_chaser_type": by_type, "splits": split,
            "crossings_by_payload_and_dilution": table}


# --- P4, P7 and P8 with corrected intervals ----------------------------------------

def cv_review(events: list[Event], repeats: int = 5, folds: int = 5, seed: int = 42,
              inner: int = 3, log=print) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from backend.app.models.lab.paper import feature_sets

    b = protocol.Bundle(events)
    with rotated_physics() as rot_counts:
        b_rot = protocol.Bundle(events)
    cols = b.columns["benchmark"]
    assert b_rot.columns["benchmark"] == cols
    xb, xbr = b.X["benchmark"], b_rot.X["benchmark"]
    xpub, pub_cols = R4.public_matrix(xb, cols)
    sets = feature_sets(cols)
    sub = {k: (xb[:, np.array([b.colindex[c] for c in sets[name]], int)], sets[name])
           for k, name in (("A1", "A1_without_covariance"), ("A2", "A2_serving_plus_covariance"))}
    variants = {"A0": (xb, cols), "A0R": (xbr, cols), "PUB": (xpub, pub_cols),
                "LIVE": (b.X["serving"], b.columns["serving"]), **sub}
    nested = ("A0", "A0R", "PUB")       # false-alarm thresholds from inner folds
    n = len(events)
    oof = {k: np.full((repeats, n), np.nan) for k in list(variants) + ["calibrated", "isotonic"]}
    el, cand = b.eligible, b.cand
    calls = el & ~cand
    stay = b.y >= HIGH_RISK
    dom = cand & el
    per = {"logloss": {k: [] for k in ("A0", "A0R", "PUB", "calibrated", "isotonic")},
           "fa_removed": {k: [] for k in nested + ("latest",)},
           "kept": {k: [] for k in nested + ("latest",)},
           "pr_auc": {k: [] for k in list(variants) + ["latest"]},
           "roc_auc": {k: [] for k in list(variants) + ["latest"]}}
    totals = {k: {"fa_below": 0, "fa": 0, "kept": 0, "stay": 0} for k in nested + ("latest",)}

    def hurdle(X, vc, idx, s):
        ctx = C.Context(X[idx], b.y[idx], b.latest[idx], b.strata[idx], s, b.eligible[idx],
                        {"colindex": {c: i for i, c in enumerate(vc)}, "colnames": vc,
                         "y_aux": b.y_aux[idx]})
        h = C.Hurdle("hurdle_review")
        h.fit(ctx)
        return h

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), b.strata)):
            t0 = time.time()
            tr = tr[b.trainable[tr]]
            s = seed + 1000 * r + f
            for k, (X, vc) in variants.items():
                oof[k][r, te] = hurdle(X, vc, tr, s).prob_high(X[te], b.latest[te])
            oof["calibrated"][r, te] = R4.calibrated_baseline(b.latest[tr], b.y[tr])(b.latest[te])
            oof["isotonic"][r, te] = isotonic_baseline(b.latest[tr], b.y[tr])(b.latest[te])

            # thresholds from data the held-out fold never saw
            thr = {"latest": keep_threshold(b.latest[tr][calls[tr] & stay[tr]])}
            ikf = StratifiedKFold(n_splits=inner, shuffle=True, random_state=s)
            for k in nested:
                X, vc = variants[k]
                sc = np.full(n, np.nan)
                for itr, ite in ikf.split(np.zeros(len(tr)), b.strata[tr]):
                    a, z = tr[itr], tr[ite]
                    z = z[calls[z]]
                    if len(z):
                        sc[z] = hurdle(X, vc, a, s + 7).prob_high(X[z], b.latest[z])
                tc = tr[calls[tr] & stay[tr]]
                thr[k] = keep_threshold(sc[tc][np.isfinite(sc[tc])])

            te_el, te_calls, te_dom = te[el[te]], te[calls[te]], te[dom[te]]
            y_hr = (b.y[te_el] >= HIGH_RISK).astype(float)
            for k in per["logloss"]:
                per["logloss"][k].append(R4._logloss(y_hr, oof[k][r, te_el]))
            score = {**{k: oof[k][r] for k in nested}, "latest": b.latest}
            st = stay[te_calls]
            for k in nested + ("latest",):
                v = score[k][te_calls]
                below_fa, kept = int(np.sum(v[~st] < thr[k])), int(np.sum(v[st] >= thr[k]))
                per["fa_removed"][k].append(below_fa / max(1, int((~st).sum())))
                per["kept"][k].append(kept / max(1, int(st.sum())))
                for key, val in (("fa_below", below_fa), ("fa", int((~st).sum())),
                                 ("kept", kept), ("stay", int(st.sum()))):
                    totals[k][key] += val
            y_ew = b.esc[te_dom]
            if 0 < y_ew.sum() < len(y_ew):
                for k in per["pr_auc"]:
                    v = b.latest[te_dom] if k == "latest" else oof[k][r, te_dom]
                    per["pr_auc"][k].append(metric.average_precision(y_ew, v))
                    per["roc_auc"][k].append(metric.roc_auc(y_ew, v))
            else:
                for k in per["pr_auc"]:
                    per["pr_auc"][k].append(float("nan"))
                    per["roc_auc"][k].append(float("nan"))
            log(f"  repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")

    # the registered runs, reproduced from the same folds and seeds
    pub = json.loads(Path("reports/paper/public.json").read_text())
    ewp = json.loads(Path("reports/paper/ew_posthoc.json").read_text())
    y_el = (b.y[el] >= HIGH_RISK).astype(float)
    pooled_ll = {k: float(np.mean([R4._logloss(y_el, oof[k][r, el]) for r in range(repeats)]))
                 for k in ("A0", "PUB", "calibrated")}
    reproduced = {f"p8_logloss_{k}": abs(pooled_ll[k] - pub["p8_forecast"][k]["logloss"]) < 1e-12 for k in pooled_ll}
    y_ew_all = b.esc[dom]
    for k in ("A0", "PUB", "LIVE", "A1", "A2"):
        pr = float(np.mean([metric.average_precision(y_ew_all, oof[k][r, dom]) for r in range(repeats)]))
        reproduced[f"ew_pr_auc_{k}"] = abs(pr - ewp["early_warning"]["pooled_pr_auc"][k]) < 1e-12

    def contrast(metric_name, a, z):
        return corrected_t(np.array(per[metric_name][a]) - np.array(per[metric_name][z]), folds)

    def level(metric_name, k):
        return corrected_t(np.array(per[metric_name][k]), folds)

    forecast = {"logloss_mean_per_fold": {k: float(np.mean(v)) for k, v in per["logloss"].items()},
                "A0_minus_calibrated": contrast("logloss", "A0", "calibrated"),
                "A0_minus_isotonic": contrast("logloss", "A0", "isotonic"),
                "PUB_minus_isotonic": contrast("logloss", "PUB", "isotonic"),
                "isotonic_minus_calibrated": contrast("logloss", "isotonic", "calibrated"),
                "A0R_minus_A0": contrast("logloss", "A0R", "A0")}
    for k in ("A0_minus_calibrated", "A0_minus_isotonic", "PUB_minus_isotonic"):
        base = forecast["logloss_mean_per_fold"][k.split("_minus_")[1]]
        forecast[k]["relative_pct"] = 100.0 * forecast[k]["mean"] / base
    false_alarms = {"thresholds": "set to keep 95% of the training folds' true calls (nested CV for models)",
                    "fa_removed": {k: level("fa_removed", k) for k in per["fa_removed"]},
                    "true_calls_kept": {k: level("kept", k) for k in per["kept"]},
                    "pooled_totals": {k: {**v, "fa_removed": v["fa_below"] / max(1, v["fa"]),
                                          "kept": v["kept"] / max(1, v["stay"])} for k, v in totals.items()},
                    "A0_minus_latest": contrast("fa_removed", "A0", "latest"),
                    "PUB_minus_latest": contrast("fa_removed", "PUB", "latest"),
                    "A0R_minus_A0": contrast("fa_removed", "A0R", "A0")}
    early = {"n_candidates": int(dom.sum()), "n_escalations": int(b.esc[dom].sum()),
             "pr_auc_mean_per_fold": {k: float(np.nanmean(v)) for k, v in per["pr_auc"].items()},
             "roc_auc_mean_per_fold": {k: float(np.nanmean(v)) for k, v in per["roc_auc"].items()},
             "pr_vs_latest": {k: contrast("pr_auc", k, "latest") for k in variants},
             "roc_vs_latest": {k: contrast("roc_auc", k, "latest") for k in variants},
             "A0R_minus_A0": {"pr": contrast("pr_auc", "A0R", "A0"), "roc": contrast("roc_auc", "A0R", "A0")}}
    return {"analysis": "Post hoc (review): corrected repeated k-fold intervals, an isotonic latest-risk "
                        "baseline, false-alarm thresholds set inside the training folds, and rotated "
                        "covariance features (pooled CV; the folds and seeds of the registered runs)",
            "repeats": repeats, "folds": folds, "inner_folds": inner, "seed": seed, "n_events": n,
            "n_eligible": int(el.sum()), "n_high_risk_eligible": int(y_el.sum()),
            "class_mix": "pooling the training and test events restores ESA's full population of "
                         "eligible events, so these are at the operational class mix",
            "rotation_coverage": rot_counts, "reproduces_registered": reproduced,
            "p8_forecast": forecast, "p8_false_alarms": false_alarms, "early_warning": early}


def main(argv=None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description="VIGIL Lab: post-hoc checks from the paper's review")
    p.add_argument("analysis", choices=("coherence", "cv"))
    p.add_argument("--train", default=DEFAULTS["train"])
    p.add_argument("--test", default=DEFAULTS["test"])
    p.add_argument("--test-targets", default=DEFAULTS["test_targets"])
    p.add_argument("--raw", default=R4.RAW)
    p.add_argument("--out", default="reports/paper")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    out, t0 = Path(a.out), time.time()
    train, test = data.load(a.train, a.test, a.test_targets)
    if a.analysis == "coherence":
        tdf, tgt = pd.read_csv(a.test), pd.read_csv(a.test_targets)
        late = R4.match_raw(tdf, tgt, pd.read_csv(a.raw))
        print(f"loaded {len(test)} test events; {len(late)} matched to a unique raw sequence")
        _write(out, "review_coherence", coherence_review(test, late), t0)
        return 0
    print(f"loaded {len(train)} training events, {len(test)} test events")
    _write(out, "review_cv", cv_review(train + test, seed=a.seed), t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
