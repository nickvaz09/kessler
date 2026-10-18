"""Post-hoc checks added after the paper was drafted (not pre-registered).

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
    mix        the training-file analyses re-scored at the operational class mix.
               The organizers moved most high-risk events into the test, so the
               training file's own share (0.8%) is depleted; the pooled events,
               ESA's full population, hold 2.1%. Covers Round 2's go/no-go CV, the
               winner's rules out of sample (P2(d)) and the information horizon
               (P1), each in its registered folds, seeds and bootstrap draws

The sealed test stays spent: nothing here trains on the training file and scores
the official test as a held-out set. The registered functions are not changed.

    .venv/bin/python -m backend.app.models.lab.review coherence --out reports/paper
    .venv/bin/python -m backend.app.models.lab.review cv --out reports/paper
    .venv/bin/python -m backend.app.models.lab.review mix --out reports/paper
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
from backend.app.models.lab import compete, data, metric, physics, protocol
from backend.app.models.lab import paper as P
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


# --- the class mix -----------------------------------------------------------------

def class_mixes(train_events: list[Event], test_events: list[Event]) -> dict:
    """High-risk shares of the training file's eligible events, the test and the two
    pooled. The pooled share is ESA's full population, so it is the operational one."""
    def share(evs):
        k = int(sum(e.target >= HIGH_RISK for e in evs))
        return {"n": len(evs), "n_high_risk": k, "prevalence": k / len(evs)}
    el = [e for e in train_events if e.test_eligible]
    pooled = [e for e in list(train_events) + list(test_events) if e.test_eligible]
    return {"training_file": share(el), "test": share([e for e in test_events if e.test_eligible]),
            "pooled": share(pooled)}


def _weights(y, prevalence):
    return None if prevalence is None else metric.label_shift_weights(y >= HIGH_RISK, prevalence)


def round2_cv_mix(train_events: list[Event], prevalence: float, repeats: int = 5, folds: int = 5,
                  seed: int = 42, bootstrap: int = 1000, log=print) -> dict:
    """Round 2's go/no-go cross-validation for its primary alone, scored at three mixes.
    Same folds, seeds and bootstrap stream as protocol.cross_validate. Decisions: the
    model's own (tuned on the training folds), Round 2's (the test's mix) and one
    tuned for ``prevalence``."""
    from sklearn.model_selection import StratifiedKFold

    b = protocol.Bundle(train_events)
    n, el = len(train_events), b.eligible
    m_ew = int((b.cand & el).sum())        # cross_validate's bootstrap draws these too
    names = ("own", "round2", "operational")
    oof = {k: [] for k in names}
    for r in range(repeats):
        o = {k: np.full(n, np.nan) for k in names}
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), b.strata)):
            t0 = time.time()
            tr = tr[b.trainable[tr]]
            mdl = next(c for c in C.regression_pool() if c.name == "hurdle_lrp_serving")
            mdl.fit(protocol._ctx(b, tr, mdl.track, seed + 1000 * r + f))
            X, lat = b.X[mdl.track][te], b.latest[te]
            o["own"][te] = mdl.predict(X, lat)
            o["round2"][te] = mdl.with_prior("hurdle_lrp_serving_bench", C.BENCH_HR_PREVALENCE).predict(X, lat)
            o["operational"][te] = mdl.with_prior("hurdle_lrp_serving_op", prevalence).predict(X, lat)
            log(f"  round 2 cv: repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")
        for k in names:
            oof[k].append(o[k][el])
    y, lrp = b.y[el], metric.lrp(b.latest[el])
    mixes = {"training_file": None, "test": C.BENCH_HR_PREVALENCE, "operational": prevalence}
    out = {}
    for mx, prev in mixes.items():
        w = _weights(y, prev)
        base = P._score(lrp, y, w)
        rows = {"lrp": base}
        for k in names:
            s = [P._score(oof[k][r], y, w) for r in range(repeats)]
            rows[k] = {**{q: float(np.mean([x[q] for x in s])) for q in base},
                       "delta_L_pct": float(np.mean([P._rel(x["L"], base["L"]) for x in s]))}
        out[mx] = rows
    rng = np.random.RandomState(seed + 7)
    boot = {(mx, k): np.full(bootstrap, np.nan) for mx in mixes for k in names}
    ny = len(y)
    for bi in range(bootstrap):
        i = rng.randint(0, ny, ny)
        if m_ew:
            rng.randint(0, m_ew, m_ew)     # the early-warning draw, so the stream stays aligned
        yi = y[i]
        for mx, prev in mixes.items():
            w = _weights(yi, prev)
            base = metric.official(lrp[i], yi, weights=w)["L"]
            for k in names:
                vals = [P._rel(metric.official(oof[k][r][i], yi, weights=w)["L"], base) for r in range(repeats)]
                if np.isfinite(vals).any():
                    boot[(mx, k)][bi] = np.nanmean(vals)
    for (mx, k), v in boot.items():
        out[mx][k]["delta_L_pct_ci95"] = protocol._ci(v)
    reg = json.loads(Path("reports/vigil-lab-r2-sealed/lab_report.json").read_text())["cv"]["regression"]
    r2, own = reg["hurdle_lrp_serving_bench"], reg["hurdle_lrp_serving"]
    reproduced = {
        "training_file_round2": abs(out["training_file"]["round2"]["delta_L_pct"] - r2["delta_L_pct_vs_lrp"]["mean"]) < 1e-9,
        "training_file_round2_ci": np.allclose(out["training_file"]["round2"]["delta_L_pct_ci95"],
                                               r2["delta_L_pct_vs_lrp"]["ci95"], rtol=0, atol=1e-9),
        "training_file_own": abs(out["training_file"]["own"]["delta_L_pct"] - own["delta_L_pct_vs_lrp"]["mean"]) < 1e-9,
        "test_round2": abs(out["test"]["round2"]["delta_L_pct"] - r2["bench_mix"]["delta_L_pct"]["mean"]) < 1e-9,
    }
    return {"model": "hurdle_lrp_serving (the Round 2 primary's fitted model)", "repeats": repeats,
            "folds": folds, "bootstrap": bootstrap, "n_eligible": int(ny), "n_high_risk": int((y >= HIGH_RISK).sum()),
            "reproduces_registered": reproduced, "mixes": out}


def winner_rules_mix(train_events: list[Event], test_events: list[Event], prevalence: float,
                     bootstrap: int = 2000, seed: int = 42) -> dict:
    """P2(d), sesc's published cascade on the training file, at three mixes. The
    random stream first replays P2 (a) and (c), so the registered intervals are
    reproduced; no model is scored on the test."""
    rng = np.random.RandomState(seed)
    s = P.side(test_events)
    hr_test = s["y"] >= HIGH_RISK
    pos, neg, nt = np.flatnonzero(hr_test), np.flatnonzero(~hr_test), len(hr_test)
    for _ in range(bootstrap):          # (a)
        rng.randint(0, nt, nt)
        rng.choice(pos, len(pos))
        rng.choice(neg, len(neg))
    for _ in range(bootstrap):          # (c)
        P.stratified_mask(hr_test, 0.9, rng)
    t = P.side([e for e in train_events if e.test_eligible])
    y = t["y"]
    lp, sp = metric.lrp(t["latest"]), compete.sesc_cascade(t)
    mixes = {"training_file": None, "test": C.BENCH_HR_PREVALENCE, "operational": prevalence}
    out = {}
    for mx, prev in mixes.items():
        w = _weights(y, prev)
        out[mx] = {"lrp": P._score(lp, y, w), "sesc_cascade": P._score(sp, y, w)}
        out[mx]["delta_L_pct"] = P._rel(out[mx]["sesc_cascade"]["L"], out[mx]["lrp"]["L"])
    boot = {mx: [] for mx in mixes}
    for _ in range(bootstrap):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        if not (yi >= HIGH_RISK).any():
            continue
        for mx, prev in mixes.items():
            w = _weights(yi, prev)
            boot[mx].append(P._rel(metric.official(sp[i], yi, weights=w)["L"],
                                   metric.official(lp[i], yi, weights=w)["L"]))
    for mx in mixes:
        out[mx]["delta_L_pct_ci95"] = protocol._ci(boot[mx])
    reg = json.loads(Path("reports/paper/resolution.json").read_text())["d_sesc_out_of_sample"]
    reproduced = {f"{mx}{suffix}": np.allclose(np.atleast_1d(out[mx][key]), np.atleast_1d(reg[rk][key]),
                                               rtol=0, atol=1e-9)
                  for mx, rk in (("training_file", "natural"), ("test", "bench_mix"))
                  for key, suffix in (("delta_L_pct", ""), ("delta_L_pct_ci95", "_ci"))}
    return {"n": int(len(y)), "n_high_risk": int((y >= HIGH_RISK).sum()), "bootstrap": bootstrap,
            "reproduces_registered": reproduced, "mixes": out}


def horizon_mix(train_events: list[Event], prevalence: float, repeats: int = 3, folds: int = 5,
                seed: int = 42, bootstrap: int = 1000, log=print) -> dict:
    """P1 at three mixes: paper.horizon's folds, seeds and bootstrap draws, plus the same
    fitted model with its decision re-tuned for ``prevalence``."""
    from sklearn.model_selection import StratifiedKFold

    cutoffs = P.CUTOFFS
    preds, y = {}, None
    for c in cutoffs:
        t0 = time.time()
        evs = [P.at_cutoff(e, c) for e in train_events]
        b = P.CutoffBundle(evs, c)
        el = b.eligible
        if y is None:
            y = b.y[el]
        elif not np.array_equal(y, b.y[el]):
            raise RuntimeError("the scored events changed between cutoffs")
        o = {k: np.full((repeats, len(evs)), np.nan) for k in ("own", "test", "operational")}
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
            for tr, te in skf.split(np.zeros(len(evs)), b.strata):
                tr = tr[b.trainable[tr]]
                m = C.Hurdle("hurdle_lrp", mode="lrp")
                m.fit(protocol._ctx(b, tr, "benchmark", seed))
                X, lat = b.X["benchmark"][te], b.latest[te]
                o["own"][r, te] = m.predict(X, lat)
                o["test"][r, te] = m.with_prior("hurdle_lrp_bench", C.BENCH_HR_PREVALENCE).predict(X, lat)
                o["operational"][r, te] = m.with_prior("hurdle_lrp_op", prevalence).predict(X, lat)
        preds[c] = {"lrp": metric.lrp(b.latest[el]), **{k: v[:, el] for k, v in o.items()}}
        log(f"  horizon: cutoff {c:.2f} d  ({time.time() - t0:.0f}s)")
    # each mix scores the decision tuned for it, as P1 did for its two
    mixes = {"training_file": ("own", None), "test": ("test", C.BENCH_HR_PREVALENCE),
             "operational": ("operational", prevalence)}
    rows = {str(c): {} for c in cutoffs}
    for mx, (k, prev) in mixes.items():
        w = _weights(y, prev)
        L2 = metric.official(preds[cutoffs[0]]["lrp"], y, weights=w)["L"]
        for c in cutoffs:
            p = preds[c]
            Lb = metric.official(p["lrp"], y, weights=w)["L"]
            Lm = [metric.official(p[k][r], y, weights=w)["L"] for r in range(repeats)]
            rows[str(c)][mx] = {"lrp_L": Lb, "model_L": float(np.mean(Lm)), "lrp_L_ratio_to_2d": Lb / L2,
                                "delta_L_pct": float(np.mean([P._rel(v, Lb) for v in Lm]))}
    rng = np.random.RandomState(seed + 7)
    boot = {(str(c), mx, q): np.full(bootstrap, np.nan) for c in cutoffs for mx in mixes for q in ("dl", "ratio")}
    n = len(y)
    for bi in range(bootstrap):
        i = rng.randint(0, n, n)
        yi = y[i]
        if not (yi >= HIGH_RISK).any():
            continue
        for mx, (k, prev) in mixes.items():
            w = _weights(yi, prev)
            b2 = metric.official(preds[cutoffs[0]]["lrp"][i], yi, weights=w)["L"]
            for c in cutoffs:
                p = preds[c]
                Lb = metric.official(p["lrp"][i], yi, weights=w)["L"]
                Lm = [metric.official(p[k][r][i], yi, weights=w)["L"] for r in range(repeats)]
                boot[(str(c), mx, "dl")][bi] = np.nanmean([P._rel(v, Lb) for v in Lm])
                boot[(str(c), mx, "ratio")][bi] = Lb / b2 if b2 > 0 else np.nan
    for c in cutoffs:
        for mx in mixes:
            rows[str(c)][mx]["delta_L_pct_ci95"] = protocol._ci(boot[(str(c), mx, "dl")])
            rows[str(c)][mx]["lrp_L_ratio_to_2d_ci95"] = protocol._ci(boot[(str(c), mx, "ratio")])
    reg = json.loads(Path("reports/paper/horizon.json").read_text())["rows"]
    reproduced = {}
    for c in cutoffs:
        g, mine = reg[str(c)], rows[str(c)]
        for mx, suffix in (("training_file", ""), ("test", "_bench_mix")):
            for key in ("delta_L_pct", "lrp_L_ratio_to_2d"):
                reproduced[f"{c}_{mx}_{key}"] = (
                    abs(mine[mx][key] - g[key + suffix]) < 1e-9
                    and np.allclose(mine[mx][key + "_ci95"], g[key + suffix + "_ci95"], rtol=0, atol=1e-9))
    return {"cutoffs": list(cutoffs), "repeats": repeats, "folds": folds, "bootstrap": bootstrap,
            "n_scored": int(n), "n_high_risk": int((y >= HIGH_RISK).sum()),
            "reproduces_registered": reproduced, "rows": rows}


def mix_review(train_events: list[Event], test_events: list[Event], seed: int = 42, log=print) -> dict:
    mixes = class_mixes(train_events, test_events)
    op = mixes["pooled"]["prevalence"]
    log(f"operational mix: {mixes['pooled']['n_high_risk']} of {mixes['pooled']['n']} high-risk ({100 * op:.2f}%)")
    winner = winner_rules_mix(train_events, test_events, op, seed=seed)
    log(f"  winner's rules at the operational mix: {winner['mixes']['operational']['delta_L_pct']:+.1f}%")
    r2 = round2_cv_mix(train_events, op, seed=seed, log=log)
    hor = horizon_mix(train_events, op, seed=seed, log=log)
    return {"analysis": "Post hoc (review): the training-file analyses re-scored at the operational "
                        "class mix with label-shift weights",
            "why": "the organizers moved most high-risk events into the test, so the training file's own "
                   "share is depleted; the pooled events are ESA's full population",
            "class_mixes": mixes, "operational_prevalence": op, "round2_cv": r2,
            "winner_rules": winner, "horizon": hor}


def main(argv=None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description="VIGIL Lab: post-hoc checks added after the paper was drafted")
    p.add_argument("analysis", choices=("coherence", "cv", "mix"))
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
    if a.analysis == "mix":
        _write(out, "review_mix", mix_review(train, test, seed=a.seed), t0)
        return 0
    _write(out, "review_cv", cv_review(train + test, seed=a.seed), t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
