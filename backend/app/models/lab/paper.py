"""Round 3: the analyses behind the research paper.

Pre-registered in docs/v2/vigil-lab.md ("Round 3 pre-registration", commit
c568d6a) before any of them ran. The sealed test is spent, so none of these
trains a model on the training file and scores it on the official test as a
held-out set:

    horizon     P1  how the score moves as the cutoff nears TCA (training file)
    resolution  P2  what the official test can resolve (LRP, published rules)
    attack      P3  zero-information leaderboard climbing, 2019 protocol
    ablation    P4  which fields carry VIGIL-E's signal (pooled CV)
    nbcorr      P5  the organisers' t-test, corrected for overlapping splits

    .venv/bin/python -m backend.app.models.lab.paper horizon --out reports/paper

Each subcommand writes ``<out>/<name>.json``; the file paths default to the
Kelvins files under data/kelvins/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import compete, data, metric, protocol
from backend.app.models.lab import features as F
from backend.app.models.lab.data import HIGH_RISK, Event

KELVINS = "data/kelvins"
TEST_DIR = f"{KELVINS}/Collision Avoidance Challenge - Dataset/kelvins_competition_data"
DEFAULTS = {"train": f"{KELVINS}/train_data.csv", "test": f"{TEST_DIR}/test_data.csv",
            "test_targets": f"{TEST_DIR}/test_data_private.csv"}
SEALED_REPORTS = ("reports/vigil-lab-sealed/lab_report.json",
                  "reports/vigil-lab-r2-sealed/lab_report.json")
POOLED_REPORTS = ("reports/vigil-lab-pooled/lab_report.json",
                  "reports/vigil-lab-r2-pooled/lab_report.json")


# --- shared helpers -------------------------------------------------------------

def _rel(L: float, base: float) -> float:
    """Percent change of L against base (negative = better than base)."""
    return 100.0 * (L / base - 1.0) if np.isfinite(L) and np.isfinite(base) and base > 0 else math.nan


def _dist(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [2.5, 5, 50, 95, 97.5])
    return {"n": int(len(x)), "mean": float(np.mean(x)), "sd": float(np.std(x)),
            "p025": float(q[0]), "p05": float(q[1]), "median": float(q[2]),
            "p95": float(q[3]), "p975": float(q[4])}


def _score(pred, y, w=None) -> dict:
    s = metric.official(pred, y, weights=w)
    return {k: s[k] for k in ("L", "mse_hr", "f2", "precision", "recall")}


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def side(events: list[Event]) -> dict:
    """What LRP and sesc's published cascade read from each event's latest
    admissible CDM (the same fields compete.fit_sides takes from the
    benchmark table)."""
    last = [e.inputs[-1] for e in events]
    return {
        "latest": np.array([e.latest_risk for e in events], float),
        "y": np.array([e.target for e in events], float),
        "payload": np.array([str(r.get("c_object_type", "") or "").strip().upper() == "PAYLOAD"
                             for r in last]),
        "t_span": np.array([_num(r.get("t_span")) for r in last], float),
        "miss": np.array([_num(r.get("miss_distance")) for r in last], float),
    }


def _subset(s: dict, mask) -> dict:
    return {k: v[mask] for k, v in s.items()}


def stratified_mask(hr: np.ndarray, frac: float, rng) -> np.ndarray:
    """A random subset holding ``frac`` of each class (the organisers' visible
    subset, p_high = p_low = frac)."""
    m = np.zeros(len(hr), bool)
    for cls in (True, False):
        idx = np.flatnonzero(hr == cls)
        m[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return m


def _code() -> dict:
    from backend.app.models.lab.__main__ import _code_version
    return _code_version()


# --- P1: the information horizon ------------------------------------------------

CUTOFFS = (2.0, 1.75, 1.5, 1.25, 1.0)


def at_cutoff(ev: Event, c: float) -> Event:
    """The event as seen at cutoff ``c`` days: every CDM with time_to_tca >= c,
    never the final CDM (whose risk is the target)."""
    extra = [r for r in ev.late
             if float(r["time_to_tca"]) >= c and float(r["time_to_tca"]) > ev.final_tta]
    return Event(f"{ev.event_id}@{c}", ev.inputs + extra, ev.target, ev.final_tta,
                 ev.n_cdm_total, ev.first_tta, ev.source)


class CutoffBundle:
    """protocol.Bundle for the benchmark track only, with the training set set
    by the cutoff: events whose final CDM lies inside it (for the others the
    target is itself an admissible input)."""

    def __init__(self, events: list[Event], cutoff: float):
        self.events = events
        self.y = np.array([e.target for e in events], float)
        self.latest = np.array([e.latest_risk for e in events], float)
        self.eligible = np.array([e.test_eligible for e in events], bool)
        self.trainable = np.array([e.final_tta < cutoff for e in events], bool)
        self.y_aux = self.y.copy()          # near-miss labels are unused here
        esc = (self.latest < HIGH_RISK) & (self.y >= HIGH_RISK)
        self.strata = 2 * esc.astype(int) + (self.y >= HIGH_RISK).astype(int)
        df = F.table(events, "benchmark")
        ct = [c for c in df.columns if c.startswith("l_ctype_")]
        if ct:
            df[ct] = df[ct].fillna(0.0)
        self.columns = {"benchmark": list(df.columns)}
        self.X = {"benchmark": df.to_numpy(dtype=float)}
        protocol._check_float32(self.X["benchmark"], self.columns["benchmark"], "benchmark")
        self.colindex = {c: i for i, c in enumerate(self.columns["benchmark"])}


def horizon(train_events: list[Event], repeats: int = 3, folds: int = 5, seed: int = 42,
            bootstrap: int = 1000, cutoffs=CUTOFFS, log=print) -> dict:
    """P1. LRP and ``hurdle_lrp`` at each cutoff, scored on the training file's
    test-eligible events (the same events at every cutoff). CV seeds as in
    protocol.ceiling, so c = 2.0, repeat 1 reproduces its admissible row."""
    from sklearn.model_selection import StratifiedKFold

    preds, y_el = {}, None
    for c in cutoffs:
        t0 = time.time()
        evs = [at_cutoff(e, c) for e in train_events]
        b = CutoffBundle(evs, c)
        el = b.eligible
        if y_el is None:
            y_el = b.y[el]
        elif not np.array_equal(y_el, b.y[el]):
            raise RuntimeError("the scored events changed between cutoffs")
        nat = np.full((repeats, len(evs)), np.nan)
        bench = np.full((repeats, len(evs)), np.nan)
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
            for tr, te in skf.split(np.zeros(len(evs)), b.strata):
                tr = tr[b.trainable[tr]]
                m = C.Hurdle("hurdle_lrp", mode="lrp")
                m.fit(protocol._ctx(b, tr, "benchmark", seed))
                nat[r, te] = m.predict(b.X["benchmark"][te], b.latest[te])
                d = m.with_prior("hurdle_lrp_bench", C.BENCH_HR_PREVALENCE)
                bench[r, te] = d.predict(b.X["benchmark"][te], b.latest[te])
        preds[c] = {"lrp": metric.lrp(b.latest[el]), "nat": nat[:, el], "bench": bench[:, el],
                    "n_inputs": float(np.mean([len(e.inputs) for e, ok in zip(evs, el) if ok])),
                    "n_trainable": int(b.trainable.sum()),
                    "lrp_recall_events": int(((b.latest[el] >= HIGH_RISK) & (y_el >= HIGH_RISK)).sum())}
        L0 = metric.official(preds[c]["lrp"], y_el)["L"]
        Lm = np.mean([metric.official(nat[r, el], y_el)["L"] for r in range(repeats)])
        log(f"  cutoff {c:.2f} d: LRP L {L0:.3f}, hurdle_lrp L {Lm:.3f}  ({time.time() - t0:.0f}s)")

    hr = y_el >= HIGH_RISK
    w = metric.label_shift_weights(hr, C.BENCH_HR_PREVALENCE)
    rows = {}
    for c in cutoffs:
        p = preds[c]
        lrp_n, lrp_b = _score(p["lrp"], y_el), _score(p["lrp"], y_el, w)
        mod_n = [_score(p["nat"][r], y_el) for r in range(repeats)]
        mod_b = [_score(p["bench"][r], y_el, w) for r in range(repeats)]
        rows[str(c)] = {
            "cutoff_days": c, "mean_inputs_per_event": p["n_inputs"], "n_trainable": p["n_trainable"],
            "lrp_high_calls_correct": p["lrp_recall_events"],
            "lrp": lrp_n, "lrp_bench_mix": lrp_b,
            "model": {k: float(np.mean([s[k] for s in mod_n])) for k in lrp_n},
            "model_bench_mix": {k: float(np.mean([s[k] for s in mod_b])) for k in lrp_n},
            "model_L_per_repeat": [s["L"] for s in mod_n],
            "delta_L_pct": float(np.mean([_rel(s["L"], lrp_n["L"]) for s in mod_n])),
            "delta_L_pct_bench_mix": float(np.mean([_rel(s["L"], lrp_b["L"]) for s in mod_b])),
            "lrp_L_ratio_to_2d": lrp_n["L"] / metric.official(preds[cutoffs[0]]["lrp"], y_el)["L"],
            "lrp_L_ratio_to_2d_bench_mix":
                lrp_b["L"] / metric.official(preds[cutoffs[0]]["lrp"], y_el, w)["L"],
        }
    # paired bootstrap: one resample of events per replicate, shared by every
    # cutoff and model; model rows average over repeats inside the replicate
    rng = np.random.RandomState(seed + 7)
    keys = ("dl", "dl_bench", "ratio", "ratio_bench", "lrp_L", "model_L")
    boot = {str(c): {k: np.full(bootstrap, np.nan) for k in keys} for c in cutoffs}
    n = len(y_el)
    for bi in range(bootstrap):
        i = rng.randint(0, n, n)
        yi = y_el[i]
        if not (yi >= HIGH_RISK).any():
            continue
        wi = metric.label_shift_weights(yi >= HIGH_RISK, C.BENCH_HR_PREVALENCE)
        b2 = metric.official(preds[cutoffs[0]]["lrp"][i], yi)["L"]
        b2w = metric.official(preds[cutoffs[0]]["lrp"][i], yi, weights=wi)["L"]
        for c in cutoffs:
            p, bc = preds[c], boot[str(c)]
            Lb = metric.official(p["lrp"][i], yi)["L"]
            Lbw = metric.official(p["lrp"][i], yi, weights=wi)["L"]
            Lm = [metric.official(p["nat"][r][i], yi)["L"] for r in range(repeats)]
            Lmw = [metric.official(p["bench"][r][i], yi, weights=wi)["L"] for r in range(repeats)]
            bc["dl"][bi] = np.nanmean([_rel(v, Lb) for v in Lm])
            bc["dl_bench"][bi] = np.nanmean([_rel(v, Lbw) for v in Lmw])
            bc["ratio"][bi] = Lb / b2 if b2 > 0 else np.nan
            bc["ratio_bench"][bi] = Lbw / b2w if b2w > 0 else np.nan
            bc["lrp_L"][bi], bc["model_L"][bi] = Lb, float(np.nanmean(Lm))
    for c in cutoffs:
        bc, row = boot[str(c)], rows[str(c)]
        row["delta_L_pct_ci95"] = protocol._ci(bc["dl"])
        row["delta_L_pct_bench_mix_ci95"] = protocol._ci(bc["dl_bench"])
        row["lrp_L_ratio_to_2d_ci95"] = protocol._ci(bc["ratio"])
        row["lrp_L_ratio_to_2d_bench_mix_ci95"] = protocol._ci(bc["ratio_bench"])
        row["lrp_L_ci95"] = protocol._ci(bc["lrp_L"])
        row["model_L_ci95"] = protocol._ci(bc["model_L"])
    return {"analysis": "P1 information horizon", "cutoffs": list(cutoffs), "repeats": repeats,
            "folds": folds, "bootstrap": bootstrap, "seed": seed, "n_scored": int(n),
            "n_scored_high_risk": int(hr.sum()), "bench_prevalence": C.BENCH_HR_PREVALENCE,
            "rows": rows}


# --- P2: what the official test can resolve ---------------------------------------

def _sealed_mde(paths=SEALED_REPORTS) -> list[dict]:
    """MDE of every pre-registered primary in the committed sealed reports, and
    the high-risk test events needed to detect a 2% and a 5% gain."""
    out = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        test = rep.get("test") or {}
        n_hr = test.get("n_hr")
        for name, row in (test.get("regression") or {}).items():
            d = row.get("delta_L_pct_vs_lrp") or {}
            mde = _num(d.get("mde"))
            if name == "lrp" or not d.get("sd_boot") or not math.isfinite(mde):
                continue    # the baseline itself, or a ratio undefined on most replicates
            out.append({"report": path, "candidate": name, "primary": bool(row.get("primary")),
                        "track": row.get("track"), "n_hr": n_hr, "L": row["L"]["mean"],
                        "delta_L_pct": d["mean"], "ci95": d["ci95"], "ci975": d.get("ci975"),
                        "sd_boot": d["sd_boot"], "mde_pct": mde,
                        "n_hr_for_2pct": int(math.ceil(n_hr * (mde / 2.0) ** 2)) if n_hr else None,
                        "n_hr_for_5pct": int(math.ceil(n_hr * (mde / 5.0) ** 2)) if n_hr else None})
    return out


def resolution(train_events: list[Event], test_events: list[Event], bootstrap: int = 2000,
               seed: int = 42, log=print) -> dict:
    """P2 (a)-(d). Uses only LRP and sesc's published cascade on the test."""
    rng = np.random.RandomState(seed)
    s = side(test_events)
    y, hr = s["y"], s["y"] >= HIGH_RISK
    lrp_pred, sesc_pred = metric.lrp(s["latest"]), compete.sesc_cascade(s)
    full = {"lrp": _score(lrp_pred, y), "sesc_cascade": _score(sesc_pred, y),
            "n": int(len(y)), "n_hr": int(hr.sum())}
    full["sesc_delta_L_pct"] = _rel(full["sesc_cascade"]["L"], full["lrp"]["L"])
    log(f"  full test: LRP L {full['lrp']['L']:.4f}, sesc cascade L {full['sesc_cascade']['L']:.4f}")

    # (a) bootstrap of LRP's L over test events (plain; stratified as a sensitivity)
    plain, strat = {k: [] for k in ("L", "mse_hr", "f2")}, {k: [] for k in ("L", "mse_hr", "f2")}
    pos, neg = np.flatnonzero(hr), np.flatnonzero(~hr)
    for _ in range(bootstrap):
        i = rng.randint(0, len(y), len(y))
        j = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for store, idx in ((plain, i), (strat, j)):
            sc = metric.official(lrp_pred[idx], y[idx])
            for k in store:
                store[k].append(sc[k])
    a = {"plain": {k: _dist(v) for k, v in plain.items()},
         "stratified": {k: _dist(v) for k, v in strat.items()}}

    # (b) minimum detectable effects of the sealed primaries
    b = _sealed_mde()

    # (c) the 2019 design: visible subsets with p_high = p_low = 0.9
    vis_rel, d_vis, d_full, d_hold, lrp_hold = [], [], [], [], []
    for _ in range(bootstrap):
        v = stratified_mask(hr, 0.9, rng)
        Lv = metric.official(lrp_pred[v], y[v])["L"]
        Lh = metric.official(lrp_pred[~v], y[~v])["L"]
        vis_rel.append(_rel(Lv, full["lrp"]["L"]))
        lrp_hold.append(Lh)
        d_vis.append(_rel(metric.official(sesc_pred[v], y[v])["L"], Lv))
        d_hold.append(_rel(metric.official(sesc_pred[~v], y[~v])["L"], Lh))
    c = {"lrp_visible_vs_full_pct": _dist(vis_rel),
         "lrp_visible_vs_full_abs_pct": _dist(np.abs(vis_rel)),
         "lrp_holdout_L": _dist(lrp_hold),
         "sesc_delta_L_pct_visible": _dist(d_vis), "sesc_delta_L_pct_full": full["sesc_delta_L_pct"],
         "sesc_delta_L_pct_holdout": _dist(d_hold),
         "holdout_n_hr": int(round(0.1 * hr.sum()))}

    # (d) sesc's published cascade out of sample: the training file's eligible events
    el = [e for e in train_events if e.test_eligible]
    t = side(el)
    yt, hrt = t["y"], t["y"] >= HIGH_RISK
    wt = metric.label_shift_weights(hrt, C.BENCH_HR_PREVALENCE)
    lp, sp = metric.lrp(t["latest"]), compete.sesc_cascade(t)
    d = {"n": int(len(yt)), "n_hr": int(hrt.sum()),
         "natural": {"lrp": _score(lp, yt), "sesc_cascade": _score(sp, yt)},
         "bench_mix": {"lrp": _score(lp, yt, wt), "sesc_cascade": _score(sp, yt, wt)}}
    for mix in ("natural", "bench_mix"):
        d[mix]["delta_L_pct"] = _rel(d[mix]["sesc_cascade"]["L"], d[mix]["lrp"]["L"])
    bn, bb = [], []
    for _ in range(bootstrap):
        i = rng.randint(0, len(yt), len(yt))
        yi = yt[i]
        if not (yi >= HIGH_RISK).any():
            continue
        wi = metric.label_shift_weights(yi >= HIGH_RISK, C.BENCH_HR_PREVALENCE)
        bn.append(_rel(metric.official(sp[i], yi)["L"], metric.official(lp[i], yi)["L"]))
        bb.append(_rel(metric.official(sp[i], yi, weights=wi)["L"],
                       metric.official(lp[i], yi, weights=wi)["L"]))
    d["natural"]["delta_L_pct_ci95"], d["bench_mix"]["delta_L_pct_ci95"] = protocol._ci(bn), protocol._ci(bb)
    log(f"  sesc cascade on the training file: {d['bench_mix']['delta_L_pct']:+.1f}% at the test mix "
        f"{d['bench_mix']['delta_L_pct_ci95']}")
    return {"analysis": "P2 test resolution", "bootstrap": bootstrap, "seed": seed, "full_test": full,
            "a_lrp_bootstrap": a, "b_sealed_mde": b, "c_visible_subsets": c, "d_sesc_out_of_sample": d}


# --- P3: zero-information leaderboard climbing ----------------------------------

def climb(latest: np.ndarray, y: np.ndarray, visible: np.ndarray, rng, k: int = 20,
          budget: int = 90) -> dict:
    """One simulated 2019 team with no information beyond the LRP. Each
    submission perturbs ``k`` events drawn from the whole test set and is kept
    only if the visible-subset score (the only feedback) improves."""
    base = metric.lrp(latest)
    pred = base.copy()
    best = metric.official(pred[visible], y[visible])["L"]
    accepted = 0
    for _ in range(budget):
        idx = rng.choice(len(y), k, replace=False)
        cand = pred.copy()
        if rng.rand() < 0.5:
            cand[idx] = np.where(cand[idx] >= HIGH_RISK, data.LOW_PREDICTION, -5.5)
        else:
            sel = idx[cand[idx] >= HIGH_RISK]
            cand[sel] = cand[sel] + rng.normal(0.0, 0.3, len(sel))
            cand = metric.clip_low(np.minimum(cand, 0.0))
        L = metric.official(cand[visible], y[visible])["L"]
        if L < best:
            pred, best = cand, L
            accepted += 1
    out = {"accepted": accepted, "full_L": metric.official(pred, y)["L"]}
    for name, m in (("visible", visible), ("full", np.ones(len(y), bool)), ("holdout", ~visible)):
        out[f"{name}_dl_pct"] = _rel(metric.official(pred[m], y[m])["L"],
                                     metric.official(base[m], y[m])["L"])
    return out


def attack(test_events: list[Event], runs: int = 500, budget: int = 90, ks=(20, 5, 50),
           seed: int = 42, log=print) -> dict:
    """P3. ``runs`` independent teams per subset size k, each with a fresh
    visible subset (90% of each class)."""
    s = side(test_events)
    latest, y = s["latest"], s["y"]
    hr = y >= HIGH_RISK
    lrp_full = metric.official(metric.lrp(latest), y)["L"]
    res = {}
    for k in ks:
        t0 = time.time()
        rng = np.random.RandomState(seed + k)
        rows = [climb(latest, y, stratified_mask(hr, 0.9, rng), rng, k, budget) for _ in range(runs)]
        med_L = float(np.median([r["full_L"] for r in rows]))
        res[str(k)] = {"k": k, "visible_dl_pct": _dist([r["visible_dl_pct"] for r in rows]),
                       "full_dl_pct": _dist([r["full_dl_pct"] for r in rows]),
                       "holdout_dl_pct": _dist([r["holdout_dl_pct"] for r in rows]),
                       "full_L": _dist([r["full_L"] for r in rows]),
                       "accepted": _dist([r["accepted"] for r in rows]),
                       "median_full_L_rank": compete.placement(med_L),
                       "share_beating_lrp_full": float(np.mean([r["full_L"] < lrp_full for r in rows]))}
        log(f"  k={k}: full-test ΔL median {res[str(k)]['full_dl_pct']['median']:+.1f}% "
            f"(hold-out {res[str(k)]['holdout_dl_pct']['median']:+.1f}%), "
            f"median rank {res[str(k)]['median_full_L_rank']}  ({time.time() - t0:.0f}s)")
    return {"analysis": "P3 zero-information leaderboard climbing", "runs": runs, "budget": budget,
            "visible_fraction": 0.9, "primary_k": ks[0], "seed": seed, "lrp_full_L": lrp_full,
            "leaderboard": compete.LEADERBOARD, "n_teams": compete.N_TEAMS, "by_k": res}


# --- P4: which fields carry VIGIL-E's signal --------------------------------------

COV_HISTORY = {"h_log_sigma_rate", "h_c_log_sigma_rate", "h_sigma_shrink_to_eval",
               "h_maha_slope", "h_headroom_mre"}
_CORRELATION = re.compile(r"^[tc]_c(t|n|rdot|tdot|ndot)_")


def is_covariance_feature(name: str) -> bool:
    """The pre-registered name rule for covariance-derived benchmark features."""
    base = name[2:] if name.startswith("l_") else name
    return ("_sigma_" in base or "covariance_det" in base or "mahalanobis" in base
            or "max_risk_" in base or bool(_CORRELATION.match(base))
            or name.startswith("p_") or name in COV_HISTORY)


def feature_sets(columns: list[str]) -> dict:
    cov = [c for c in columns if is_covariance_feature(c)]
    return {"A0_all_benchmark": list(columns),
            "A1_without_covariance": [c for c in columns if not is_covariance_feature(c)],
            "A2_serving_plus_covariance": [c for c in columns
                                           if c.startswith("s_") or is_covariance_feature(c)],
            "_covariance": cov}


def ablation(events: list[Event], repeats: int = 5, folds: int = 5, seed: int = 42,
             bootstrap: int = 1000, log=print) -> dict:
    """P4. e_hurdle refit on three feature sets in pooled CV, with the folds
    and seeds of protocol.cross_validate."""
    from sklearn.model_selection import StratifiedKFold

    b = protocol.Bundle(events)
    cols = b.columns["benchmark"]
    sets = feature_sets(cols)
    variants = [k for k in sets if not k.startswith("_")]
    jj = {k: np.array([b.colindex[c] for c in sets[k]], int) for k in variants}
    n = len(events)
    dom = b.cand & b.eligible
    y_ew = b.esc[dom]
    oof = {k: np.full((repeats, n), np.nan) for k in variants + ["e_logit"]}
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), b.strata)):
            t0 = time.time()
            tr = tr[b.trainable[tr]]
            s = seed + 1000 * r + f
            for k in variants:
                X = b.X["benchmark"][:, jj[k]]
                ctx = C.Context(X[tr], b.y[tr], b.latest[tr], b.strata[tr], s, b.eligible[tr],
                                {"colindex": {c: i for i, c in enumerate(sets[k])},
                                 "colnames": sets[k], "y_aux": b.y_aux[tr]})
                h = C.Hurdle(f"hurdle_{k}")
                h.fit(ctx)
                oof[k][r, te] = h.prob_high(X[te], b.latest[te])
            lg = C.ELogit().fit(protocol._ctx(b, tr, "benchmark", s))
            oof["e_logit"][r, te] = lg.score(None, b.latest[te])
            log(f"  repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")
    names = variants + ["e_logit"]
    scores = {k: oof[k][:, dom] for k in names}
    per = {k: [metric.early_warning(y_ew, scores[k][r], None, 0.05) for r in range(repeats)]
           for k in names}
    summary = {k: {m: float(np.mean([p[m] for p in per[k]]))
                   for m in ("pr_auc", "roc_auc", "pauc_fpr10", "recall_at_budget")} for k in names}
    contrasts = {"A0_minus_A1": ("A0_all_benchmark", "A1_without_covariance"),
                 "A1_minus_logit": ("A1_without_covariance", "e_logit"),
                 "A2_minus_logit": ("A2_serving_plus_covariance", "e_logit"),
                 "A2_minus_A0": ("A2_serving_plus_covariance", "A0_all_benchmark"),
                 "A0_minus_logit": ("A0_all_benchmark", "e_logit")}
    point = {c: float(np.mean([per[a][r]["pr_auc"] - per[z][r]["pr_auc"] for r in range(repeats)]))
             for c, (a, z) in contrasts.items()}
    rng = np.random.RandomState(seed + 7)
    boot = {c: np.full(bootstrap, np.nan) for c in contrasts}
    m = len(y_ew)
    for bi in range(bootstrap):
        j = rng.randint(0, m, m)
        yj = y_ew[j]
        if yj.sum() == 0:
            continue
        ap = {k: [metric.average_precision(yj, scores[k][r][j]) for r in range(repeats)] for k in names}
        for c, (a, z) in contrasts.items():
            boot[c][bi] = np.mean([ap[a][r] - ap[z][r] for r in range(repeats)])
    out_c = {c: {"mean": point[c], "ci95": protocol._ci(boot[c]),
                 "sd_boot": float(np.nanstd(boot[c])), "mde": 2.8 * float(np.nanstd(boot[c]))}
             for c in contrasts}
    a01 = out_c["A0_minus_A1"]["ci95"]
    a1l = out_c["A1_minus_logit"]["ci95"]
    if a01[0] is not None and a01[0] > 0 and a1l[0] is not None and a1l[0] <= 0 <= a1l[1]:
        reading = "covariance carries the signal"
    elif a01[0] is not None and a01[0] <= 0 <= a01[1]:
        reading = "inconclusive"
    else:
        reading = "other"
    reference = {}
    for path in POOLED_REPORTS:
        p = Path(path)
        if p.exists():
            ew = json.loads(p.read_text())["cv"]["early_warning"]
            reference[path] = {k: ew[k]["pr_auc"]["mean"] for k in ("e_hurdle", "e_logit") if k in ew}
    return {"analysis": "P4 VIGIL-E feature ablation (pooled CV)", "repeats": repeats, "folds": folds,
            "bootstrap": bootstrap, "seed": seed, "n_events": n, "n_candidates": int(dom.sum()),
            "n_escalations": int(y_ew.sum()),
            "feature_counts": {k: len(v) for k, v in sets.items()},
            "covariance_features": sets["_covariance"], "summary": summary, "contrasts": out_c,
            "reading": reading, "reference_pooled_reports": reference}


# --- P5: the organisers' t-test, corrected for overlapping splits --------------

def nb_correction(t: float = -10.23, J: int = 505, test_frac: float = 0.2,
                  f2_lgbm: float = 0.60732, f2_lrp: float = 0.59967) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio 2003) applied to the paired
    t-test of Uriot et al. 2022, §6.3. The naive test treats J overlapping
    random splits as independent; the correction inflates the variance of the
    mean difference from s²/J to s²·(1/J + n_test/n_train)."""
    from scipy import stats

    ratio = test_frac / (1.0 - test_frac)
    factor = math.sqrt((1.0 / J) / (1.0 / J + ratio))
    diff = f2_lgbm - f2_lrp
    sd = abs(diff) * math.sqrt(J) / abs(t)            # SD of per-split differences implied by t
    se_corr = sd * math.sqrt(1.0 / J + ratio)
    tq = stats.t.ppf(0.975, J - 1)
    t_corr = t * factor
    return {"analysis": "P5 corrected resampled t-test", "t_naive": t, "J": J, "test_frac": test_frac,
            "p_naive": float(2 * stats.t.sf(abs(t), J - 1)), "factor": factor, "t_corrected": t_corr,
            "p_corrected": float(2 * stats.t.sf(abs(t_corr), J - 1)),
            "mean_f2_gain": diff, "implied_sd_per_split": sd,
            "f2_gain_ci95_corrected": [diff - tq * se_corr, diff + tq * se_corr],
            "f2_gain_pct": 100 * diff / f2_lrp,
            "f2_gain_pct_ci95_corrected": [100 * (diff - tq * se_corr) / f2_lrp,
                                           100 * (diff + tq * se_corr) / f2_lrp]}


# --- CLI --------------------------------------------------------------------------------

def _write(out: Path, name: str, payload: dict, t0: float) -> None:
    payload = {**payload, "code": _code(), "runtime_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"→ {out / f'{name}.json'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VIGIL Lab Round 3: the paper's analyses")
    p.add_argument("analysis", choices=("horizon", "resolution", "attack", "ablation", "nbcorr"))
    p.add_argument("--train", default=DEFAULTS["train"])
    p.add_argument("--test", default=DEFAULTS["test"])
    p.add_argument("--test-targets", default=DEFAULTS["test_targets"])
    p.add_argument("--out", default="reports/paper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-events", type=int, default=0, help="debug: subsample training events")
    a = p.parse_args(argv)
    out, t0 = Path(a.out), time.time()
    if a.analysis == "nbcorr":
        _write(out, "nbcorr", nb_correction(), t0)
        return 0
    need_test = a.analysis in ("resolution", "attack", "ablation")
    train, test = data.load(a.train, a.test if need_test else None,
                            a.test_targets if need_test else None)
    if a.limit_events:
        import random
        random.Random(a.seed).shuffle(train)
        train = train[:a.limit_events]
    print(f"loaded {len(train)} training events, {len(test)} test events")
    if a.analysis == "horizon":
        res = horizon(train, seed=a.seed)
    elif a.analysis == "resolution":
        res = resolution(train, test, seed=a.seed)
    elif a.analysis == "attack":
        res = attack(test, seed=a.seed)
    else:
        res = ablation(train + test, seed=a.seed)
    if a.limit_events:
        res["debug_limit_events"] = a.limit_events
    _write(out, a.analysis, res, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
