"""Evaluation protocol: repeated grouped CV -> paired bootstrap -> sealed test.

* One row per event, so a row split IS an event split. Augmentation rows
  (earlier cutoffs of the same event) enter only the training fold of their
  parent event and follow it through every inner split; they are never scored.
* Training uses only events whose final CDM lies inside the input cutoff
  (``trainable``): for the others the "final" CDM is itself an admissible
  input, so their target equals the latest input by construction and they
  do not represent the prediction task.
* Stratified on (escalates, high_risk); R repeats x K folds.
* Scoring uses ONLY test-eligible events (the official filters) - for VIGIL
  with the official L, for VIGIL-E on eligible escalation candidates, so CV
  and the sealed test apply the same filters. Their class mix still differs:
  the organisers enriched the test set with high-risk events (Uriot et al.
  2022, §4.2: 0.8% high-risk in CV vs 6.9% on the test), so CV magnitudes
  are not replicates of the sealed ones.
* Paired bootstrap with SHARED resample indices: every replicate draws one
  set of events and evaluates every candidate and its baseline on it, for
  every repeat, averaging over repeats inside the replicate. Intervals are
  therefore for the same quantity as the reported mean. Exploratory
  candidates also get max-T simultaneous intervals (multiplicity-adjusted);
  the pre-registered primaries (candidates.PRIMARY) are single tests.
* The sealed official test set is scored once, by the final run (the CLI
  refuses --test-targets without --final), with its own paired bootstrap,
  plus a fidelity check that the latest-risk baseline reproduces the
  organisers' published score.
"""
from __future__ import annotations

import time

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import features as F
from backend.app.models.lab import metric
from backend.app.models.lab.data import HIGH_RISK, INPUT_CUTOFF_DAYS, Event

AUG_CUTOFFS = (2.5, 3.0, 4.0)
# Uriot et al. 2022, Table 3 (latest-risk baseline on the official test set).
# VERIFY against the paper before quoting; used only as a data-fidelity check.
PUBLISHED_LRP = {"L": 0.694, "mse_hr": 0.513, "f2": 0.739}
TRACKS = ("benchmark", "last", "serving", "eserving")


class Bundle:
    """Feature matrices per track for a list of events, aligned to a column
    order (the training columns)."""

    def __init__(self, events: list[Event], columns: dict | None = None):
        self.events = events
        self.y = np.array([e.target for e in events], float)
        self.latest = np.array([e.latest_risk for e in events], float)
        self.eligible = np.array([e.test_eligible for e in events], bool)
        self.trainable = np.array([e.final_tta < INPUT_CUTOFF_DAYS for e in events], bool)
        # near-miss label (training LABEL only, never an input): the highest
        # risk of any CDM inside the cutoff, the final one included
        self.y_aux = np.array([_max_late_risk(e) for e in events], float)
        self.cand = self.latest < HIGH_RISK
        self.esc = self.cand & (self.y >= HIGH_RISK)
        self.strata = 2 * self.esc.astype(int) + (self.y >= HIGH_RISK).astype(int)
        self.X, self.columns = {}, {}
        for track in TRACKS:
            df = F.table(events, track)
            if columns and track in columns:
                df = df.reindex(columns=columns[track])
            ct = [c for c in df.columns if c.startswith("l_ctype_")]
            if ct:
                df[ct] = df[ct].fillna(0.0)
            self.columns[track] = list(df.columns)
            self.X[track] = df.to_numpy(dtype=float)
            _check_float32(self.X[track], self.columns[track], track)
        self.colindex = {c: i for i, c in enumerate(self.columns["benchmark"])}


def _max_late_risk(ev: Event) -> float:
    vals = [ev.target]
    for r in ev.late:
        try:
            v = float(r["risk"])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return max(vals)


F32_MAX = float(np.finfo(np.float32).max)


def _check_float32(X: np.ndarray, columns: list, track: str) -> None:
    """Fail fast, naming the column, on values xgboost (float32) would read
    as inf - instead of an opaque crash deep inside a fit."""
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(X) & (np.abs(X) > F32_MAX)
    if bad.any():
        cols = sorted({columns[j] for j in np.flatnonzero(bad.any(axis=0))})
        raise ValueError(f"{track} features exceed float32 range in {cols}; transform them "
                         "(see features._slog10) before fitting")


def truncate(ev: Event, cutoff: float) -> Event | None:
    ins = [r for r in ev.inputs if float(r["time_to_tca"]) >= cutoff]
    if not ins:
        return None
    return Event(event_id=f"{ev.event_id}@{cutoff}", inputs=ins, target=ev.target,
                 final_tta=ev.final_tta, n_cdm_total=ev.n_cdm_total,
                 first_tta=ev.first_tta, source=ev.source)


def augmentation(events: list[Event], columns: dict):
    """(parent_index, Bundle) of earlier-cutoff copies of trainable events."""
    aug, parent = [], []
    for i, ev in enumerate(events):
        if ev.final_tta >= INPUT_CUTOFF_DAYS:
            continue
        for c in AUG_CUTOFFS:
            t = truncate(ev, c)
            if t is not None and len(t.inputs) < len(ev.inputs):
                aug.append(t); parent.append(i)
    if not aug:
        return np.zeros(0, int), None
    return np.array(parent, int), Bundle(aug, columns)


def _ctx(bundle: Bundle, idx, track, seed, aug=None, aug_parent=None):
    extra = {"colindex": bundle.colindex, "colnames": bundle.columns[track],
             "y_aux": bundle.y_aux[idx]}
    if aug is not None and track == "benchmark" and len(aug_parent):
        pos = {int(g): i for i, g in enumerate(idx)}
        sel = np.array([int(p) in pos for p in aug_parent])
        if sel.any():
            local = np.array([pos[int(p)] for p in aug_parent[sel]], int)
            extra["aug"] = (aug.X["benchmark"][sel], aug.y[sel], aug.latest[sel], local)
    return C.Context(bundle.X[track][idx], bundle.y[idx], bundle.latest[idx],
                     bundle.strata[idx], seed, bundle.eligible[idx], extra)


def fit_predict_all(train: Bundle, tr_idx, test: Bundle, te_idx, seed, aug=None,
                    aug_parent=None, log=None):
    """Fit the whole pool on the trainable part of train[tr_idx]; predict
    test[te_idx]. Returns (reg: {name: pred}, ew: {name: score}, models)."""
    tr_idx = np.asarray(tr_idx)[train.trainable[tr_idx]]
    reg, ew, models = {}, {}, {}
    for cand in C.regression_pool():
        t0 = time.time()
        cand.fit(_ctx(train, tr_idx, cand.track, seed, aug, aug_parent))
        reg[cand.name] = cand.predict(test.X[cand.track][te_idx], test.latest[te_idx])
        models[cand.name] = cand
        if log:
            log(f"      {cand.name:<19} {time.time() - t0:6.1f}s")
    lat = test.latest[te_idx]
    ew["e_quant"] = models["gbm_quant"].prob_high(test.X["benchmark"][te_idx], lat)
    ew["e_hurdle"] = models["hurdle"].prob_high(test.X["benchmark"][te_idx], lat)
    for k in ("hurdle_eserving", "hurdle_eserving_aux"):
        ew[f"e_{k}"] = models[k].prob_high(test.X["eserving"][te_idx], lat)
    for name, (base, prevalence) in C.DERIVED.items():   # re-tuned decision, no refit
        d = models[base].with_prior(name, prevalence)
        reg[name] = d.predict(test.X[d.track][te_idx], test.latest[te_idx])
        models[name] = d
    for cand in C.early_warning_pool():
        cand.fit(_ctx(train, tr_idx, cand.track, seed))
        ew[cand.name] = cand.score(test.X[cand.track][te_idx], lat)
        models[cand.name] = cand
    return reg, ew, models


def _blend(ew: dict, mask: np.ndarray) -> np.ndarray:
    return np.mean([C.rank01(ew[k][mask]) for k in ("e_hurdle", "e_clf", "e_physics")], axis=0)


def _ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def _maxT(boot: dict, point: dict, level=0.95):
    """Simultaneous (max-T) intervals across candidates from shared-index
    bootstrap draws: point ± c* · sd, c* = level-quantile of max_c |t_c|."""
    names = [k for k in boot if np.isfinite(boot[k]).sum() > 2 and np.nanstd(boot[k]) > 0]
    if not names:
        return {}, None
    M = np.vstack([np.asarray(boot[k], float) for k in names])
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        t = np.nanmax(np.abs((M - mu) / sd), axis=0)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return {}, None
    c = float(np.quantile(t, level))
    return {k: [float(point[k] - c * sd[i, 0]), float(point[k] + c * sd[i, 0])]
            for i, k in enumerate(names)}, c


def _agg(rows, key):
    vals = np.array([row[key] for row in rows], float)
    return {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals))}


REG_NAMES = [c.name for c in C.regression_pool()] + list(C.DERIVED)
EW_NAMES = ["e_logit", "e_max", "e_physics", "e_quant", "e_hurdle", "e_clf",
            "e_rank_blend", "e_serving", "e_hurdle_eserving", "e_hurdle_eserving_aux"]
EW_SERVING = ("e_serving", "e_hurdle_eserving", "e_hurdle_eserving_aux")
EW_PROBABILISTIC = ("e_logit", "e_quant", "e_hurdle", "e_clf", "e_serving",
                    "e_hurdle_eserving", "e_hurdle_eserving_aux")


def _track_of(name):
    if name in C.DERIVED:
        name = C.DERIVED[name][0]
    for c in C.regression_pool():
        if c.name == name:
            return "serving" if c.track == "eserving" else c.track
    return "serving" if name in EW_SERVING else "benchmark"


def summarize(y_reg, oof_reg, y_ew, oof_ew, B, rng, budget):
    """Shared-index paired bootstrap over events, averaged over repeats.

    y_reg / oof_reg: truth and {name: [pred per repeat]} on the scored
    (eligible) regression events; y_ew / oof_ew: escalation labels and
    {name: [score per repeat]} on the scored early-warning events.
    """
    R = len(oof_reg["lrp"])
    per_rep = {k: [metric.official(oof_reg[k][r], y_reg) for r in range(R)] for k in oof_reg}

    def rel(L, base):
        return 100 * (L / base - 1) if np.isfinite(L) and np.isfinite(base) and base > 0 else np.nan

    point_dl = {k: float(np.nanmean([rel(per_rep[k][r]["L"], per_rep["lrp"][r]["L"])
                                     for r in range(R)])) for k in oof_reg}
    # the same predictions scored at the official test's published class mix
    # (label-shift importance weights): what CV says about a sealed look
    w_bench = metric.label_shift_weights(y_reg >= HIGH_RISK, C.BENCH_HR_PREVALENCE)
    bench_rep = {k: [metric.official(oof_reg[k][r], y_reg, weights=w_bench)["L"] for r in range(R)]
                 for k in oof_reg}
    ew_rep = {k: [metric.early_warning(y_ew, oof_ew[k][r], None, budget) for r in range(R)]
              for k in oof_ew}
    point_dap = {k: float(np.mean([ew_rep[k][r]["pr_auc"] - ew_rep["e_logit"][r]["pr_auc"]
                                   for r in range(R)])) for k in oof_ew}
    point_drb = {k: float(np.mean([ew_rep[k][r]["recall_at_budget"]
                                   - ew_rep["e_logit"][r]["recall_at_budget"]
                                   for r in range(R)])) for k in oof_ew}
    boot_dl = {k: np.full(B, np.nan) for k in oof_reg if k != "lrp"}
    boot_dap = {k: np.full(B, np.nan) for k in oof_ew if k != "e_logit"}
    boot_drb = {k: np.full(B, np.nan) for k in oof_ew if k != "e_logit"}
    n, m = len(y_reg), len(y_ew)
    for b in range(B):
        i = rng.randint(0, n, n)
        yi = y_reg[i]
        base = [metric.official(oof_reg["lrp"][r][i], yi)["L"] for r in range(R)]
        for k in boot_dl:
            vals = [rel(metric.official(oof_reg[k][r][i], yi)["L"], base[r]) for r in range(R)]
            if np.isfinite(vals).any():
                boot_dl[k][b] = np.nanmean(vals)
        if m == 0:
            continue
        j = rng.randint(0, m, m)
        yj = y_ew[j]
        if yj.sum() == 0:
            continue
        bap = [metric.average_precision(yj, oof_ew["e_logit"][r][j]) for r in range(R)]
        brb = [metric.recall_at_budget(yj, oof_ew["e_logit"][r][j], budget) for r in range(R)]
        for k in boot_dap:
            boot_dap[k][b] = np.mean([metric.average_precision(yj, oof_ew[k][r][j]) - bap[r]
                                      for r in range(R)])
            boot_drb[k][b] = np.mean([metric.recall_at_budget(yj, oof_ew[k][r][j], budget) - brb[r]
                                      for r in range(R)])
    adj_dl, c_reg = _maxT(boot_dl, point_dl)
    adj_dap, c_ew = _maxT(boot_dap, point_dap)
    reg = {}
    for k in oof_reg:
        rows = per_rep[k]
        s = {mk: _agg(rows, mk) for mk in ("L", "mse_hr", "f2", "precision", "recall")}
        if k == "lrp":
            s["delta_L_pct_vs_lrp"] = {"mean": 0.0, "ci95": [0.0, 0.0], "ci975": [0.0, 0.0],
                                       "ci95_adjusted": [0.0, 0.0], "sd_boot": 0.0, "mde": None}
        else:
            sd = float(np.nanstd(boot_dl[k]))
            s["delta_L_pct_vs_lrp"] = {"mean": point_dl[k], "ci95": _ci(boot_dl[k]),
                                       "ci975": _ci(boot_dl[k], 1.25, 98.75),
                                       "ci95_adjusted": adj_dl.get(k, [None, None]),
                                       "sd_boot": sd, "mde": 2.8 * sd,
                                       # replicates with an undefined ratio (F2 = 0) are
                                       # dropped from the CI; disclosed, not hidden
                                       "undefined_frac": float(np.mean(~np.isfinite(boot_dl[k])))}
        dlb = [rel(bench_rep[k][r], bench_rep["lrp"][r]) for r in range(R)]
        s["bench_mix"] = {"L": _agg([{"v": v} for v in bench_rep[k]], "v"),
                          "delta_L_pct": _agg([{"v": v} for v in dlb], "v")}
        s["track"] = _track_of(k)
        s["primary"] = k in C.PRIMARY.values()
        reg[k] = s
    ew = {}
    for k in oof_ew:
        rows = ew_rep[k]
        s = {mk: _agg(rows, mk) for mk in ("pr_auc", "roc_auc", "pauc_fpr10", "recall_at_budget")}
        if k != "e_logit":
            sd = float(np.nanstd(boot_dap[k]))
            s["delta_pr_auc_vs_logit"] = {"mean": point_dap[k], "ci95": _ci(boot_dap[k]),
                                          "ci975": _ci(boot_dap[k], 1.25, 98.75),
                                          "ci95_adjusted": adj_dap.get(k, [None, None]),
                                          "sd_boot": sd, "mde": 2.8 * sd}
            s["delta_recall_budget_vs_logit"] = {"mean": point_drb[k], "ci95": _ci(boot_drb[k]),
                                                 "sd_boot": float(np.nanstd(boot_drb[k]))}
        s["track"] = _track_of(k)
        s["primary"] = k in C.PRIMARY.values()
        ew[k] = s
    return reg, ew, {"maxT_c_regression": c_reg, "maxT_c_early_warning": c_ew}


def _calibration(y, p):
    if p is None or len(y) == 0 or np.sum(y) == 0:
        return None
    s, i = metric.calibration_slope_intercept(y, p)
    return {"slope": s, "intercept": i, "brier": metric.brier(y, p)}


def cross_validate(train_events: list[Event], repeats=5, folds=5, seed=42,
                   bootstrap=500, budget=0.05, augment=True, log=print) -> dict:
    from sklearn.model_selection import StratifiedKFold

    bundle = Bundle(train_events)
    aug_parent, aug = (augmentation(train_events, bundle.columns) if augment
                       else (np.zeros(0, int), None))
    n = len(train_events)
    elig = bundle.eligible
    ew_dom = bundle.cand & elig
    log(f"CV: {n} events ({int(bundle.trainable.sum())} trainable, {int(elig.sum())} test-eligible "
        f"with {int((bundle.y[elig] >= HIGH_RISK).sum())} high-risk; {int(ew_dom.sum())} eligible "
        f"escalation candidates with {int(bundle.esc[ew_dom].sum())} escalations); "
        f"{repeats}x{folds} folds, {len(aug.events) if aug else 0} augmentation rows")
    oof_reg = {k: [] for k in REG_NAMES}
    oof_ew = {k: [] for k in EW_NAMES}
    for r in range(repeats):
        o_reg = {k: np.full(n, np.nan) for k in REG_NAMES}
        o_ew = {k: np.full(n, np.nan) for k in EW_NAMES}
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), bundle.strata)):
            t0 = time.time()
            reg, ew, _ = fit_predict_all(bundle, tr, bundle, te, seed + 1000 * r + f, aug, aug_parent)
            for k, v in reg.items():
                o_reg[k][te] = v
            for k, v in ew.items():
                o_ew[k][te] = v
            log(f"  repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")
        o_ew["e_rank_blend"][ew_dom] = _blend(o_ew, ew_dom)
        for k in REG_NAMES:
            oof_reg[k].append(o_reg[k][elig])
        for k in EW_NAMES:
            oof_ew[k].append(o_ew[k][ew_dom])
    rng = np.random.RandomState(seed + 7)
    y_ew = bundle.esc[ew_dom]
    reg, ew, extra = summarize(bundle.y[elig], oof_reg, y_ew, oof_ew, bootstrap, rng, budget)
    for k in EW_PROBABILISTIC:
        cal = [_calibration(y_ew, oof_ew[k][r]) for r in range(repeats)]
        cal = [c for c in cal if c]
        if cal:
            ew[k]["cal_slope"] = {"mean": float(np.nanmean([c["slope"] for c in cal]))}
            ew[k]["brier"] = {"mean": float(np.nanmean([c["brier"] for c in cal]))}
    return {
        "n_events": n, "n_trainable": int(bundle.trainable.sum()),
        "n_eligible": int(elig.sum()), "n_hr_eligible": int((bundle.y[elig] >= HIGH_RISK).sum()),
        "n_candidates": int(ew_dom.sum()), "n_escalations": int(y_ew.sum()),
        "n_candidates_all": int(bundle.cand.sum()), "n_escalations_all": int(bundle.esc.sum()),
        "repeats": repeats, "folds": folds, "bootstrap": bootstrap, "budget": budget,
        "augmented_rows": int(len(aug.events)) if aug else 0,
        "regression": reg, "early_warning": ew, "multiplicity": extra,
    }


def sealed_test(train_events, test_events, seed=42, budget=0.05, augment=True,
                bootstrap=1000, log=print) -> dict:
    """Fit every candidate on all trainable training events; score the
    official test set once, with a paired bootstrap on the test events."""
    tr = Bundle(train_events)
    te = Bundle(test_events, tr.columns)
    aug_parent, aug = (augmentation(train_events, tr.columns) if augment
                       else (np.zeros(0, int), None))
    reg, ew, _ = fit_predict_all(tr, np.arange(len(train_events)), te,
                                 np.arange(len(test_events)), seed, aug, aug_parent, log=log)
    elig = te.eligible
    ew_dom = te.cand & elig
    ew["e_rank_blend"] = np.full(len(test_events), np.nan)
    ew["e_rank_blend"][ew_dom] = _blend(ew, ew_dom)
    rng = np.random.RandomState(seed + 99)
    rsum, esum, extra = summarize(te.y[elig], {k: [v[elig]] for k, v in reg.items()},
                                  te.esc[ew_dom], {k: [v[ew_dom]] for k, v in ew.items()},
                                  bootstrap, rng, budget)
    lrp = rsum["lrp"]
    fidelity = {k: {"lab": lrp[k]["mean"], "published": PUBLISHED_LRP[k],
                    "abs_diff": abs(lrp[k]["mean"] - PUBLISHED_LRP[k])} for k in PUBLISHED_LRP}
    fidelity["reproduces_published"] = bool(all(fidelity[k]["abs_diff"] <= 0.02 for k in PUBLISHED_LRP))
    return {"n_events": len(test_events), "n_eligible": int(elig.sum()),
            "n_hr": int((te.y[elig] >= HIGH_RISK).sum()),
            "n_candidates": int(ew_dom.sum()), "n_escalations": int(te.esc[ew_dom].sum()),
            "regression": rsum, "early_warning": esum, "multiplicity": extra,
            "lrp_fidelity": fidelity}


def ceiling(train_events, repeats=1, folds=5, seed=42, log=print) -> dict:
    """NOT ADMISSIBLE - diagnostic only. The primary model, allowed to also
    see CDMs between 2 and 1 day before TCA. The gap to the admissible run is
    the part of the outcome decided by late tracking updates."""
    from sklearn.model_selection import StratifiedKFold

    late_events = []
    for ev in train_events:
        extra = [r for r in ev.late if 1.0 <= float(r["time_to_tca"]) and
                 float(r["time_to_tca"]) > ev.final_tta]
        late_events.append(Event(ev.event_id + "@late", ev.inputs + extra, ev.target,
                                 ev.final_tta, ev.n_cdm_total, ev.first_tta, ev.source))
    res = {}
    for label, evs in (("admissible", train_events), ("with_late_cdms", late_events)):
        b = Bundle(evs)
        oof, base = np.full(len(evs), np.nan), np.full(len(evs), np.nan)
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
            for tr, te in skf.split(np.zeros(len(evs)), b.strata):
                tr = tr[b.trainable[tr]]
                m = C.Hurdle("hurdle_lrp", mode="lrp")
                m.fit(_ctx(b, tr, "benchmark", seed))
                oof[te] = m.predict(b.X["benchmark"][te], b.latest[te])
                base[te] = metric.lrp(b.latest[te])
        el = b.eligible
        res[label] = {"model": metric.official(oof[el], b.y[el]),
                      "lrp": metric.official(base[el], b.y[el])}
        log(f"  ceiling[{label}]: L={res[label]['model']['L']:.3f} (lrp {res[label]['lrp']['L']:.3f})")
    return res
