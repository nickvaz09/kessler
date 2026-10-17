"""Round 4: is the CDM risk a coherent forecast of itself, and does early
warning work from public-data counterparts?

Pre-registered in docs/v2/vigil-lab.md ("Round 4 pre-registration", commit
2ebc3e3) before any confirmatory computation. The sealed test stays spent:
nothing here trains a model on the training file and scores it on the
official test as a held-out set.

    coherence  P6     coherence of the CDM risk (test events; no model)
    public     P7-P8  early warning from public-data counterparts, and VIGIL's
                      classifier as a probability forecast (pooled CV)
    posthoc    P6     post-hoc checks added after the results (not pre-registered):
                      the update test in the encounter plane, which corrects an
                      inflated P6(c), and two robustness checks on P6(a)
    ewposthoc  P7-P8  post-hoc check (not pre-registered): the latest-risk
                      baselines ranked by the latest risk itself, without the
                      pooled per-fold fits of the registered baselines

    .venv/bin/python -m backend.app.models.lab.round4 coherence --out reports/paper

Each subcommand writes ``<out>/<name>.json`` (posthoc: coherence_posthoc.json).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import data, metric, protocol
from backend.app.models.lab.data import HIGH_RISK, Event
from backend.app.models.lab.paper import DEFAULTS, KELVINS, _write
from backend.app.models.lab.physics import covariance_rtn

RAW = f"{KELVINS}/Collision Avoidance Challenge - Dataset/raw_data/raw_data_2015-2019.gz"
MU_KM3_S2 = 398600.4418
CHI2_3_P99 = 11.344867      # 99th percentile of chi-squared with 3 degrees of freedom
CHI2_2_P99 = 9.210340       # ... and with 2 (-2 ln 0.01)
MAX_TARGET_ECC = 0.1        # the frame rotation assumes a near-circular target orbit
# The organisers' outcome-based split (Uriot et al. 2022): of the eligible
# events, 150 of 216 high-risk and 2,017 of 10,244 others went to the test.
S_HR, S_OTHER = 150 / 216, 2017 / 10244
RHO = S_HR / S_OTHER
RTN = ("r", "t", "n")


def _num(row: dict, key: str) -> float:
    try:
        v = float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _vec(row: dict, prefix: str) -> np.ndarray:
    return np.array([_num(row, f"{prefix}_{a}") for a in RTN])


# --- P6: physics with the chaser's covariance in the target's frame ------------

def chaser_to_target(row: dict) -> np.ndarray | None:
    """3x3 rotation taking chaser-RTN vectors to target-RTN at TCA.

    CCSDS CDMs give each object's covariance in its own RTN frame. At TCA the
    objects are metres apart, so their radial axes coincide; the frames differ
    by a rotation about R that aligns each along-track axis with that object's
    horizontal velocity. The target's velocity is taken as (0, sqrt(mu/a), 0),
    which holds for a near-circular orbit; the chaser's is that plus the
    relative velocity (object 2 minus object 1 in the target's frame)."""
    a, e = _num(row, "t_j2k_sma"), _num(row, "t_j2k_ecc")
    vrel = _vec(row, "relative_velocity")
    if not (np.isfinite(a) and a > 6000.0 and np.all(np.isfinite(vrel))):
        return None
    if not np.isfinite(e) or e > MAX_TARGET_ECC:
        return None
    v_c = np.array([0.0, 1000.0 * math.sqrt(MU_KM3_S2 / a), 0.0]) + vrel
    horiz = np.array([0.0, v_c[1], v_c[2]])
    norm = np.linalg.norm(horiz)
    if norm <= 0:
        return None
    t_axis = horiz / norm
    r_axis = np.array([1.0, 0.0, 0.0])
    return np.column_stack([r_axis, t_axis, np.cross(r_axis, t_axis)])


def combined_covariance(row: dict, rotate: bool = True) -> np.ndarray | None:
    """Combined relative-position covariance [m^2] in the target's RTN frame."""
    ct, cc = covariance_rtn(row, "t"), covariance_rtn(row, "c")
    if ct is None or cc is None:
        return None
    if rotate:
        m = chaser_to_target(row)
        if m is None:
            return None
        cc = m @ cc @ m.T
    return ct + cc


_GR, _WR = np.polynomial.legendre.leggauss(16)
_GT, _WT = np.polynomial.legendre.leggauss(32)


def pc_disk(x: np.ndarray, c2: np.ndarray, hbr: float) -> float:
    """Integral of the 2-D Gaussian N(x, c2) over the disk of radius hbr at the
    origin (polar Gauss-Legendre quadrature)."""
    rho = 0.5 * hbr * (_GR + 1.0)
    th = math.pi * (_GT + 1.0)
    rr, tt = np.meshgrid(rho, th, indexing="ij")
    w = np.outer(_WR * 0.5 * hbr, _WT * math.pi) * rr
    p = np.stack([rr * np.cos(tt) - x[0], rr * np.sin(tt) - x[1]], -1)
    q = np.einsum("...i,ij,...j->...", p, np.linalg.inv(c2), p)
    return float(np.sum(w * np.exp(-0.5 * q)) / (2.0 * math.pi * math.sqrt(np.linalg.det(c2))))


def recomputed_log10_pc(row: dict, rotate: bool = True) -> float:
    """log10 Pc from the CDM's miss vector, covariances and object spans
    (hard-body radius = the mean of the two spans), in the encounter plane."""
    r, v = _vec(row, "relative_position"), _vec(row, "relative_velocity")
    c = combined_covariance(row, rotate)
    ts, cs = _num(row, "t_span"), _num(row, "c_span")
    if c is None or not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.isfinite(ts)
                         and np.isfinite(cs)) or np.linalg.norm(v) <= 0:
        return float("nan")
    vh = v / np.linalg.norm(v)
    u1 = r - np.dot(r, vh) * vh
    if np.linalg.norm(u1) < 1e-9:
        u1 = np.cross(vh, [0.0, 0.0, 1.0])
    u1 = u1 / np.linalg.norm(u1)
    b = np.vstack([u1, np.cross(vh, u1)])
    c2 = b @ c @ b.T
    if np.linalg.det(c2) <= 0:
        return float("nan")
    pc = pc_disk(b @ r, c2, 0.5 * (ts + cs))
    return math.log10(pc) if pc > 0 else float("nan")


def update_d2(rows: list[dict]) -> list[float]:
    """For each consecutive pair of CDMs (in message order), the squared
    Mahalanobis length of the change in the predicted relative position at
    TCA, in the earlier message's rotated combined covariance."""
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        c = combined_covariance(a, rotate=True)
        xa, xb = _vec(a, "relative_position"), _vec(b, "relative_position")
        if c is None or not (np.all(np.isfinite(xa)) and np.all(np.isfinite(xb))):
            continue
        dx = xb - xa
        out.append(float(dx @ np.linalg.solve(c, dx)))
    return out


def plane_basis(v: np.ndarray) -> np.ndarray:
    """2x3 orthonormal basis of the encounter plane (perpendicular to v)."""
    vh = v / np.linalg.norm(v)
    u1 = np.cross(vh, [1.0, 0.0, 0.0])
    if np.linalg.norm(u1) < 1e-6:
        u1 = np.cross(vh, [0.0, 1.0, 0.0])
    u1 = u1 / np.linalg.norm(u1)
    return np.vstack([u1, np.cross(vh, u1)])


def update_d2_plane(rows: list[dict]) -> list[float]:
    """Post hoc (not pre-registered): update_d2 in the earlier message's
    encounter plane, with the rotated combined covariance projected onto it.

    Each CDM gives the relative position at TCA, which is perpendicular to the
    relative velocity: an error along the direction of motion shows up as a
    change of TCA, not of that position. So the change between messages lies
    in the encounter plane, and for a vector in that plane the 3-D Mahalanobis
    length of update_d2 is at least the planar one, and much larger when the
    covariance correlates the plane with the direction of motion. For a
    coherent sequence this d^2 is no larger, in distribution, than chi-squared
    with 2 degrees of freedom."""
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        c = combined_covariance(a, rotate=True)
        xa, xb = _vec(a, "relative_position"), _vec(b, "relative_position")
        v = _vec(a, "relative_velocity")
        if c is None or not (np.all(np.isfinite(xa)) and np.all(np.isfinite(xb)) and np.all(np.isfinite(v))):
            continue
        if np.linalg.norm(v) <= 0:
            continue
        m = plane_basis(v)
        p = m @ (xb - xa)
        out.append(float(p @ np.linalg.solve(m @ c @ m.T, p)))
    return out


def chernoff_p(k: int, mean: float) -> float:
    """Upper bound on P(N >= k) for a sum N of independent indicators whose
    expectation is at most ``mean``: exp(-mean) * (e * mean / k)^k for k > mean."""
    if k <= mean:
        return 1.0
    if mean <= 0:
        return 0.0
    return float(math.exp(-mean + k * (1.0 + math.log(mean) - math.log(k))))


def coherent_bound(latest: np.ndarray, rho: float = RHO) -> np.ndarray:
    """Most a coherent Pc allows for P(a later Pc >= 1e-6), per event, inside
    a sample that over-represents high-risk outcomes by the odds factor rho."""
    return np.minimum(1.0, rho * 10.0 ** (np.asarray(latest, float) - HIGH_RISK))


def match_raw(test_df, targets_df, raw_df) -> dict:
    """{test event_id: raw rows after the cutoff (message order)} for every test
    event whose pre-cutoff CDMs all appear in exactly one raw event whose final
    CDM equals the published final CDM; unmatched events are absent."""
    key = ["mission_id", "time_to_tca", "risk", "miss_distance"]

    def keys(df):
        return list(zip(*[df[c].astype(float).round(9) for c in key]))

    raw = raw_df.copy()
    raw["_k"] = keys(raw)
    index: dict = {}
    for eid, ks in raw.groupby("event_id")["_k"]:
        for k in ks:
            index.setdefault(k, set()).add(eid)
    finals = targets_df.set_index("event_id")
    tcol = "true_risk" if "true_risk" in finals.columns else "risk"
    groups = {eid: g for eid, g in raw.groupby("event_id")}
    t = test_df.copy()
    t["_k"] = keys(t)
    out = {}
    for eid, g in t.groupby("event_id"):
        hits = set.intersection(*[index.get(k, set()) for k in g["_k"]])
        if len(hits) != 1:
            continue
        rg = groups[next(iter(hits))].sort_values("time_to_tca", ascending=False, kind="mergesort")
        last = rg.iloc[-1]
        f = finals.loc[eid]
        if not (abs(float(last["risk"]) - float(f[tcol])) < 1e-6
                and abs(float(last["time_to_tca"]) - float(f["time_to_tca"])) < 1e-6):
            continue
        late = rg[rg["time_to_tca"] < data.INPUT_CUTOFF_DAYS]
        out[eid] = late.to_dict("records")
    return out


def coherence(test_events: list[Event], late: dict, bootstrap: int = 2000, seed: int = 42) -> dict:
    """P6 (a)-(d) on the official test events."""
    rng = np.random.RandomState(seed)
    ids = [int(e.event_id.split(":")[1]) for e in test_events]
    r = np.array([e.latest_risk for e in test_events])
    y = np.array([e.target for e in test_events])
    matched = np.array([i in late for i in ids])
    cross = np.array([max([y_i] + [float(c["risk"]) for c in late.get(i, [])]) >= HIGH_RISK
                      for i, y_i in zip(ids, y)])
    final_hr = y >= HIGH_RISK
    bound = coherent_bound(r)
    cand = r < HIGH_RISK

    def band(lo, hi, use_cross, sentinel=False):
        m = cand & ((r <= -29.9) if sentinel else ((r > lo) & (r < hi)))
        if use_cross:
            m = m & matched
        hit = cross if use_cross else final_hr
        k, mean = int(hit[m].sum()), float(bound[m].sum())
        return {"n": int(m.sum()), "observed": k, "coherent_max": mean,
                "ratio": (k / mean) if mean > 0 else None, "p_chernoff": chernoff_p(k, mean)}

    low = {"primary_any_later_cdm": band(-29.9, -8.0, True),
           "final_cdm_only": band(-29.9, -8.0, False),
           "band_-29.9_-12_any_later": band(-29.9, -12.0, True),
           "band_-12_-8_any_later": band(-12.0, -8.0, True),
           "band_-8_-6_any_later": band(-8.0, HIGH_RISK, True),
           "sentinel_-30_any_later": band(0, 0, True, sentinel=True)}
    low["reading"] = ("incoherent" if low["primary_any_later_cdm"]["p_chernoff"] < 1e-3
                      else "not shown")

    # (b) the high end: weighted mean later Pc over weighted mean latest Pc
    w = np.where(final_hr, 1.0 / S_HR, 1.0 / S_OTHER)
    calls = (r >= HIGH_RISK) & (r < -4.0)

    def ratio(m, later):
        i = np.flatnonzero(m)
        p0, p1 = 10.0 ** r, 10.0 ** later
        est = float(np.sum(w[i] * p1[i]) / np.sum(w[i] * p0[i]))
        boot = []
        for _ in range(bootstrap):
            j = rng.choice(i, len(i))
            boot.append(np.sum(w[j] * p1[j]) / np.sum(w[j] * p0[j]))
        return {"n": int(len(i)), "ratio": est, "ci95": protocol._ci(np.array(boot))}

    nxt = np.array([float(late[i][0]["risk"]) if late.get(i) else np.nan for i in ids])
    high = {"final": ratio(calls, y), "next_cdm": ratio(calls & np.isfinite(nxt), np.nan_to_num(nxt, nan=-30.0))}
    high["final"]["share_still_high_risk_weighted"] = float(np.sum(w[calls] * final_hr[calls]) / np.sum(w[calls]))
    hi_ci = high["final"]["ci95"]
    high["reading"] = "drifts down" if hi_ci[1] is not None and hi_ci[1] < 1.0 else "not shown"

    # (c) update realism over the pre-cutoff sequences
    per_event = [np.array(update_d2(e.inputs)) for e in test_events]
    allv = np.concatenate([d for d in per_event if len(d)]) if per_event else np.zeros(0)
    over = [(int(np.sum(d > CHI2_3_P99)), len(d)) for d in per_event]
    k_over = np.array([o[0] for o in over]); n_pairs = np.array([o[1] for o in over])
    boot = []
    for _ in range(bootstrap):
        j = rng.randint(0, len(over), len(over))
        boot.append(k_over[j].sum() / max(1, n_pairs[j].sum()))
    share = float(k_over.sum() / max(1, n_pairs.sum()))
    ci = protocol._ci(np.array(boot))
    updates = {"pairs": int(n_pairs.sum()), "events": int(np.sum(n_pairs > 0)),
               "median_d2": float(np.median(allv)) if len(allv) else None,
               "share_above_chi2_3_p99": share, "ci95": ci,
               "reading": "unrealistic" if ci[0] is not None and ci[0] > 0.01 else "not shown"}

    # (d) methods check: ESA's risk against the recomputed Pc
    errs = {"unrotated": [], "rotated": []}
    for e in test_events:
        for row in e.inputs:
            risk = _num(row, "risk")
            if not (np.isfinite(risk) and risk > -20.0):
                continue
            a, b = recomputed_log10_pc(row, rotate=False), recomputed_log10_pc(row, rotate=True)
            if np.isfinite(a) and np.isfinite(b):
                errs["unrotated"].append(abs(a - risk)); errs["rotated"].append(abs(b - risk))
    check = {k: {"n": len(v), "median_abs_dlog10": float(np.median(v)) if v else None,
                 "share_within_0.1": float(np.mean(np.array(v) < 0.1)) if v else None}
             for k, v in errs.items()}

    return {"analysis": "P6 coherence of the CDM risk (official test events, no model)",
            "n_events": len(test_events), "n_matched_raw": int(matched.sum()),
            "selection": {"s_high_risk": S_HR, "s_other": S_OTHER, "rho": RHO},
            "bootstrap": bootstrap, "seed": seed,
            "a_low_end": low, "b_high_end": high, "c_updates": updates, "d_recompute_check": check}


def covariance_rtn_clip(row: dict, prefix: str, clip: float | None) -> np.ndarray | None:
    """physics.covariance_rtn with a chosen correlation clip (None: no clipping),
    for the sensitivity of P6(c) to near-singular covariances; clip=0.999
    reproduces covariance_rtn."""
    s = np.array([_num(row, f"{prefix}_sigma_{a}") for a in RTN])
    if not np.all(np.isfinite(s)) or s.min() <= 0:
        return None
    rho = [_num(row, f"{prefix}_{k}") for k in ("ct_r", "cn_r", "cn_t")]
    rho = [0.0 if not np.isfinite(x) else (float(np.clip(x, -clip, clip)) if clip else x) for x in rho]
    corr = np.array([[1.0, rho[0], rho[1]], [rho[0], 1.0, rho[2]], [rho[1], rho[2], 1.0]])
    w, V = np.linalg.eigh(corr * np.outer(s, s))
    return (V * np.clip(w, 1e-6, None)) @ V.T


def match_raw_tol(test_df, targets_df, raw_df, tol: float = 1e-9) -> dict:
    """match_raw with tolerances instead of rounded keys (rounding can split two
    copies of one value that straddle a rounding boundary): the same rule,
    exactly one raw event holding every pre-cutoff CDM, whose final CDM equals
    the published final CDM."""
    by_mission = {m: g.sort_values("time_to_tca", kind="mergesort") for m, g in raw_df.groupby("mission_id")}
    finals = targets_df.set_index("event_id")
    tcol = "true_risk" if "true_risk" in finals.columns else "risk"
    groups = {eid: g.sort_values("time_to_tca", ascending=False, kind="mergesort") for eid, g in raw_df.groupby("event_id")}

    def holders(row):
        g = by_mission.get(row.mission_id)
        if g is None:
            return set()
        t = g["time_to_tca"].to_numpy()
        lo = np.searchsorted(t, row.time_to_tca - tol)
        hi = np.searchsorted(t, row.time_to_tca + tol, side="right")
        c = g.iloc[lo:hi]
        c = c[(np.abs(c["risk"] - row.risk) < tol) & (np.abs(c["miss_distance"] - row.miss_distance) < 1e-6)]
        return set(c["event_id"].tolist())

    out = {}
    for eid, g in test_df.groupby("event_id"):
        hits = set.intersection(*[holders(row) for row in g.itertuples(index=False)])
        if len(hits) != 1:
            continue
        rg = groups[next(iter(hits))]
        last, f = rg.iloc[-1], finals.loc[eid]
        if not (abs(float(last["risk"]) - float(f[tcol])) < 1e-6
                and abs(float(last["time_to_tca"]) - float(f["time_to_tca"])) < 1e-6):
            continue
        out[eid] = rg[rg["time_to_tca"] < data.INPUT_CUTOFF_DAYS].to_dict("records")
    return out


def posthoc(test_events: list[Event], late: dict, train_events: list[Event], bootstrap: int = 2000,
            seed: int = 42, late_tol: dict | None = None) -> dict:
    """Post-hoc checks on P6, added after the Round 4 results were recorded
    (docs/v2/vigil-lab.md, "Round 4 correction"). Not pre-registered.

    c_updates_plane  P6(c) in the encounter plane against chi-squared(2), which
                     corrects the inflated registered statistic
    a_robustness     for P6(a): the missions behind the crossings, the bound
                     without independence, and how often training candidates
                     pass the test filters, which depend on later messages"""
    rng = np.random.RandomState(seed)
    d3, d2, over, cosv = [], [], [], []
    for e in test_events:
        k = n = 0
        for a, b in zip(e.inputs[:-1], e.inputs[1:]):
            full, plane = update_d2([a, b]), update_d2_plane([a, b])
            if not (full and plane):
                continue
            d3.append(full[0])
            d2.append(plane[0])
            k += int(plane[0] > CHI2_2_P99)
            n += 1
        over.append((k, n))
        for row in e.inputs:
            x, v = _vec(row, "relative_position"), _vec(row, "relative_velocity")
            if np.all(np.isfinite(x)) and np.all(np.isfinite(v)) and np.linalg.norm(x) > 0 and np.linalg.norm(v) > 0:
                cosv.append(abs(float(x @ v)) / (np.linalg.norm(x) * np.linalg.norm(v)))
    d3, d2 = np.array(d3), np.array(d2)
    k_over = np.array([o[0] for o in over])
    n_pairs = np.array([o[1] for o in over])
    boot = []
    for _ in range(bootstrap):
        j = rng.randint(0, len(over), len(over))
        boot.append(k_over[j].sum() / max(1, n_pairs[j].sum()))
    pos = d2 > 0
    updates = {"pairs": int(n_pairs.sum()), "events": int(np.sum(n_pairs > 0)),
               "median_d2": float(np.median(d2)), "median_chi2_2": 2.0 * math.log(2.0),
               "share_above_chi2_2_p99": float(k_over.sum() / max(1, n_pairs.sum())),
               "ci95": protocol._ci(np.array(boot)),
               "registered_share_above_chi2_3_p99": float(np.mean(d3 > CHI2_3_P99)),
               "median_ratio_registered_to_plane": float(np.median(d3[pos] / d2[pos])),
               "median_abs_cos_position_velocity": float(np.median(cosv)),
               "n_positions": len(cosv)}

    # (a) robustness: the primary band, as in coherence()
    ids = [int(e.event_id.split(":")[1]) for e in test_events]
    r = np.array([e.latest_risk for e in test_events])
    band = (r > -29.9) & (r < -8.0) & np.array([i in late for i in ids])
    cross = np.array([max([e.target] + [float(c["risk"]) for c in late.get(i, [])]) >= HIGH_RISK
                      for e, i in zip(test_events, ids)])
    hits = np.flatnonzero(band & cross)
    missions = sorted({int(float(test_events[i].inputs[-1]["mission_id"])) for i in hits})
    mean = float(coherent_bound(r[band]).sum())
    rt = np.array([e.latest_risk for e in train_events])
    el = np.array([bool(e.test_eligible) for e in train_events])

    def share(m):
        return {"n": int(m.sum()), "share_eligible": float(el[m].mean()) if m.any() else None}

    robust = {"crossings": int(len(hits)), "coherent_max": mean, "missions": missions,
              "n_missions": len(missions),
              "markov_bound_without_independence": min(1.0, mean / len(hits)) if len(hits) else None,
              "training_candidates_passing_test_filters": {
                  "band_-29.9_-12": share((rt > -29.9) & (rt < -12.0)),
                  "band_-12_-8": share((rt > -12.0) & (rt < -8.0)),
                  "band_-29.9_-8": share((rt > -29.9) & (rt < -8.0))}}
    return {"analysis": "P6 post-hoc checks (not pre-registered): the update test in the encounter plane, "
                        "and robustness of the low-end test",
            "n_events": len(test_events), "bootstrap": bootstrap, "seed": seed,
            "c_updates_plane": updates, "a_robustness": robust,
            "c_regularization": _plane_share_by_clip(test_events),
            "d_esa_mahalanobis": _esa_mahalanobis(test_events),
            "a_tolerance_matching": _tolerance_bands(test_events, late_tol) if late_tol is not None else None}


def _plane_share_by_clip(test_events: list[Event]) -> dict:
    """The encounter-plane share of P6(c) under other correlation clips."""
    out = {}
    for clip in (None, 0.9999, 0.999, 0.99):
        d2, n_corr, n_beyond = [], 0, 0
        for e in test_events:
            for a, b in zip(e.inputs[:-1], e.inputs[1:]):
                for o in "tc":
                    for k in ("ct_r", "cn_r", "cn_t"):
                        v = _num(a, f"{o}_{k}")
                        if np.isfinite(v):
                            n_corr += 1
                            n_beyond += int(abs(v) > 0.999)
                ct, cc, m = covariance_rtn_clip(a, "t", clip), covariance_rtn_clip(a, "c", clip), chaser_to_target(a)
                xa, xb, v = _vec(a, "relative_position"), _vec(b, "relative_position"), _vec(a, "relative_velocity")
                if ct is None or cc is None or m is None or not (
                        np.all(np.isfinite(xa)) and np.all(np.isfinite(xb)) and np.all(np.isfinite(v))):
                    continue
                c = ct + m @ cc @ m.T
                pb = plane_basis(v)
                p = pb @ (xb - xa)
                d2.append(float(p @ np.linalg.solve(pb @ c @ pb.T, p)))
        out["none" if clip is None else str(clip)] = {"pairs": len(d2), "share_above_chi2_2_p99": float(np.mean(np.array(d2) > CHI2_2_P99))}
    out["share_of_correlations_beyond_0.999"] = n_beyond / max(1, n_corr)
    return out


def _esa_mahalanobis(test_events: list[Event]) -> dict:
    """ESA's mahalanobis_distance column against the 3-D Mahalanobis length of
    the miss vector in the combined covariance, rotated and unrotated."""
    ratios = {"rotated": [], "unrotated": []}
    for e in test_events:
        for row in e.inputs:
            esa, x = _num(row, "mahalanobis_distance"), _vec(row, "relative_position")
            if not (np.isfinite(esa) and esa > 0 and np.all(np.isfinite(x))):
                continue
            c_rot, c_un = combined_covariance(row, True), combined_covariance(row, False)
            if c_rot is None or c_un is None:
                continue
            ratios["rotated"].append(math.sqrt(x @ np.linalg.solve(c_rot, x)) / esa)
            ratios["unrotated"].append(math.sqrt(x @ np.linalg.solve(c_un, x)) / esa)
    return {k: {"n": len(v), "share_within_1pct": float(np.mean(np.abs(np.array(v) - 1) < 0.01)),
                "median_abs_dlog10": float(np.median(np.abs(np.log10(v))))} for k, v in ratios.items()}


def _tolerance_bands(test_events: list[Event], late: dict) -> dict:
    """P6(a)'s counts with match_raw_tol."""
    ids = [int(e.event_id.split(":")[1]) for e in test_events]
    r = np.array([e.latest_risk for e in test_events])
    matched = np.array([i in late for i in ids])
    cross = np.array([max([e.target] + [float(c["risk"]) for c in late.get(i, [])]) >= HIGH_RISK
                      for e, i in zip(test_events, ids)])
    bound = coherent_bound(r)
    cand = r < HIGH_RISK

    def band(m):
        m = m & matched & cand
        return {"n": int(m.sum()), "observed": int(cross[m].sum()), "coherent_max": float(bound[m].sum())}

    return {"n_matched": int(matched.sum()), "primary_any_later_cdm": band((r > -29.9) & (r < -8.0)),
            "band_-29.9_-12_any_later": band((r > -29.9) & (r < -12.0)), "band_-12_-8_any_later": band((r > -12.0) & (r < -8.0)),
            "band_-8_-6_any_later": band((r > -8.0) & (r < HIGH_RISK)), "sentinel_-30_any_later": band(r <= -29.9)}


# --- P7-P8: early warning from public-data counterparts ------------------------

NOT_PUBLIC_HISTORY = {"h_log_sigma_rate", "h_c_log_sigma_rate", "h_sigma_shrink_to_eval",
                      "h_maha_slope", "h_headroom_mre"}
PUBLIC_LAST = ("l_time_to_tca", "l_risk", "l_miss_distance", "l_relative_speed",
               "l_relative_velocity_r", "l_relative_velocity_t", "l_relative_velocity_n",
               "l_geocentric_latitude", "l_azimuth", "l_elevation",
               "l_t_j2k_sma", "l_t_j2k_ecc", "l_t_j2k_inc", "l_t_h_apo", "l_t_h_per",
               "l_c_j2k_sma", "l_c_j2k_ecc", "l_c_j2k_inc", "l_c_h_apo", "l_c_h_per",
               "l_t_cd_area_over_mass", "l_c_cd_area_over_mass",
               "l_F10", "l_F3M", "l_SSN", "l_AP")
RCS = ("l_t_rcs_estimate", "l_c_rcs_estimate")   # only as the catalogue's size class


def public_columns(columns: list[str]) -> list[str]:
    """The pre-registered public-counterpart rule (without the RCS classes)."""
    return [c for c in columns if c.startswith("s_") or c.startswith("l_ctype_")
            or c in PUBLIC_LAST or (c.startswith("h_") and c not in NOT_PUBLIC_HISTORY)]


def rcs_class(x) -> np.ndarray:
    """Public catalogue RCS size: small < 0.1 m^2 <= medium < 1 m^2 <= large."""
    x = np.asarray(x, float)
    return np.where(np.isnan(x), np.nan, np.where(x < 0.1, 0.0, np.where(x < 1.0, 1.0, 2.0)))


def public_matrix(X: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = {c: i for i, c in enumerate(columns)}
    cols = public_columns(columns)
    mats = [X[:, idx[c]] for c in cols] + [rcs_class(X[:, idx[c]]) for c in RCS if c in idx]
    return np.column_stack(mats), cols + [f"{c}_class" for c in RCS if c in idx]


def calibrated_baseline(latest_tr, y_tr):
    """P8's baseline: P(final >= -6) from the latest risk alone, calibrated."""
    from sklearn.linear_model import LogisticRegression

    def design(lat):
        lat = np.asarray(lat, float)
        return np.c_[lat, (lat >= HIGH_RISK).astype(float), np.maximum(lat, -12.0)]

    m = LogisticRegression(C=10.0, max_iter=2000).fit(design(latest_tr), np.asarray(y_tr) >= HIGH_RISK)
    return lambda lat: m.predict_proba(design(lat))[:, 1]


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fa_removed(stay, score, keep=0.95):
    """Share of false alarms (calls that fade) scored below the threshold that
    keeps ``keep`` of the true calls."""
    s = np.sort(score[stay])[::-1]
    thr = s[int(math.ceil(keep * len(s))) - 1]
    return float(np.mean(score[~stay] < thr))


def public(events: list[Event], repeats: int = 5, folds: int = 5, seed: int = 42,
           bootstrap: int = 1000, log=print) -> dict:
    """P7 and P8: the folds and seeds of paper.ablation (P4)."""
    from sklearn.model_selection import StratifiedKFold

    b = protocol.Bundle(events)
    cols = b.columns["benchmark"]
    xb = b.X["benchmark"]
    xpub, pub_cols = public_matrix(xb, cols)
    variants = {"A0": (xb, cols), "PUB": (xpub, pub_cols),
                "LIVE": (b.X["serving"], b.columns["serving"])}
    n = len(events)
    names = list(variants) + ["e_logit", "calibrated"]
    oof = {k: np.full((repeats, n), np.nan) for k in names}
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), b.strata)):
            t0 = time.time()
            tr = tr[b.trainable[tr]]
            s = seed + 1000 * r + f
            for k, (X, vc) in variants.items():
                ctx = C.Context(X[tr], b.y[tr], b.latest[tr], b.strata[tr], s, b.eligible[tr],
                                {"colindex": {c: i for i, c in enumerate(vc)}, "colnames": vc,
                                 "y_aux": b.y_aux[tr]})
                h = C.Hurdle(f"hurdle_{k}")
                h.fit(ctx)
                oof[k][r, te] = h.prob_high(X[te], b.latest[te])
            lg = C.ELogit().fit(protocol._ctx(b, tr, "benchmark", s))
            oof["e_logit"][r, te] = lg.score(None, b.latest[te])
            oof["calibrated"][r, te] = calibrated_baseline(b.latest[tr], b.y[tr])(b.latest[te])
            log(f"  repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")
    rng = np.random.RandomState(seed + 7)
    is_test = np.array([e.source == "test" for e in events])

    # P7: early warning among eligible escalation candidates
    def ew_block(dom, contrasts):
        y_ew = b.esc[dom]
        sc = {k: oof[k][:, dom] for k in names}
        per = {k: [metric.early_warning(y_ew, sc[k][r], None, 0.05) for r in range(repeats)] for k in names}
        summary = {k: {m: float(np.mean([p[m] for p in per[k]]))
                       for m in ("pr_auc", "roc_auc", "recall_at_budget")} for k in names}
        point = {c: float(np.mean([per[a][r]["pr_auc"] - per[z][r]["pr_auc"] for r in range(repeats)]))
                 for c, (a, z) in contrasts.items()}
        boot = {c: np.full(bootstrap, np.nan) for c in contrasts}
        m = len(y_ew)
        for bi in range(bootstrap):
            j = rng.randint(0, m, m)
            if y_ew[j].sum() == 0:
                continue
            ap = {k: [metric.average_precision(y_ew[j], sc[k][r][j]) for r in range(repeats)] for k in names}
            for c, (a, z) in contrasts.items():
                boot[c][bi] = np.mean([ap[a][r] - ap[z][r] for r in range(repeats)])
        return {"n_candidates": int(dom.sum()), "n_escalations": int(y_ew.sum()), "summary": summary,
                "contrasts": {c: {"mean": point[c], "ci95": protocol._ci(boot[c]),
                                  "mde": 2.8 * float(np.nanstd(boot[c]))} for c in contrasts}}

    contrasts = {"PUB_minus_logit": ("PUB", "e_logit"), "A0_minus_PUB": ("A0", "PUB"),
                 "PUB_minus_LIVE": ("PUB", "LIVE"), "A0_minus_logit": ("A0", "e_logit"),
                 "LIVE_minus_logit": ("LIVE", "e_logit")}
    dom = b.cand & b.eligible
    p7 = {"pooled": ew_block(dom, contrasts), "test_events_only": ew_block(dom & is_test, contrasts)}
    pc = p7["pooled"]["contrasts"]["PUB_minus_logit"]["ci95"]
    p7["reading"] = ("supported" if pc[0] is not None and pc[0] > 0 else "inconclusive")
    reference = {}
    abl = Path("reports/paper/ablation.json")
    if abl.exists():
        import json
        reference["P4_A0_pr_auc"] = json.loads(abl.read_text())["summary"]["A0_all_benchmark"]["pr_auc"]

    # P8: probability forecast of the final high-risk status, all eligible events
    el = b.eligible
    y_hr = (b.y[el] >= HIGH_RISK).astype(float)
    ll = {k: [_logloss(y_hr, oof[k][r, el]) for r in range(repeats)] for k in ("A0", "PUB", "calibrated")}
    br = {k: [float(np.mean((oof[k][r, el] - y_hr) ** 2)) for r in range(repeats)]
          for k in ("A0", "PUB", "calibrated")}
    m_el = int(el.sum())
    boot_ll = {k: np.full(bootstrap, np.nan) for k in ("A0", "PUB")}
    boot_br = {k: np.full(bootstrap, np.nan) for k in ("A0", "PUB")}
    P = {k: oof[k][:, el] for k in ("A0", "PUB", "calibrated")}
    for bi in range(bootstrap):
        j = rng.randint(0, m_el, m_el)
        yj = y_hr[j]
        base = [_logloss(yj, P["calibrated"][r][j]) for r in range(repeats)]
        base_b = [float(np.mean((P["calibrated"][r][j] - yj) ** 2)) for r in range(repeats)]
        for k in ("A0", "PUB"):
            boot_ll[k][bi] = np.mean([_logloss(yj, P[k][r][j]) - base[r] for r in range(repeats)])
            boot_br[k][bi] = np.mean([float(np.mean((P[k][r][j] - yj) ** 2)) - base_b[r]
                                      for r in range(repeats)])
    forecast = {k: {"logloss": float(np.mean(ll[k])), "brier": float(np.mean(br[k]))} for k in ll}
    for k in ("A0", "PUB"):
        forecast[k]["d_logloss_vs_calibrated"] = {
            "mean": float(np.mean(ll[k]) - np.mean(ll["calibrated"])), "ci95": protocol._ci(boot_ll[k])}
        forecast[k]["d_brier_vs_calibrated"] = {
            "mean": float(np.mean(br[k]) - np.mean(br["calibrated"])), "ci95": protocol._ci(boot_br[k])}
    d = forecast["A0"]["d_logloss_vs_calibrated"]["ci95"]
    forecast["reading"] = "better" if d[1] is not None and d[1] < 0 else "not shown"

    # P8: LRP's high-risk calls - which stay high-risk
    calls = el & ~b.cand
    stay = b.y[calls] >= HIGH_RISK
    S = {k: oof[k][:, calls] for k in ("A0", "PUB", "calibrated")}
    auc = {k: [metric.roc_auc(stay, S[k][r]) for r in range(repeats)] for k in S}
    fa = {k: [_fa_removed(stay, S[k][r]) for r in range(repeats)] for k in S}
    mc = int(calls.sum())
    boot_auc = {k: np.full(bootstrap, np.nan) for k in ("A0", "PUB")}
    boot_fa = {k: np.full(bootstrap, np.nan) for k in ("A0", "PUB", "calibrated")}
    for bi in range(bootstrap):
        j = rng.randint(0, mc, mc)
        sj = stay[j]
        if sj.all() or not sj.any():
            continue
        base = [metric.roc_auc(sj, S["calibrated"][r][j]) for r in range(repeats)]
        for k in ("A0", "PUB"):
            boot_auc[k][bi] = np.mean([metric.roc_auc(sj, S[k][r][j]) - base[r] for r in range(repeats)])
        for k in boot_fa:
            boot_fa[k][bi] = np.mean([_fa_removed(sj, S[k][r][j]) for r in range(repeats)])
    filt = {"n_calls": mc, "n_stay_high_risk": int(stay.sum()),
            "roc_auc": {k: float(np.mean(v)) for k, v in auc.items()},
            "false_alarms_removed_at_95pct": {k: {"mean": float(np.mean(fa[k])), "ci95": protocol._ci(boot_fa[k])}
                                              for k in fa}}
    for k in ("A0", "PUB"):
        filt[f"d_roc_auc_{k}_vs_calibrated"] = {"mean": float(np.mean(auc[k]) - np.mean(auc["calibrated"])),
                                                 "ci95": protocol._ci(boot_auc[k])}
    return {"analysis": "P7-P8 early warning from public-data counterparts; VIGIL's classifier as a "
                        "probability forecast (pooled CV)",
            "repeats": repeats, "folds": folds, "bootstrap": bootstrap, "seed": seed, "n_events": n,
            "n_eligible": m_el, "n_high_risk_eligible": int(y_hr.sum()),
            "feature_counts": {k: len(v[1]) for k, v in variants.items()},
            "public_features": pub_cols,
            "p7_early_warning": p7, "p8_forecast": forecast, "p8_false_alarms": filt,
            "reference": reference}


def ew_posthoc(events: list[Event], repeats: int = 5, folds: int = 5, seed: int = 42,
               bootstrap: int = 1000, log=print) -> dict:
    """Post hoc (not pre-registered): the latest-risk baselines without pooled fits.

    The registered baselines (e_logit for early warning, the calibrated latest
    risk for P8) are fitted per fold, and their out-of-fold scores are pooled
    into one ranking. Each fold's curve is monotone in the latest risk, but the
    curves differ, so pooling reorders events with similar risk. Ranking by the
    latest risk itself needs no fit. This repeats the registered run of public()
    (same folds and seeds, checked against public.json), then compares every
    model with that ranking, pooled and within folds."""
    import json
    from sklearn.model_selection import StratifiedKFold
    from backend.app.models.lab.paper import feature_sets

    b = protocol.Bundle(events)
    cols = b.columns["benchmark"]
    xb = b.X["benchmark"]
    xpub, pub_cols = public_matrix(xb, cols)
    sets = feature_sets(cols)
    sub = {k: (xb[:, np.array([b.colindex[c] for c in sets[name]], int)], sets[name])
           for k, name in (("A1", "A1_without_covariance"), ("A2", "A2_serving_plus_covariance"))}
    variants = {"A0": (xb, cols), "PUB": (xpub, pub_cols),
                "LIVE": (b.X["serving"], b.columns["serving"]), **sub}
    n = len(events)
    names = list(variants) + ["e_logit", "calibrated"]
    oof = {k: np.full((repeats, n), np.nan) for k in names}
    fold_of = np.full((repeats, n), -1)
    # the loop of public() (and, for A1 and A2, of paper.ablation), kept separate so that
    # the registered functions stay unchanged
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101 * r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), b.strata)):
            t0 = time.time()
            fold_of[r, te] = f
            tr = tr[b.trainable[tr]]
            s = seed + 1000 * r + f
            for k, (X, vc) in variants.items():
                ctx = C.Context(X[tr], b.y[tr], b.latest[tr], b.strata[tr], s, b.eligible[tr],
                                {"colindex": {c: i for i, c in enumerate(vc)}, "colnames": vc,
                                 "y_aux": b.y_aux[tr]})
                h = C.Hurdle(f"hurdle_{k}")
                h.fit(ctx)
                oof[k][r, te] = h.prob_high(X[te], b.latest[te])
            lg = C.ELogit().fit(protocol._ctx(b, tr, "benchmark", s))
            oof["e_logit"][r, te] = lg.score(None, b.latest[te])
            oof["calibrated"][r, te] = calibrated_baseline(b.latest[tr], b.y[tr])(b.latest[te])
            log(f"  repeat {r + 1}/{repeats} fold {f + 1}/{folds}  {time.time() - t0:.0f}s")
    rng = np.random.RandomState(seed + 13)
    models = ("A0", "A1", "A2", "PUB", "LIVE", "e_logit")

    # early warning among eligible escalation candidates
    dom = b.cand & b.eligible
    y_ew, lat = b.esc[dom], b.latest[dom]
    sc = {k: oof[k][:, dom] for k in models}
    pooled = {k: float(np.mean([metric.average_precision(y_ew, sc[k][r]) for r in range(repeats)])) for k in models}
    reg = json.loads(Path("reports/paper/public.json").read_text())["p7_early_warning"]["pooled"]["summary"]
    abl = json.loads(Path("reports/paper/ablation.json").read_text())["summary"]
    reg_pr = {**{k: reg[k]["pr_auc"] for k in ("A0", "PUB", "LIVE", "e_logit")},
              "A1": abl["A1_without_covariance"]["pr_auc"], "A2": abl["A2_serving_plus_covariance"]["pr_auc"]}
    reproduced = {k: abs(pooled[k] - reg_pr[k]) < 1e-12 for k in models}
    roc = {k: float(np.mean([metric.roc_auc(y_ew, sc[k][r]) for r in range(repeats)])) for k in models}
    recall = {k: float(np.mean([metric.recall_at_budget(y_ew, sc[k][r], 0.05) for r in range(repeats)])) for k in models}
    ap_lat = metric.average_precision(y_ew, lat)
    m = len(y_ew)
    boot = {k: np.full(bootstrap, np.nan) for k in models}
    boot_roc = {k: np.full(bootstrap, np.nan) for k in models}
    for bi in range(bootstrap):
        j = rng.randint(0, m, m)
        if y_ew[j].sum() == 0:
            continue
        base, base_roc = metric.average_precision(y_ew[j], lat[j]), metric.roc_auc(y_ew[j], lat[j])
        for k in models:
            boot[k][bi] = np.mean([metric.average_precision(y_ew[j], sc[k][r][j]) for r in range(repeats)]) - base
            boot_roc[k][bi] = np.mean([metric.roc_auc(y_ew[j], sc[k][r][j]) for r in range(repeats)]) - base_roc
    within = {k: [] for k in models + ("latest",)}
    fd = fold_of[:, dom]
    for r in range(repeats):
        for f in range(folds):
            i = fd[r] == f
            if y_ew[i].sum() == 0:
                continue
            within["latest"].append(metric.average_precision(y_ew[i], lat[i]))
            for k in models:
                within[k].append(metric.average_precision(y_ew[i], sc[k][r][i]))
    is_test = np.array([e.source == "test" for e in events])[dom]
    early = {"n_candidates": int(m), "n_escalations": int(y_ew.sum()),
             "pooled_pr_auc": pooled, "pooled_roc_auc": roc, "recall_at_budget": recall,
             "reproduces_registered_reports": reproduced,
             "latest_risk_ranking": {"pr_auc": ap_lat, "roc_auc": metric.roc_auc(y_ew, lat),
                                     "recall_at_budget": metric.recall_at_budget(y_ew, lat, 0.05),
                                     "pr_auc_test_events_only": metric.average_precision(y_ew[is_test], lat[is_test])},
             "contrasts_vs_latest_risk": {k: {"mean": pooled[k] - ap_lat, "ci95": protocol._ci(boot[k])}
                                          for k in models},
             "roc_contrasts_vs_latest_risk": {k: {"mean": roc[k] - metric.roc_auc(y_ew, lat),
                                                  "ci95": protocol._ci(boot_roc[k])} for k in models},
             "within_fold_mean_pr_auc": {k: float(np.mean(v)) for k, v in within.items()},
             "n_folds_with_escalations": len(within["latest"])}

    # P8: LRP's high-risk calls, ranked by the latest risk itself
    el = b.eligible
    calls = el & ~b.cand
    stay = b.y[calls] >= HIGH_RISK
    lat_c = b.latest[calls]
    S = {k: oof[k][:, calls] for k in ("A0", "PUB", "calibrated")}
    auc = {k: float(np.mean([metric.roc_auc(stay, S[k][r]) for r in range(repeats)])) for k in S}
    mc = int(calls.sum())
    boot_auc = {k: np.full(bootstrap, np.nan) for k in ("A0", "PUB")}
    boot_fa = np.full(bootstrap, np.nan)
    for bi in range(bootstrap):
        j = rng.randint(0, mc, mc)
        sj = stay[j]
        if sj.all() or not sj.any():
            continue
        base = metric.roc_auc(sj, lat_c[j])
        for k in boot_auc:
            boot_auc[k][bi] = np.mean([metric.roc_auc(sj, S[k][r][j]) for r in range(repeats)]) - base
        boot_fa[bi] = _fa_removed(sj, lat_c[j])
    filt = {"n_calls": mc, "pooled_roc_auc": auc,
            "latest_risk_ranking": {"roc_auc": metric.roc_auc(stay, lat_c),
                                    "false_alarms_removed_at_95pct": {"mean": _fa_removed(stay, lat_c),
                                                                      "ci95": protocol._ci(boot_fa)}},
            "d_roc_auc_vs_latest_risk": {k: {"mean": auc[k] - metric.roc_auc(stay, lat_c),
                                             "ci95": protocol._ci(boot_auc[k])} for k in boot_auc}}
    return {"analysis": "Post-hoc check (not pre-registered): early-warning and false-alarm baselines "
                        "ranked by the latest risk itself, without pooled per-fold fits",
            "repeats": repeats, "folds": folds, "bootstrap": bootstrap, "seed": seed, "n_events": n,
            "early_warning": early, "false_alarms": filt}


def main(argv=None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description="VIGIL Lab Round 4")
    p.add_argument("analysis", choices=("coherence", "public", "posthoc", "ewposthoc"))
    p.add_argument("--train", default=DEFAULTS["train"])
    p.add_argument("--test", default=DEFAULTS["test"])
    p.add_argument("--test-targets", default=DEFAULTS["test_targets"])
    p.add_argument("--raw", default=RAW)
    p.add_argument("--out", default="reports/paper")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    out, t0 = Path(a.out), time.time()
    if a.analysis == "coherence":
        _, test = data.load(a.train, a.test, a.test_targets)
        tdf, tgt = pd.read_csv(a.test), pd.read_csv(a.test_targets)
        late = match_raw(tdf, tgt, pd.read_csv(a.raw))
        print(f"loaded {len(test)} test events; {len(late)} matched to a unique raw sequence")
        _write(out, "coherence", coherence(test, late, seed=a.seed), t0)
        return 0
    if a.analysis == "posthoc":
        train, test = data.load(a.train, a.test, a.test_targets)
        tdf, tgt = pd.read_csv(a.test), pd.read_csv(a.test_targets)
        raw = pd.read_csv(a.raw)
        late, late_tol = match_raw(tdf, tgt, raw), match_raw_tol(tdf, tgt, raw)
        print(f"loaded {len(train)} training and {len(test)} test events; {len(late)} matched "
              f"({len(late_tol)} with tolerances)")
        _write(out, "coherence_posthoc", posthoc(test, late, train, seed=a.seed, late_tol=late_tol), t0)
        return 0
    train, test = data.load(a.train, a.test, a.test_targets)
    print(f"loaded {len(train)} training events, {len(test)} test events")
    if a.analysis == "ewposthoc":
        _write(out, "ew_posthoc", ew_posthoc(train + test, seed=a.seed), t0)
        return 0
    _write(out, "public", public(train + test, seed=a.seed), t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
