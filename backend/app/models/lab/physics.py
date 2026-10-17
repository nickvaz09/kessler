"""Encounter-plane physics features from a Kelvins CDM row.

Why: the risk in a CDM is a 2-D encounter-plane probability that depends on
the miss vector and the combined position covariance. Whether a LOW risk is
reassuring depends on where the event sits on the Pc-vs-covariance curve
(Alfano's maximum-probability analysis; NASA CARA "dilution region"): if the
covariance is large relative to the miss distance, shrinking it as tracking
improves RAISES Pc - exactly how quiet events escalate. These features give
the model that curve explicitly instead of hoping trees rediscover it.

Model (standard short-encounter approximation):
    C  = C_target + C_chaser            (RTN, 3x3; see APPROXIMATION below)
    B  = orthonormal basis of the plane perpendicular to v_rel
    C2 = B C B^T,  m^2 = r2^T C2^-1 r2   (squared in-plane Mahalanobis)
    Pc ~= HBR^2 / (2 sqrt(det C2)) * exp(-m^2 / 2)         (HBR << sigma)

Scaling the covariance by k (C2 -> k C2) gives Pc(k) ∝ exp(-m^2/(2k)) / k,
maximised at k* = m^2/2. So:
    dilution_index = log(m^2 / 2)      < 0  -> dilution region (Pc rises if
                                             the covariance shrinks)
    log10 Pc_max - log10 Pc = log10(2 / (e m^2)) + (m^2/2) / ln 10  (>= 0);
        reaching Pc_max needs a SHRINKING covariance only when m^2 < 2
        (dilution region) - above it, the covariance would have to grow
    projected risk after shrinking sigma by factor s (k = s^2):
        log10 Pc(k) = risk - log10 k - (m^2/2)(1/k - 1)/ln 10

APPROXIMATION: CCSDS CDMs give each object's covariance in its OWN RTN
frame; adding them directly (as here) ignores the rotation between frames.
For LEO conjunctions the frames differ mainly by the crossing angle, so the
sum is a known-biased but monotone proxy - good as a feature, not as an
operational Pc. The Kelvins ``risk`` column remains the authoritative Pc.
"""
from __future__ import annotations

import math

import numpy as np

_LN10 = math.log(10.0)


def _f(row: dict, key: str) -> float:
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def covariance_rtn(row: dict, prefix: str) -> np.ndarray | None:
    """3x3 RTN position covariance [m^2] of object ``prefix`` ('t' or 'c')
    from sigma_r/t/n [m] and correlations ct_r, cn_r, cn_t."""
    sr, st, sn = (_f(row, f"{prefix}_sigma_{a}") for a in "rtn")
    if not all(np.isfinite([sr, st, sn])) or min(sr, st, sn) <= 0:
        return None
    rho_tr = _f(row, f"{prefix}_ct_r")
    rho_nr = _f(row, f"{prefix}_cn_r")
    rho_nt = _f(row, f"{prefix}_cn_t")
    rho_tr, rho_nr, rho_nt = (0.0 if not np.isfinite(x) else float(np.clip(x, -0.999, 0.999))
                              for x in (rho_tr, rho_nr, rho_nt))
    s = np.array([sr, st, sn])
    corr = np.array([[1.0, rho_tr, rho_nr],
                     [rho_tr, 1.0, rho_nt],
                     [rho_nr, rho_nt, 1.0]])
    cov = corr * np.outer(s, s)
    # nearest PSD (clip eigenvalues) - correlation triples can be inconsistent
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-6, None)
    return (V * w) @ V.T


def encounter_plane(row: dict) -> dict:
    """Physics features for one CDM row; NaN where inputs are missing."""
    nan = float("nan")
    out = {"m2": nan, "log_det_c2": nan, "dilution_index": nan,
           "log10_pcmax_gap": nan, "log10_hbr_implied": nan,
           "sigma_ratio_t_c": nan, "proj_risk_s50": nan, "proj_risk_s25": nan}
    r = np.array([_f(row, f"relative_position_{a}") for a in "rtn"])
    v = np.array([_f(row, f"relative_velocity_{a}") for a in "rtn"])
    ct, cc = covariance_rtn(row, "t"), covariance_rtn(row, "c")
    if ct is None or cc is None or not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        return out
    vn = np.linalg.norm(v)
    if vn <= 0:
        return out
    vh = v / vn
    r_perp = r - np.dot(r, vh) * vh
    if np.linalg.norm(r_perp) < 1e-9:
        # head-on miss vector degenerate: any in-plane axis
        a = np.array([1.0, 0.0, 0.0]) if abs(vh[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r_perp = a - np.dot(a, vh) * vh
        miss_in_plane = 0.0
    else:
        miss_in_plane = float(np.linalg.norm(r_perp))
    u1 = r_perp / np.linalg.norm(r_perp)
    u2 = np.cross(vh, u1)
    B = np.vstack([u1, u2])
    C = ct + cc
    C2 = B @ C @ B.T
    det = float(np.linalg.det(C2))
    if det <= 0:
        return out
    r2 = np.array([miss_in_plane, 0.0])
    m2 = float(r2 @ np.linalg.solve(C2, r2))
    out["m2"] = m2
    out["log_det_c2"] = math.log(det)
    m2c = max(m2, 1e-9)
    out["dilution_index"] = math.log(m2c / 2.0)
    out["log10_pcmax_gap"] = max(0.0, math.log10(2.0 / (math.e * m2c)) + (m2c / 2.0) / _LN10)
    risk = _f(row, "risk")
    if np.isfinite(risk) and risk > -29.9:
        # Pc = HBR^2/(2 sqrt det) exp(-m2/2)  ->  implied HBR (a sanity/size proxy)
        log10_hbr2 = risk + math.log10(2.0 * math.sqrt(det)) + (m2 / 2.0) / _LN10
        out["log10_hbr_implied"] = 0.5 * log10_hbr2
        for s, key in ((0.5, "proj_risk_s50"), (0.25, "proj_risk_s25")):
            k = s * s
            out[key] = risk - math.log10(k) - (m2 / 2.0) * (1.0 / k - 1.0) / _LN10
    st, sc = _f(row, "t_sigma_t"), _f(row, "c_sigma_t")
    if np.isfinite(st) and np.isfinite(sc) and st > 0 and sc > 0:
        out["sigma_ratio_t_c"] = math.log10(sc / st)
    return out


def projected_risk(risk: float, m2: float, shrink: float) -> float:
    """log10 Pc after the combined in-plane covariance scales by shrink^2."""
    if not (np.isfinite(risk) and np.isfinite(m2)) or shrink <= 0 or risk <= -29.9:
        return float("nan")
    k = shrink * shrink
    return risk - math.log10(k) - (m2 / 2.0) * (1.0 / k - 1.0) / _LN10
