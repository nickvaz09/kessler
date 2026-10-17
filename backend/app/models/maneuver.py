"""What-if maneuver sandbox - two-body Kepler propagation of a post-burn state.

The maneuver endpoint (POST /conjunctions/{id}/maneuver) needs to answer "what
would an along-track burn do to this encounter?" in milliseconds. Full-force
re-propagation is neither possible (TLEs cannot be re-fit around an impulsive
burn) nor necessary, so the post-burn primary is modeled as a TWO-BODY orbit:
SGP4 gives the osculating state at the burn instant, the delta-v is added along
the velocity unit vector, and the resulting state is converted to Kepler
elements that propagate analytically (Kepler's equation, Newton-solved).

Documented approximation
------------------------
Two-body only: drag and all geopotential harmonics (J2 secular node/perigee
drift, short-period oscillations) are ignored after the burn. Absolute
two-body positions therefore diverge from SGP4 truth by hundreds of meters
within minutes and by tens of kilometers over several hours - which is why the
API layer never uses the post-burn trajectory absolutely. It differences two
two-body propagations from the same burn state (with and without the dv) so
the omitted-force error is common-mode and cancels, leaving the clean
maneuver response. Valid on the hours scale only; mu = 398600.4418 km^3/s^2.

Element set and degeneracies
----------------------------
Elements are the classical Keplerian set, degrees per repo convention:
{a_km, e, inc_deg, raan_deg, argp_deg, mean_anomaly_deg}. Circular and/or
equatorial states make raan/argp individually ill-defined; every angle here is
extracted with atan2 from a frame built in one consistent chain
(w-hat = h/|h| -> raan -> node axis -> argp -> perifocal axes -> anomaly), and
elements_to_rv rebuilds the exact same chain from the stored angles. Any
convention noise in one angle is therefore compensated by the next, so
r,v -> elements -> r,v round-trips to sub-millimeter for all orbit shapes
(circular, equatorial, both, retrograde). atan2(0, 0) = 0 is the convention
fallback for the fully degenerate axes. Elliptical orbits only: a state with
specific energy >= 0 (parabolic/hyperbolic) raises ValueError.

Pure numpy/math throughout; sgp4 is imported lazily inside apply_maneuver, and
optimize_dv lazily reuses the maneuver endpoint's recompute helpers in
backend.app.api.conjunctions - one implementation, no duplicated propagation.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np

MU_KM3_S2 = 398600.4418   # Earth gravitational parameter (matches propagate.py)
TWO_PI = 2.0 * math.pi

_KEPLER_TOL_RAD = 1e-14   # Newton convergence on E - e*sinE - M
_KEPLER_MAX_ITER = 50


def _parse_iso_utc(s: str) -> datetime:
    """ISO-8601 string -> aware UTC datetime; naive input is treated as UTC."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _perifocal_axes(inc_rad: float, raan_rad: float,
                    argp_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """(p_hat, q_hat) perifocal unit axes in ECI for the given angles.

    Built through the same node/normal chain rv_to_elements extracts with, so
    extraction and reconstruction agree to machine precision even when the
    angles are pure convention (circular/equatorial degeneracies).
    """
    si, ci = math.sin(inc_rad), math.cos(inc_rad)
    so_, co_ = math.sin(raan_rad), math.cos(raan_rad)
    w_hat = np.array([so_ * si, -co_ * si, ci])          # orbit normal
    n_hat = np.array([co_, so_, 0.0])                    # node axis
    m_hat = np.cross(w_hat, n_hat)                       # in-plane, node + 90 deg
    sw, cw = math.sin(argp_rad), math.cos(argp_rad)
    p_hat = cw * n_hat + sw * m_hat                      # toward periapsis
    q_hat = np.cross(w_hat, p_hat)
    return p_hat, q_hat


def rv_to_elements(r_km, v_kms) -> dict:
    """Osculating two-body elements from an ECI state (km, km/s).

    Returns {a_km, e, inc_deg, raan_deg, argp_deg, mean_anomaly_deg} with
    angles in [0, 360). See the module docstring for the degeneracy
    conventions (raan/argp fall back to atan2(0,0)=0 axes, self-consistently
    with elements_to_rv). Raises ValueError for non-elliptical or rectilinear
    states.
    """
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_kms, dtype=float)
    rmag = float(np.linalg.norm(r))
    if rmag == 0.0:
        raise ValueError("zero position vector")
    v2 = float(np.dot(v, v))
    energy = 0.5 * v2 - MU_KM3_S2 / rmag
    if energy >= 0.0:
        raise ValueError(
            "state is not elliptical (specific energy >= 0); the two-body "
            "sandbox handles closed orbits only")
    a = -MU_KM3_S2 / (2.0 * energy)

    h = np.cross(r, v)
    hmag = float(np.linalg.norm(h))
    if hmag == 0.0:
        raise ValueError("rectilinear trajectory (r parallel to v)")
    w_hat = h / hmag

    inc = math.atan2(math.hypot(w_hat[0], w_hat[1]), w_hat[2])
    raan = math.atan2(w_hat[0], -w_hat[1])   # node = z-hat x w-hat; 0 when equatorial
    n_hat = np.array([math.cos(raan), math.sin(raan), 0.0])
    m_hat = np.cross(w_hat, n_hat)

    e_vec = ((v2 - MU_KM3_S2 / rmag) * r - float(np.dot(r, v)) * v) / MU_KM3_S2
    e = float(np.linalg.norm(e_vec))
    argp = math.atan2(float(np.dot(e_vec, m_hat)), float(np.dot(e_vec, n_hat)))

    p_hat = math.cos(argp) * n_hat + math.sin(argp) * m_hat
    q_hat = np.cross(w_hat, p_hat)
    nu = math.atan2(float(np.dot(r, q_hat)), float(np.dot(r, p_hat)))

    # true -> eccentric -> mean anomaly (elliptical only, e < 1 by energy check)
    ecc_anom = math.atan2(math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(nu),
                          e + math.cos(nu))
    mean_anom = ecc_anom - e * math.sin(ecc_anom)

    return {
        "a_km": a,
        "e": e,
        "inc_deg": math.degrees(inc),
        "raan_deg": math.degrees(raan) % 360.0,
        "argp_deg": math.degrees(argp) % 360.0,
        "mean_anomaly_deg": math.degrees(mean_anom) % 360.0,
    }


def _solve_kepler(mean_anom_rad: np.ndarray, e: float) -> np.ndarray:
    """Vectorized Newton solve of Kepler's equation E - e*sinE = M."""
    m = np.mod(mean_anom_rad, TWO_PI)
    ecc_anom = m + e * np.sin(m)          # good starter for e < ~0.9
    for _ in range(_KEPLER_MAX_ITER):
        f = ecc_anom - e * np.sin(ecc_anom) - m
        ecc_anom = ecc_anom - f / (1.0 - e * np.cos(ecc_anom))
        if float(np.max(np.abs(f))) < _KEPLER_TOL_RAD:
            break
    return ecc_anom


def propagate_two_body(elements: dict, dt_s):
    """Two-body state(s) dt_s seconds after the elements' epoch.

    dt_s may be a scalar or an array-like of N offsets [s]. Returns
    (r_km, v_kms): shape (3,)/(3,) for scalar input, (N,3)/(N,3) for arrays.
    Kepler's equation is Newton-solved (vectorized); mu = 398600.4418.
    """
    a = float(elements["a_km"])
    e = float(elements["e"])
    if a <= 0.0 or not 0.0 <= e < 1.0:
        raise ValueError(f"non-elliptical elements (a={a}, e={e})")
    n = math.sqrt(MU_KM3_S2 / (a * a * a))       # mean motion [rad/s]
    m0 = math.radians(elements["mean_anomaly_deg"])

    dt = np.asarray(dt_s, dtype=float)
    scalar = dt.ndim == 0
    mean_anom = m0 + n * np.atleast_1d(dt)
    ecc_anom = _solve_kepler(mean_anom, e)

    cos_e, sin_e = np.cos(ecc_anom), np.sin(ecc_anom)
    axis_ratio = math.sqrt(1.0 - e * e)          # b/a
    x_pf = a * (cos_e - e)                       # perifocal coordinates
    y_pf = a * axis_ratio * sin_e
    rmag = a * (1.0 - e * cos_e)
    vf = math.sqrt(MU_KM3_S2 * a) / rmag
    vx_pf = -vf * sin_e
    vy_pf = vf * axis_ratio * cos_e

    p_hat, q_hat = _perifocal_axes(math.radians(elements["inc_deg"]),
                                   math.radians(elements["raan_deg"]),
                                   math.radians(elements["argp_deg"]))
    r = np.outer(x_pf, p_hat) + np.outer(y_pf, q_hat)
    v = np.outer(vx_pf, p_hat) + np.outer(vy_pf, q_hat)
    if scalar:
        return r[0], v[0]
    return r, v


def elements_to_rv(elements: dict) -> tuple[np.ndarray, np.ndarray]:
    """ECI state (km, km/s) at the elements' own mean anomaly (dt = 0)."""
    return propagate_two_body(elements, 0.0)


DV_TOLERANCE_MS = 0.01          # optimizer convergence tolerance on |dv|
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0   # golden ratio conjugate ~0.618

OPTIMIZE_METHOD = (
    "golden-section per sign over the two-body maneuver sandbox "
    "(endpoint recompute reused, 0.01 m/s tolerance); two-body "
    "approximation — drag/J2 cancel common-mode only"
)


def optimize_dv(conn, conjunction_id: int, target_miss_m: float,
                burn_iso: str) -> dict:
    """Minimum-|dv| along-track burn achieving new_miss >= target_miss_m.

    Searches dv in [-5, +5] m/s (the endpoint's own bound) by golden-section
    interval shrink ON EACH SIGN, reusing the maneuver endpoint's recompute
    (api.conjunctions.load_maneuver_context / evaluate_maneuver - the SGP4
    grid and DB reads happen once; each candidate dv costs one two-body
    propagation).

    Why the per-sign search is sound: near TCA the encounter geometry is
    locally linear in dv, so miss(dv) ~ |m0 + k*dv| is convex on each sign.
    With miss(0) < target that means (a) the sign's maximum sits at its +/-5
    endpoint - an infeasible endpoint certifies the whole sign unreachable -
    and (b) when the endpoint is feasible there is exactly one crossing, and
    the bracket invariant miss(lo) < target <= miss(hi) pins it. Each probe
    lands at the golden point of the bracket, so the interval shrinks by at
    least the golden ratio per evaluation down to DV_TOLERANCE_MS.

    Returns {conjunction_id, target_miss_m, dv_ms, achieved_miss_m, burn_iso,
    evaluations, reachable, method}. When the target is unreachable within
    +/-5 m/s, reachable is False and dv_ms/achieved_miss_m report the best
    (largest-miss) point evaluated. Raises LookupError for an unknown
    conjunction id and ValueError for burn/TLE/propagation problems or a
    non-positive target (same messages as the endpoint's 422 details).
    """
    # Lazy import: the single implementation of the encounter recompute lives
    # beside the endpoint (refactored in place ) - never duplicated here.
    from backend.app.api import conjunctions as conj_api

    target = float(target_miss_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_miss_m must be a positive number of meters")

    burn = _parse_iso_utc(burn_iso)
    ctx = conj_api.load_maneuver_context(conn, conjunction_id, burn)
    dv_max = conj_api.MANEUVER_DV_MAX_MS

    samples: list[tuple[float, float]] = []   # every (dv, miss) evaluated

    def miss_at(dv: float) -> float:
        miss = float(conj_api.evaluate_maneuver(ctx, dv)["new_miss_m"])
        samples.append((dv, miss))
        return miss

    def result(dv: float, achieved: float, reachable: bool) -> dict:
        return {
            "conjunction_id": conjunction_id,
            "target_miss_m": round(target, 1),
            "dv_ms": round(dv, 4),
            "achieved_miss_m": round(achieved, 1),
            "burn_iso": burn.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evaluations": len(samples),
            "reachable": reachable,
            "method": OPTIMIZE_METHOD,
        }

    baseline = miss_at(0.0)
    if baseline >= target:
        return result(0.0, baseline, True)   # no burn needed: minimal |dv|=0

    best = None   # (|dv|, dv, miss) over feasible candidates, both signs
    for sign in (1.0, -1.0):
        miss_hi = miss_at(sign * dv_max)
        if miss_hi < target:
            continue   # infeasible endpoint certifies the sign (convexity)
        lo, hi = 0.0, dv_max   # invariant: miss(sign*lo) < target <= miss(sign*hi)
        while hi - lo > DV_TOLERANCE_MS:
            probe = hi - _INV_PHI * (hi - lo)
            miss_probe = miss_at(sign * probe)
            if miss_probe >= target:
                hi, miss_hi = probe, miss_probe
            else:
                lo = probe
        if best is None or hi < best[0]:
            best = (hi, sign * hi, miss_hi)

    if best is not None:
        return result(best[1], best[2], True)

    dv_best, miss_best = max(samples, key=lambda s: s[1])
    return result(dv_best, miss_best, False)


def apply_maneuver(line1: str, line2: str, burn_iso: str,
                   dv_along_ms: float) -> dict:
    """Post-burn two-body elements for an along-track impulsive burn.

    SGP4 propagates the TLE to `burn_iso` (ISO-8601 UTC; naive strings are
    read as UTC), dv_along_ms [m/s] is added along the velocity unit vector
    (positive = prograde, negative = retrograde), and the resulting state is
    converted to Kepler elements. The returned dict is rv_to_elements' output
    plus "epoch": burn_iso, ready for propagate_two_body(elements, t - burn).

    Approximation (see module docstring): drag and J2 are ignored post-burn,
    so use the result differentially (burn minus no-burn) on hours scales.
    Raises ValueError on SGP4 propagation failure or a degenerate state.
    """
    from backend.app.ingestion import propagate  # lazy: sgp4 only needed here

    burn_dt = _parse_iso_utc(burn_iso)
    r, v = propagate.positions_at(line1, line2, [burn_dt])  # ValueError on sgp4 error
    r0 = np.asarray(r[0], dtype=float)
    v0 = np.asarray(v[0], dtype=float)
    speed = float(np.linalg.norm(v0))
    if speed == 0.0:
        raise ValueError("zero velocity at burn time")
    v_burned = v0 + (float(dv_along_ms) / 1000.0) * (v0 / speed)
    elements = rv_to_elements(r0, v_burned)
    elements["epoch"] = burn_iso
    return elements


# --- One-click avoidance plan -------------------------------------------------
#
# POST /conjunctions/{id}/avoidance-plan. For a handful of lead times before
# closest approach, find the SMALLEST along-track burn that makes the encounter
# safe, then recommend one. Everything reuses the maneuver sandbox recompute
# (api.conjunctions.load_maneuver_context / evaluate_maneuver): same two-body
# differential model, same assumed-sigma Pc - so a plan the UI "applies" to
# the what-if controls reproduces exactly through POST /maneuver.
#
# Safe = both criteria (documented choice):
#   1. miss >= target_miss_m (default 1 km - a round, explainable floor that
#      is ~2-3 sigma of the TLE-class assumed uncertainty at typical ages), and
#   2. assumed-sigma Pc < 95% of config.PC_ELEVATED (the 5% margin absorbs the
#      sub-second shift of the post-burn TCA, which nudges the TLE age).
# Criterion 2 is converted to a miss distance by bisection on the Pc model
# (Pc is strictly decreasing in miss), so the search is one monotone target:
# required_miss_m = max(target, pc_required_miss_m).

PLAN_METHOD = ("two-body differential, along-track, screening-grade; "
               "not an operational maneuver plan")
PLAN_DEFAULT_TARGET_MISS_M = 1000.0
PLAN_TARGET_MISS_MAX_M = 100_000.0
PLAN_FIXED_LEADS_S = (3 * 3600.0, 6 * 3600.0, 12 * 3600.0)
PLAN_MIN_BURN_AHEAD_S = 300.0   # a burn must still be >= 5 min in the future
PLAN_DV_TOLERANCE_MS = 0.001    # finer than optimize_dv: plans read in cm/s
PLAN_PC_MARGIN = 0.95
PLAN_TIE_MS = 0.01              # |dv| within this of the smallest is a tie
PLAN_ISP_S = 220.0              # fuel context: hydrazine monopropellant class
PLAN_REF_MASS_KG = 100.0        # fuel context: grams per 100 kg of spacecraft
G0_MS2 = 9.80665

PLAN_RULE = (
    "Each lead time gets the smallest along-track burn (either direction) "
    "whose post-burn miss reaches the required miss: the larger of the "
    "target miss and the miss at which the assumed-sigma Pc falls below 95% "
    "of the elevated threshold. The recommendation is the feasible option "
    "with the smallest |dv|; options within 0.01 m/s of it count as a tie, "
    "broken toward the later burn (less extrapolation, more time for fresh "
    "tracking before committing).")


class PlanConflict(ValueError):
    """The event can no longer be planned for (closest approach passed, or
    too close for any candidate burn). The API maps this to 409."""


def min_dv_for_miss(ctx: dict, target_m: float, dv_max: float,
                    tol: float = DV_TOLERANCE_MS) -> dict:
    """Smallest |dv| (either sign) with post-burn miss >= target_m.

    Same per-sign golden-section bracket as optimize_dv (see its docstring
    for why it is sound), run against an already-loaded maneuver context.
    Returns {dv_ms, miss_m, reachable, evaluations}; an unreachable target
    reports the largest-miss point evaluated.
    """
    from backend.app.api import conjunctions as conj_api

    samples: list[tuple[float, float]] = []

    def miss_at(dv: float) -> float:
        miss = float(conj_api.evaluate_maneuver(ctx, dv)["new_miss_m"])
        samples.append((dv, miss))
        return miss

    def out(dv, miss, ok):
        return {"dv_ms": dv, "miss_m": miss, "reachable": ok,
                "evaluations": len(samples)}

    if miss_at(0.0) >= target_m:
        return out(0.0, samples[0][1], True)
    best = None
    for sign in (1.0, -1.0):
        miss_hi = miss_at(sign * dv_max)
        if miss_hi < target_m:
            continue
        lo, hi = 0.0, dv_max
        while hi - lo > tol:
            probe = hi - _INV_PHI * (hi - lo)
            miss_probe = miss_at(sign * probe)
            if miss_probe >= target_m:
                hi, miss_hi = probe, miss_probe
            else:
                lo = probe
        if best is None or hi < best[0]:
            best = (hi, sign * hi, miss_hi)
    if best is not None:
        return out(best[1], best[2], True)
    dv_best, miss_best = max(samples, key=lambda s: s[1])
    return out(dv_best, miss_best, False)


def _pc_required_miss_m(pc_fn, threshold: float) -> float | None:
    """Smallest miss [m] with pc_fn(miss) < threshold (Pc strictly decreasing
    in miss): 0 when even a zero miss clears it, None when Pc is unavailable.
    Bisection to 0.5 m over [0, 1000 km]."""
    p0 = pc_fn(0.0)
    if p0 is None:
        return None
    if p0 < threshold:
        return 0.0
    lo, hi = 0.0, 1_000_000.0
    p_hi = pc_fn(hi)
    if p_hi is None or p_hi >= threshold:
        return None
    while hi - lo > 0.5:
        mid = 0.5 * (lo + hi)
        p = pc_fn(mid)
        if p is not None and p < threshold:
            hi = mid
        else:
            lo = mid
    return hi


def propellant_g(dv_ms: float, mass_kg: float = PLAN_REF_MASS_KG,
                 isp_s: float = PLAN_ISP_S) -> float:
    """Tsiolkovsky propellant mass [g] for |dv| on mass_kg at isp_s."""
    ve = isp_s * G0_MS2
    return mass_kg * 1000.0 * (1.0 - math.exp(-abs(float(dv_ms)) / ve))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_avoidance(conn, conjunction_id: int,
                   target_miss_m: float | None = None,
                   max_dv_ms: float | None = None) -> dict:
    """One-click avoidance plan - smallest along-track burn per lead time.

    Candidate burns sit 1 orbit (the primary's TLE mean motion), 3 h, 6 h and
    12 h before the stored TCA; only burns still >= 5 min in the future are
    searched, the rest are listed under `skipped`. See the section comment
    for the safety criteria and PLAN_RULE for the recommendation rule.

    Raises LookupError (unknown id), PlanConflict (TCA passed, or no
    candidate burn left), ValueError (bad inputs, unusable TLEs).
    """
    from backend.app import config
    from backend.app.api import conjunctions as conj_api

    target = PLAN_DEFAULT_TARGET_MISS_M
    if target_miss_m is not None:
        target = float(target_miss_m)
        if not (math.isfinite(target) and 0.0 < target <= PLAN_TARGET_MISS_MAX_M):
            raise ValueError("target_miss_m must be between 0 and 100000 m")
    dv_cap = conj_api.MANEUVER_DV_MAX_MS
    if max_dv_ms is not None:
        dv_cap = float(max_dv_ms)
        if not (math.isfinite(dv_cap) and 0.0 < dv_cap <= conj_api.MANEUVER_DV_MAX_MS):
            raise ValueError("max_dv_ms must be between 0 and 5 m/s")

    row = conn.execute(conj_api._EPHEMERIS_PAIR_SQL,
                       (conjunction_id,)).fetchone()
    if row is None:
        raise LookupError(f"Conjunction {conjunction_id} not found")
    tca = _parse_iso_utc(row["tca"])
    now = datetime.now(timezone.utc)
    if tca <= now:
        raise PlanConflict("closest approach has already passed")

    tle_p = conn.execute(conj_api._LATEST_TLE_SQL,
                         (row["primary_id"],)).fetchone()
    sat_p = conj_api._satrec_or_none(tle_p)
    if sat_p is None:
        raise ValueError(
            "avoidance plan unavailable: the primary has no usable TLE")
    period_s = TWO_PI / sat_p.no_kozai * 60.0   # no_kozai: rad/min

    leads = sorted([(period_s, "1 orbit")]
                   + [(s, f"{s / 3600.0:g} h") for s in PLAN_FIXED_LEADS_S])
    candidates, skipped = [], []
    for lead_s, label in leads:
        if any(abs(lead_s - c[0]["lead_s"]) < 300.0 for c in candidates):
            continue   # an orbit period landing on a fixed lead: keep one
        burn = (tca - timedelta(seconds=lead_s)).replace(microsecond=0)
        entry = {"lead_label": label, "lead_s": round(lead_s, 1),
                 "burn_iso": _iso(burn)}
        if (burn - now).total_seconds() < PLAN_MIN_BURN_AHEAD_S:
            skipped.append({**entry, "reason": "burn time is already past "
                            "or less than 5 minutes away"})
            continue
        candidates.append((entry, burn))
    if not candidates:
        raise PlanConflict(
            "closest approach is too soon to plan a burn (every candidate "
            "burn time is past or less than 5 minutes away)")

    threshold = config.PC_ELEVATED
    options: list[dict] = []
    baseline = criteria = None
    required = target
    for entry, burn in candidates:
        ctx = conj_api.load_maneuver_context(conn, conjunction_id, burn)
        tle_s = ctx["tle_s"]
        old = ctx["old"]

        def pc_at(miss_m, when, rel_vel_ms):
            return conj_api._assumed_pc(miss_m, rel_vel_ms, when, tle_p, tle_s)

        if baseline is None:
            pc_req = _pc_required_miss_m(
                lambda m: pc_at(m, old["tca"], old["rel_vel_ms"]),
                threshold * PLAN_PC_MARGIN)
            required = max(target, pc_req or 0.0)
            baseline = {
                "miss_m": round(old["miss_m"], 1),
                "pc": pc_at(old["miss_m"], old["tca"], old["rel_vel_ms"]),
                "tca": _iso(old["tca"]),
                "safe": old["miss_m"] >= required,
            }
            criteria = {
                "target_miss_m": round(target, 1),
                "pc_threshold": threshold,
                "pc_required_miss_m": None if pc_req is None else round(pc_req, 1),
                "required_miss_m": round(required, 1),
                "max_dv_ms": dv_cap,
                "rule": PLAN_RULE,
            }
            if baseline["safe"]:
                break   # already clears both criteria: nothing to plan

        found = min_dv_for_miss(ctx, required, dv_cap, PLAN_DV_TOLERANCE_MS)
        dv = found["dv_ms"]
        new = conj_api.evaluate_maneuver(ctx, dv)
        new_pc = pc_at(new["new_miss_m"], new["new_tca"], new["rel_vel_ms"])
        options.append({
            **entry,
            "dv_ms": round(dv, 4),
            "direction": ("prograde" if dv > 0
                          else "retrograde" if dv < 0 else "none"),
            "new_miss_m": round(new["new_miss_m"], 1),
            "delta_miss_m": round(new["new_miss_m"] - old["miss_m"], 1),
            "new_tca": _iso(new["new_tca"]),
            "new_pc": new_pc,
            "reachable": found["reachable"],
            "meets_miss": new["new_miss_m"] >= target,
            "meets_pc": None if new_pc is None else new_pc < threshold,
            "propellant_g_per_100kg": round(propellant_g(dv), 2),
            "evaluations": found["evaluations"],
        })

    recommended = None
    feasible = [o for o in options if o["reachable"]]
    if feasible:
        smallest = min(abs(o["dv_ms"]) for o in feasible)
        ties = [o for o in feasible if abs(o["dv_ms"]) - smallest <= PLAN_TIE_MS]
        recommended = min(ties, key=lambda o: o["lead_s"])   # later burn

    # Honesty guard: the plan runs on the latest-TLE geometry (as does the
    # simulator). When that geometry says "already safe" but the screened
    # miss of the latest message says otherwise, never report an all-clear.
    summary = conj_api.query_summaries(conn, "c.id = ?", (conjunction_id,))
    screened = summary[0]["miss_distance_m"] if summary else None
    baseline["screened_miss_m"] = None if screened is None else round(screened, 1)
    note = None
    if baseline["safe"]:
        if screened is not None and screened < required:
            status = "geometry_mismatch"
            note = ("The latest TLEs put this pair much farther apart at "
                    "closest approach than the latest screening message "
                    "does, so no burn can be planned from them. Wait for "
                    "fresher tracking.")
        else:
            status = "no_burn_needed"
    elif recommended is not None:
        status = "burn_recommended"
    else:
        status = "unreachable"
        note = (f"No along-track burn up to {dv_cap:g} m/s at these lead "
                "times reaches the required miss distance.")

    return {
        "conjunction_id": row["id"],
        "tca": _iso(tca),
        "computed_at": _iso(now),
        "status": status,
        "note": note,
        "baseline": baseline,
        "criteria": criteria,
        "recommended": recommended,
        "options": options,
        "skipped": skipped,
        "assumptions": {
            "isp_s": PLAN_ISP_S,
            "reference_mass_kg": PLAN_REF_MASS_KG,
            "primary_period_s": round(period_s, 1),
            "pc_model": "assumed-sigma (TLE-age growth), as the maneuver sandbox",
        },
        "method": PLAN_METHOD,
        "method_detail": conj_api.MANEUVER_METHOD,
    }
