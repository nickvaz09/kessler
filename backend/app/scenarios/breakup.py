"""Illustrative debris-cloud model: NASA Standard Breakup Model, simplified.

EVERYTHING HERE IS A MODEL, NOT OBSERVED DATA. Every payload built from it is
labeled MODEL_LABEL. Reference: Johnson, Krisko, Liou & Anz-Meador, "NASA's
new breakup model of EVOLVE 4.0", Adv. Space Res. 28(9), 2001 (the SBM).

Breakup (catastrophic collision branch of the SBM)
--------------------------------------------------
- Catastrophic when the projectile's specific energy on the target,
  E* = 0.5 * m_p * v_rel^2 / m_t, exceeds 40 J/g (m_p = lighter body).
- Fragment count above characteristic length Lc [m]:
      N(>Lc) = 0.1 * M^0.75 * Lc^-1.71,   M = m_1 + m_2 [kg]
  Lengths are sampled from that power law (inverse CDF), capped at 2 m.
  Fragments are split between the parents in proportion to mass
  (simplification: the SBM describes the cloud as a whole).
- Area-to-mass: the SBM spacecraft-fragment distribution for L > 11 cm, a
  two-component mixture in chi = log10(A/m) with weights and moments that
  are piecewise-linear in lambda = log10(L) (alpha, mu1, sigma1, mu2, sigma2
  below).
- Ejection speed: log10(dv [m/s]) ~ N(0.9 chi + 2.9, 0.4) (collision
  branch), in an isotropic random direction, added to the parent's velocity.

Propagation (illustrative; not SGP4)
------------------------------------
Each fragment's osculating Keplerian elements at the collision instant are
advanced with J2 secular rates only:
    dRAAN/dt = -1.5 n J2 (Re/p)^2 cos i
    dargp/dt =  0.75 n J2 (Re/p)^2 (5 cos^2 i - 1)
    dM/dt    =  n + 0.75 n J2 (Re/p)^2 sqrt(1 - e^2) (3 cos^2 i - 1)
Differential mean motion spreads the cloud along-track (a point becomes a
ring within about a day); differential nodal precession fans the rings
apart over months. A simple drag term lowers each orbit on an analytic
exponential-atmosphere decay, h(t) = h0 + H ln(1 - t/tau) with
tau = H / (B rho(h_eff) sqrt(mu a)), B = Cd A/m (Cd = 2.2), density from
the Vallado exponential atmosphere table; a fragment counts as re-entered
once its perigee falls below 120 km. Solar activity, third bodies, SRP and
short-period terms are ignored.

Deterministic: a fixed seed per scenario keeps the cloud identical between
requests and between the server and the page.
"""
import math

import numpy as np

try:
    from backend.app.ingestion.propagate import MU_KM3_S2, RE_KM
except ImportError:  # propagate swapped for a stub (API tests): same constants
    MU_KM3_S2, RE_KM = 398600.4418, 6378.137

MODEL_LABEL = "illustrative model (NASA Standard Breakup Model, simplified)"
MODEL_VERSION = "sbm-simplified-1"
J2 = 1.08262668e-3
CATASTROPHIC_J_PER_G = 40.0
LC_M = 0.10                 # characteristic length floor: 10 cm
L_MAX_M = 2.0
CD = 2.2
REENTRY_ALT_KM = 120.0
DEFAULT_SEED = 20090210

# Vallado exponential atmosphere (base altitude km, density kg/m^3, scale
# height km) - "Fundamentals of Astrodynamics and Applications", table 8-4.
_ATMOS = (
    (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00),
)

FRAGMENT_FIELDS = ["parent", "length_m", "area_to_mass", "dv_ms", "a_km", "e",
                   "i_rad", "raan_rad", "argp_rad", "m0_rad", "raan_dot",
                   "argp_dot", "m_dot", "tau_s", "scale_h_km"]


def atmosphere(h_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(density kg/m^3, scale height km) at altitude(s) h_km."""
    h = np.atleast_1d(np.asarray(h_km, dtype=float))
    bases = np.array([b for b, _r, _h in _ATMOS], dtype=float)
    idx = np.clip(np.searchsorted(bases, h, side="right") - 1, 0, len(_ATMOS) - 1)
    rho0 = np.array([r for _b, r, _h in _ATMOS])[idx]
    sh = np.array([s for _b, _r, s in _ATMOS])[idx]
    return rho0 * np.exp(-(h - bases[idx]) / sh), sh


def specific_energy_j_per_g(m1_kg: float, m2_kg: float, v_rel_ms: float) -> float:
    m_p, m_t = sorted((m1_kg, m2_kg))
    return 0.5 * m_p * v_rel_ms ** 2 / (m_t * 1000.0)


def fragment_count(total_mass_kg: float, lc_m: float = LC_M) -> int:
    """SBM collision count N(>Lc) = 0.1 M^0.75 Lc^-1.71 (catastrophic)."""
    return int(round(0.1 * total_mass_kg ** 0.75 * lc_m ** -1.71))


def _pw(lam, x0, y0, x1, y1):
    """Piecewise-linear ramp: y0 below x0, y1 above x1, linear between."""
    t = np.clip((lam - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + (y1 - y0) * t


def sample_area_to_mass(lengths_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SBM spacecraft-fragment A/m [m^2/kg] for L > 11 cm (bimodal in chi)."""
    lam = np.log10(lengths_m)
    alpha = np.where(lam <= -1.95, 0.0, np.where(lam >= 0.55, 1.0,
                                                 0.3 + 0.4 * (lam + 1.2)))
    alpha = np.clip(alpha, 0.0, 1.0)
    mu1 = _pw(lam, -1.1, -0.6, 0.0, -0.95)
    sg1 = _pw(lam, -1.3, 0.1, -0.3, 0.3)
    mu2 = _pw(lam, -0.7, -1.2, -0.1, -2.0)
    sg2 = _pw(lam, -0.5, 0.5, 0.0, 0.3)
    first = rng.random(lam.size) < alpha
    chi = np.where(first, rng.normal(mu1, sg1), rng.normal(mu2, sg2))
    return 10.0 ** chi


def sample_delta_v(area_to_mass: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SBM collision ejection speed [m/s]: log10 dv ~ N(0.9 chi + 2.9, 0.4)."""
    chi = np.log10(area_to_mass)
    return 10.0 ** rng.normal(0.9 * chi + 2.9, 0.4)


def elements_from_state(r: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized osculating Keplerian elements from TEME states (km, km/s)."""
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)
    mu = MU_KM3_S2
    rn = np.linalg.norm(r, axis=1)
    vn2 = np.einsum("ij,ij->i", v, v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    k = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k, h)
    nn = np.linalg.norm(nvec, axis=1)
    evec = ((vn2 - mu / rn)[:, None] * r
            - np.einsum("ij,ij->i", r, v)[:, None] * v) / mu
    e = np.linalg.norm(evec, axis=1)
    energy = vn2 / 2.0 - mu / rn
    a = -mu / (2.0 * energy)
    inc = np.arccos(np.clip(h[:, 2] / hn, -1.0, 1.0))
    raan = np.arctan2(nvec[:, 1], nvec[:, 0]) % (2 * math.pi)
    safe_e = np.where(e > 1e-9, e, 1e-9)
    safe_n = np.where(nn > 1e-9, nn, 1e-9)
    argp = np.arccos(np.clip(np.einsum("ij,ij->i", nvec, evec) / (safe_n * safe_e), -1, 1))
    argp = np.where(evec[:, 2] < 0.0, 2 * math.pi - argp, argp)
    nu = np.arccos(np.clip(np.einsum("ij,ij->i", evec, r) / (safe_e * rn), -1, 1))
    nu = np.where(np.einsum("ij,ij->i", r, v) < 0.0, 2 * math.pi - nu, nu)
    ea = 2.0 * np.arctan2(np.sqrt(np.clip(1 - e, 0, None)) * np.sin(nu / 2),
                          np.sqrt(1 + e) * np.cos(nu / 2))
    m = (ea - e * np.sin(ea)) % (2 * math.pi)
    return {"a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "m": m}


def secular_rates(a, e, i):
    """J2 secular (raan_dot, argp_dot, m_dot) in rad/s."""
    n = np.sqrt(MU_KM3_S2 / a ** 3)
    p = a * (1.0 - e * e)
    k = 0.75 * n * J2 * (RE_KM / p) ** 2
    ci = np.cos(i)
    return (-2.0 * k * ci, k * (5.0 * ci * ci - 1.0),
            n + k * np.sqrt(1.0 - e * e) * (3.0 * ci * ci - 1.0))


def decay_params(a_km, e, area_to_mass):
    """Analytic decay (tau seconds, scale height km) at the effective perigee."""
    hp = a_km * (1.0 - e) - RE_KM
    rho, sh = atmosphere(hp)
    # Elliptical orbits feel the density mostly near perigee: scale by the
    # fraction of the orbit spent within ~one scale height of it.
    frac = np.minimum(1.0, np.sqrt(sh / (2.0 * math.pi * np.maximum(a_km * e, 1e-6))))
    b = CD * area_to_mass
    da_dt = b * rho * frac * np.sqrt(MU_KM3_S2 * 1e9 * a_km * 1e3)   # m/s
    tau = np.where(da_dt > 0, sh * 1e3 / np.maximum(da_dt, 1e-30), np.inf)
    return tau, sh


def generate(parents: list[dict], v_rel_ms: float, seed: int = DEFAULT_SEED,
             lc_m: float = LC_M) -> dict:
    """Fragment cloud at the collision instant.

    parents: [{norad_id, name, mass_kg, r: [km], v: [km/s]}] (TEME at TCA).
    Returns {"counts", "energy", "fragments": np rows (FRAGMENT_FIELDS),
             "parents": elements of the intact parents}.
    """
    rng = np.random.default_rng(seed)
    masses = [float(p["mass_kg"]) for p in parents]
    total_mass = sum(masses)
    e_star = specific_energy_j_per_g(masses[0], masses[1], v_rel_ms)
    catastrophic = e_star >= CATASTROPHIC_J_PER_G
    n_total = fragment_count(total_mass, lc_m) if catastrophic else 0
    shares = [int(round(n_total * m / total_mass)) for m in masses]
    shares[-1] = n_total - sum(shares[:-1])

    rows = []
    for idx, (parent, count) in enumerate(zip(parents, shares)):
        if count <= 0:
            continue
        u = rng.random(count)
        lengths = lc_m * u ** (-1.0 / 1.71)
        over = lengths > L_MAX_M
        while over.any():
            lengths[over] = lc_m * rng.random(int(over.sum())) ** (-1.0 / 1.71)
            over = lengths > L_MAX_M
        am = sample_area_to_mass(lengths, rng)
        dv = sample_delta_v(am, rng)
        dirs = rng.normal(size=(count, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        v = np.asarray(parent["v"], dtype=float)[None, :] + dirs * (dv[:, None] / 1000.0)
        r = np.repeat(np.asarray(parent["r"], dtype=float)[None, :], count, axis=0)
        el = elements_from_state(r, v)
        raan_dot, argp_dot, m_dot = secular_rates(el["a"], el["e"], el["i"])
        tau, sh = decay_params(el["a"], el["e"], am)
        block = np.column_stack([
            np.full(count, idx, dtype=float), lengths, am, dv, el["a"], el["e"],
            el["i"], el["raan"], el["argp"], el["m"], raan_dot, argp_dot, m_dot,
            tau, sh,
        ])
        # Bound (elliptical) orbits only: hyperbolic ejecta are not physical
        # at these speeds, but guard anyway.
        rows.append(block[(el["e"] < 1.0) & (el["a"] > 0)])
    frags = np.vstack(rows) if rows else np.empty((0, len(FRAGMENT_FIELDS)))

    parent_rows = []
    for parent in parents:
        el = elements_from_state(np.asarray(parent["r"], dtype=float),
                                 np.asarray(parent["v"], dtype=float))
        rd, ad, md = secular_rates(el["a"], el["e"], el["i"])
        parent_rows.append({
            "norad_id": parent["norad_id"], "name": parent["name"],
            "elements": {
                "a_km": float(el["a"][0]), "e": float(el["e"][0]),
                "i_rad": float(el["i"][0]), "raan_rad": float(el["raan"][0]),
                "argp_rad": float(el["argp"][0]), "m0_rad": float(el["m"][0]),
                "raan_dot": float(rd[0]), "argp_dot": float(ad[0]),
                "m_dot": float(md[0]),
            },
        })
    return {
        "energy": {"specific_energy_j_per_g": round(e_star, 1),
                   "catastrophic_threshold_j_per_g": CATASTROPHIC_J_PER_G,
                   "catastrophic": bool(catastrophic)},
        "counts": {"modeled_total_ge_10cm": n_total,
                   "by_parent": {str(p["norad_id"]): s for p, s in zip(parents, shares)},
                   "total_mass_kg": total_mass},
        "fragments": frags,
        "parents": parent_rows,
    }


def altitude_drop_km(frags: np.ndarray, t_s: float) -> np.ndarray:
    """Analytic decay: h drop at time t (<= 0); -inf once re-entered."""
    tau = frags[:, 13]
    sh = frags[:, 14]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 1.0 - t_s / tau
        drop = np.where(x > 0, sh * np.log(np.where(x > 0, x, 1.0)), -np.inf)
    return drop


def in_orbit_mask(frags: np.ndarray, t_s: float) -> np.ndarray:
    a = frags[:, 4] + altitude_drop_km(frags, t_s)
    hp = a * (1.0 - frags[:, 5]) - RE_KM
    return np.isfinite(a) & (hp > REENTRY_ALT_KM)


def positions(frags: np.ndarray, t_s: float) -> np.ndarray:
    """TEME positions [km] at t seconds after the collision; NaN if re-entered."""
    with np.errstate(invalid="ignore"):
        return _positions(frags, t_s)


def _positions(frags: np.ndarray, t_s: float) -> np.ndarray:
    a = frags[:, 4] + altitude_drop_km(frags, t_s)
    e = frags[:, 5]
    i = frags[:, 6]
    raan = frags[:, 7] + frags[:, 10] * t_s
    argp = frags[:, 8] + frags[:, 11] * t_s
    m = (frags[:, 9] + frags[:, 12] * t_s) % (2 * math.pi)
    ea = m.copy()
    for _ in range(8):   # Newton on Kepler's equation (e << 1 converges fast)
        ea = ea - (ea - e * np.sin(ea) - m) / (1.0 - e * np.cos(ea))
    xp = a * (np.cos(ea) - e)
    yp = a * np.sqrt(1.0 - e * e) * np.sin(ea)
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(i), np.sin(i)
    x = (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp
    y = (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    out = np.column_stack([x, y, z])
    out[~in_orbit_mask(frags, t_s)] = np.nan
    return out


def altitude_band_km(frags: np.ndarray, lo_pct: float = 10.0,
                     hi_pct: float = 90.0) -> tuple[float, float]:
    """Altitude shell the cloud occupies: the 10th..90th percentile of the
    mean altitude (a - Re) of fragments still in orbit just after the
    breakup, rounded outward to 10 km. Percentiles keep the few very
    high-dv ejecta from stretching the shell across all of LEO."""
    live = frags[in_orbit_mask(frags, 0.0)] if frags.size else frags
    if live.size == 0:
        return (0.0, 0.0)
    h = live[:, 4] - RE_KM
    lo = math.floor(float(np.percentile(h, lo_pct)) / 10.0) * 10.0
    hi = math.ceil(float(np.percentile(h, hi_pct)) / 10.0) * 10.0
    return (lo, hi)
