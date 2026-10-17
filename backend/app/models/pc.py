"""Collision probability from screening geometry - 2-D encounter-plane model.

Formula
-------
In the short-encounter (rectilinear) approximation the relative motion during
the conjunction is fast and straight, so the collision probability is the
integral of the 2-D Gaussian relative-position PDF over the hard-body circle
in the encounter plane (the plane perpendicular to the relative velocity).
With an isotropic combined covariance (equal sigmas, zero correlation) and a
hard-body radius small relative to sigma, that integral reduces to the
standard isotropic approximation:

    Pc ~= (HBR^2 / (2 * sigma^2)) * exp(-d^2 / (2 * sigma^2))

where
    d      miss distance at TCA [m]
    HBR    combined hard-body radius [m]        (config.HARD_BODY_RADIUS_M)
    sigma  1-sigma isotropic *combined* relative-position uncertainty [m]

Assumptions (documented deviations from a full CDM-covariance Pc):
- TLE-class covariance. TLE/SGP4 states carry no covariance, so we assume a
  per-object 1-sigma position error that grows linearly with propagation age:
  sigma_at(age) = config.SIGMA_BASE_M + config.SIGMA_GROWTH_M_PER_DAY * age.
- Both objects share that uncertainty and are independent, so the combined
  relative-position variance is 2 * sigma_at^2, i.e.
  sigma = sqrt(2) * sigma_at(age).
- The covariance is isotropic in the encounter plane (circular, uncorrelated).
  Under isotropy the relative-velocity direction drops out of the result;
  ``relative_velocity_ms`` is accepted for interface symmetry with real-CDM
  Pc computations but is unused here.
- The closed form can exceed 1 when sigma <~ HBR/sqrt(2); the result is
  clamped to [0, 1].

Everything here is a pure, deterministic function of its inputs.

Covariance-aware path
---------------------
When a Space-Track restricted-class CDM supplies real per-object RTN
covariances (cdm_messages.covariance_json - see ingestion/spacetrack.py),
``compute_pc_from_covariance`` replaces the isotropic assumption with a
Foster-style 2-D Pc on the B-plane using the projected real covariance. The
assumed-sigma ``compute_pc`` remains the default whenever no covariance is
present.
"""
import math

import numpy as np

from backend.app import config


def sigma_at(propagation_age_days: float) -> float:
    """Per-object 1-sigma position uncertainty [m] at a given propagation age.

    Linear TLE-class error growth: SIGMA_BASE_M + SIGMA_GROWTH_M_PER_DAY * age.
    Negative ages are clamped to 0 (uncertainty never shrinks below epoch).
    """
    age = max(float(propagation_age_days), 0.0)
    return config.SIGMA_BASE_M + config.SIGMA_GROWTH_M_PER_DAY * age


def compute_pc(miss_distance_m: float, relative_velocity_ms: float,
               propagation_age_days: float = 1.0,
               hard_body_radius_m: float | None = None) -> float:
    """Isotropic 2-D encounter-plane collision probability, clamped to [0, 1].

    Pc = (HBR^2 / (2 sigma^2)) * exp(-d^2 / (2 sigma^2)) with
    sigma = sqrt(2) * sigma_at(propagation_age_days). See module docstring for
    the derivation and assumptions. ``relative_velocity_ms`` is unused under
    the isotropic assumption (kept for interface symmetry). Deterministic:
    Pc is maximal at d = 0 and strictly decreasing in miss distance.
    """
    hbr = config.HARD_BODY_RADIUS_M if hard_body_radius_m is None else float(hard_body_radius_m)
    d = abs(float(miss_distance_m))
    sigma = math.sqrt(2.0) * sigma_at(propagation_age_days)
    if sigma <= 0.0:  # degenerate covariance: point mass at the nominal state
        return 1.0 if d < hbr else 0.0
    two_var = 2.0 * sigma * sigma
    try:
        pc = (hbr * hbr / two_var) * math.exp(-(d * d) / two_var)
    except OverflowError:  # enormous d/sigma ratios underflow to 0 anyway
        return 0.0
    return min(1.0, max(0.0, pc))


# --- Covariance-aware Pc (Foster-style B-plane integration) -------------------

# Fixed 400-point polar quadrature: 20 Gauss-Legendre radial nodes x 20
# uniform (midpoint) angular nodes. Uniform sampling of a smooth periodic
# integrand converges spectrally; GL-20 resolves radial variation down to
# ~HBR/10 - hence the eigenvalue floor below.
_QUAD_RADIAL = 20
_QUAD_ANGULAR = 20
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_QUAD_RADIAL)

# Singular-covariance guard: each in-plane eigenvalue is floored at
# (HBR * _PLANE_SIGMA_FLOOR_FRACTION)^2. This keeps degenerate/rank-deficient
# covariances invertible and keeps the fixed quadrature honest - a Gaussian
# narrower than the radial node spacing would silently integrate to ~0, so
# the floor pins the point-mass limit to its correct values (~1 inside the
# hard-body disk, ~0 outside). Real CDM in-plane sigmas are meters and up;
# with HBR = 10 m the floor is ~1.7 m, i.e. a documented no-op in practice.
_PLANE_SIGMA_FLOOR_FRACTION = 1.0 / 6.0


def compute_pc_from_covariance(ric_m: dict, cov_rtn_combined,
                               hbr_m: float | None = None) -> float:
    """Foster-style 2-D collision probability from a REAL RTN covariance.

    Derivation
    ----------
    In the short-encounter (rectilinear) approximation the relative motion is
    fast and straight, so Pc is the integral of the 2-D Gaussian relative-
    position PDF over the hard-body disk in the B-plane - the plane through
    the primary perpendicular to the relative velocity:

        Pc = ∫∫_{|x| <= HBR}  N2(x; m, P)  dA

    with m the projected miss vector and P the projected 2x2 combined
    covariance. Construction, step by step:

    1. COMBINE. The two objects' position errors are independent, so the
       relative-position covariance is the sum C = C1 + C2 of the per-object
       3x3 RTN covariances. Approximation (documented): both matrices are
       treated as expressed in the primary's RIC axes - the CDM transmits
       each in its own object's RTN frame, but without the full states the
       exact inter-frame rotation is unavailable, and over the short
       encounter the frames are near-aligned for LEO geometries.
    2. B-PLANE BASIS from the RIC miss components alone. At TCA the miss
       vector dr is perpendicular to the relative velocity (d|dr|^2/dt = 0),
       so dr already lies in the B-plane. For orbit-crossing LEO conjunctions
       both velocities are near-horizontal, so v_rel has negligible radial
       component; the plane perpendicular to it then contains the radial axis
       R_hat and the horizontal direction of the miss itself,
       h_hat = (0, I, C)/h with h = sqrt(I^2 + C^2) (h_hat is perpendicular
       to the horizontal part of v_rel precisely because dr . v_rel = 0).
       Basis U = [R_hat; h_hat] (2x3); when h = 0 (purely radial miss) the
       in-track axis is taken for h_hat - a deterministic, documented
       convention for the degenerate case.
    3. PROJECT.  m = (R, h)  and  P = U C U^T  (2x2, m^2).
    4. GUARD (singular covariance). P is symmetrized, its eigenvalues are
       computed in closed form, and both are floored by adding c*I with
       c = max(0, floor - lambda_min), floor = (HBR/6)^2 - see
       _PLANE_SIGMA_FLOOR_FRACTION. Regularization note: rank-deficient or
       zero covariances therefore return the point-mass limit (~1 for a miss
       inside the hard-body disk, ~0 outside) instead of crashing or
       silently under-integrating.
    5. INTEGRATE. The Gaussian is integrated over the HBR disk by a fixed
       400-point polar quadrature (20-node Gauss-Legendre in r on [0, HBR],
       Jacobian r included; 20 midpoint angular nodes weighted 2*pi/20).
       Deterministic - no randomness anywhere.

    For an isotropic P = sigma^2 I and HBR << sigma this reduces to the
    compute_pc closed form (HBR^2 / (2 sigma^2)) * exp(-d^2 / (2 sigma^2)).
    The result is clamped to [0, 1].

    Parameters
    ----------
    ric_m : dict
        Secondary-minus-primary miss components at TCA in the primary's RIC
        frame, meters: {"radial": R, "in_track": I, "cross_track": C} (the
        /conjunctions/{id}/ephemeris tca_state.ric_m shape). Missing keys
        default to 0.0.
    cov_rtn_combined :
        One of (axis order R/T/N = RIC, units m^2 throughout):
        - a (sat1, sat2) pair of 3x3 per-object RTN covariances (summed),
        - a covariance_json-style dict {"sat1_rtn": 3x3, "sat2_rtn": 3x3},
        - a single 3x3 already-combined RTN covariance,
        - a 2x2 covariance already projected on the (R_hat, h_hat) B-plane
          basis defined above.
    hbr_m : float, optional
        Combined hard-body radius [m]; defaults to config.HARD_BODY_RADIUS_M.
        Non-positive radii return 0.0.
    """
    hbr = config.HARD_BODY_RADIUS_M if hbr_m is None else float(hbr_m)
    if hbr <= 0.0:
        return 0.0
    r = float(ric_m.get("radial", 0.0) or 0.0)
    i = float(ric_m.get("in_track", 0.0) or 0.0)
    c = float(ric_m.get("cross_track", 0.0) or 0.0)
    h = math.hypot(i, c)

    # accepted covariance forms -> projected in-plane 2x2 (step 1-3)
    if isinstance(cov_rtn_combined, dict):
        arr = (np.asarray(cov_rtn_combined["sat1_rtn"], dtype=float)
               + np.asarray(cov_rtn_combined["sat2_rtn"], dtype=float))
    else:
        arr = np.asarray(cov_rtn_combined, dtype=float)
        if arr.shape == (2, 3, 3):  # (sat1, sat2) pair
            arr = arr[0] + arr[1]
    if arr.shape == (3, 3):
        u_h = np.array([0.0, i / h, c / h]) if h > 0.0 \
            else np.array([0.0, 1.0, 0.0])  # degenerate: in-track convention
        basis = np.vstack([[1.0, 0.0, 0.0], u_h])
        plane = basis @ arr @ basis.T
    elif arr.shape == (2, 2):
        plane = arr.copy()
    else:
        raise ValueError(
            "covariance must be a 2x2 in-plane matrix, a 3x3 RTN matrix, a"
            " (sat1, sat2) pair of 3x3 matrices, or a covariance_json dict")

    # Step 4: symmetrize + eigenvalue floor (closed form for symmetric 2x2).
    plane = 0.5 * (plane + plane.T)
    half_tr = 0.5 * (plane[0, 0] + plane[1, 1])
    det = plane[0, 0] * plane[1, 1] - plane[0, 1] * plane[1, 0]
    root = math.sqrt(max(half_tr * half_tr - det, 0.0))
    lam_min = half_tr - root
    floor = (hbr * _PLANE_SIGMA_FLOOR_FRACTION) ** 2
    if lam_min < floor:
        plane = plane + (floor - lam_min) * np.eye(2)

    det = plane[0, 0] * plane[1, 1] - plane[0, 1] ** 2
    inv00 = plane[1, 1] / det
    inv01 = -plane[0, 1] / det
    inv11 = plane[0, 0] / det
    norm = 1.0 / (2.0 * math.pi * math.sqrt(det))

    # step 5: fixed 400-point polar quadrature over the HBR disk
    radii = 0.5 * hbr * (_GL_NODES + 1.0)             # (20,) on [0, HBR]
    w_r = 0.5 * hbr * _GL_WEIGHTS                     # GL weights, mapped
    thetas = (np.arange(_QUAD_ANGULAR) + 0.5) * (2.0 * math.pi / _QUAD_ANGULAR)
    w_theta = 2.0 * math.pi / _QUAD_ANGULAR
    dx = radii[:, None] * np.cos(thetas)[None, :] - r  # x - m, (20, 20)
    dy = radii[:, None] * np.sin(thetas)[None, :] - h
    quad_form = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
    density = np.exp(-0.5 * quad_form)                # underflows cleanly to 0
    pc = norm * w_theta * float(np.sum((w_r * radii)[:, None] * density))
    return min(1.0, max(0.0, pc))
