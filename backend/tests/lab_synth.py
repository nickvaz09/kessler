"""Synthetic Kelvins-shaped CDM data for VIGIL Lab tests (not real data).

Each event has a fixed true encounter geometry; successive CDMs carry a
noisy estimate of the miss vector and a covariance that shrinks toward TCA,
and the ``risk`` column is computed with the same short-encounter formula
the lab's physics features use. That gives the tests a world where the
physics features are informative and quiet events can genuinely escalate
(dilution region + shrinking covariance), so the harness can be checked
end to end without the ESA files. Never use it to claim model performance.
"""
from __future__ import annotations

import math

import numpy as np

HBR = 10.0  # m


def _risk(miss_vec, v, cov):
    vh = v / np.linalg.norm(v)
    rp = miss_vec - np.dot(miss_vec, vh) * vh
    d = np.linalg.norm(rp)
    if d < 1e-6:
        a = np.array([1.0, 0, 0]) if abs(vh[0]) < 0.9 else np.array([0, 1.0, 0])
        rp = a - np.dot(a, vh) * vh
    u1 = rp / np.linalg.norm(rp)
    u2 = np.cross(vh, u1)
    B = np.vstack([u1, u2])
    C2 = B @ cov @ B.T
    m2 = float(np.array([d, 0]) @ np.linalg.solve(C2, np.array([d, 0])))
    pc = HBR ** 2 / (2 * math.sqrt(np.linalg.det(C2))) * math.exp(-m2 / 2)
    r = math.log10(pc) if pc > 1e-30 else -30.0
    pcmax = r + math.log10(2 / (math.e * max(m2, 1e-9))) + (max(m2, 1e-9) / 2) / math.log(10)
    return max(-30.0, min(0.0, r)), max(-30.0, min(0.0, pcmax)), math.sqrt(m2)


def frame(n_events: int = 600, seed: int = 0, start_id: int = 0):
    import pandas as pd

    rng = np.random.RandomState(seed)
    rows = []
    for e in range(n_events):
        eid = start_id + e
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * rng.uniform(1000, 14000)
        vh = v / np.linalg.norm(v)
        a = rng.normal(size=3); a -= np.dot(a, vh) * vh; a /= np.linalg.norm(a)
        d_true = float(np.exp(rng.normal(math.log(5000), 1.3)))
        true_miss = a * d_true
        s0t, s0c = float(np.exp(rng.normal(math.log(800), 0.5))), float(np.exp(rng.normal(math.log(3000), 0.6)))
        final_tta = float(rng.uniform(0.05, 1.6))
        first_tta = float(rng.uniform(2.2, 7.0)) if rng.rand() > 0.08 else float(rng.uniform(1.7, 2.1))
        n = int(rng.randint(3, 14))
        ttas = np.sort(rng.uniform(final_tta, first_tta, n - 2))[::-1]
        ttas = np.concatenate([[first_tta], ttas, [final_tta]])
        ctype = rng.choice(["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN"])
        # Realism: each object's covariance lives in ITS OWN RTN frame (as in
        # real CDMs). The chaser frame is the target frame rotated about the
        # radial axis by the crossing angle; the risk below uses the properly
        # rotated sum, while the lab's physics features use the documented
        # unrotated approximation - so physics features are informative but
        # not a perfect copy of the target, as with real data.
        theta = float(rng.uniform(0.2, np.pi - 0.2))
        Rz = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        rho_t, rho_c = float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))
        for tta in ttas:
            f = 0.06 + 0.94 * min(1.0, tta / 7.0) ** 1.5   # covariance shrinks toward TCA
            st = s0t * f * float(np.exp(rng.normal(0, 0.05)))
            sc = s0c * f * float(np.exp(rng.normal(0, 0.05)))
            def own(sig, rho):
                s3 = np.array([0.2 * sig, sig, 0.3 * sig])
                c = np.eye(3); c[0, 1] = c[1, 0] = rho
                return c * np.outer(s3, s3)
            ct, cc_own = own(st, rho_t), own(sc, rho_c)
            cov = ct + Rz @ cc_own @ Rz.T
            est = true_miss + rng.multivariate_normal(np.zeros(3), 2.0 * cov)
            risk, pcmax, maha = _risk(est, v, cov)
            rows.append({
                "event_id": eid, "time_to_tca": float(tta), "mission_id": int(eid % 7),
                "risk": risk, "max_risk_estimate": pcmax, "max_risk_scaling": float(rng.uniform(1, 10)),
                "miss_distance": float(np.linalg.norm(est)), "relative_speed": float(np.linalg.norm(v)),
                "relative_position_r": est[0], "relative_position_t": est[1], "relative_position_n": est[2],
                "relative_velocity_r": v[0], "relative_velocity_t": v[1], "relative_velocity_n": v[2],
                "t_sigma_r": 0.2 * st, "t_sigma_t": st, "t_sigma_n": 0.3 * st,
                "c_sigma_r": 0.2 * sc, "c_sigma_t": sc, "c_sigma_n": 0.3 * sc,
                "t_ct_r": rho_t, "t_cn_r": 0.0, "t_cn_t": 0.0,
                "c_ct_r": rho_c, "c_cn_r": 0.0, "c_cn_t": 0.0,
                "mahalanobis_distance": maha, "c_object_type": ctype,
                "t_obs_used": int(rng.randint(50, 400)), "c_obs_used": int(rng.randint(5, 80)),
                "F10": float(rng.uniform(70, 200)), "AP": float(rng.uniform(2, 40)),
            })
    return pd.DataFrame(rows)
