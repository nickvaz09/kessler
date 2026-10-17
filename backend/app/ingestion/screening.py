"""Conjunction screening: coarse all-vs-primaries pass + fine TCA refinement.

Coarse pass propagates every LEO catalog object on a shared time grid with
SatrecArray (chunked along the time axis to bound memory) and computes
primary-vs-all distances per step with numpy. Because a fast crossing is under
threshold for only a few seconds, sampled distance alone aliases badly on a
60 s grid, so candidates are gated two ways per step:

  1. sampled distance under the reporting threshold, and
  2. the rectilinear minimum over the segment between consecutive samples
     (min_tau |dr + dv*tau|, closed form from the relative state at the
     segment start) under a capture radius = threshold + a curvature margin.

Candidates are refined on a 1 s grid around the coarse minimum (half-window
adapted to the coarse step) with parabolic interpolation of d^2 through the
discrete minimum for sub-second TCA/miss distance, then filtered to keep only
events whose refined miss distance is under the reporting threshold.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, SatrecArray

from backend.app import config
from backend.app.ingestion.celestrak import ISO_FMT
from backend.app.ingestion.propagate import RE_KM, _semi_major_axis_km, jday_arrays

TIME_CHUNK_STEPS = 500          # coarse steps propagated per SatrecArray call
REFINE_HALF_WINDOW_S = 90       # minimum fine-search half-window around coarse minimum
REFINE_STEP_S = 1.0
TCA_MATCH_TOLERANCE_S = 30 * 60  # same-event TCA tolerance in record_events
# The segment-minimum gate is rectilinear; differential gravity bends the
# relative track by up to a few km over a 60 s segment, so capture wider than
# the reporting threshold and let the refinement pass filter exactly.
COARSE_CAPTURE_MARGIN_KM = 5.0


@dataclass
class ConjunctionEvent:
    primary_norad: int
    secondary_norad: int
    tca: datetime
    miss_distance_m: float
    relative_velocity_ms: float


def _latest_tles(conn) -> list:
    return conn.execute(
        """
        SELECT s.norad_id AS norad_id, t.line1 AS line1, t.line2 AS line2
        FROM satellites s
        JOIN tle_history t ON t.satellite_id = s.id
        WHERE t.id = (
            SELECT t2.id FROM tle_history t2
            WHERE t2.satellite_id = s.id
            ORDER BY t2.epoch DESC, t2.id DESC LIMIT 1
        )
        """
    ).fetchall()


def _apogee_km(sat: Satrec) -> float:
    a = _semi_major_axis_km(sat.no_kozai)
    return a * (1.0 + sat.ecco) - RE_KM


def _refine(sat_p: Satrec, sat_s: Satrec, t_coarse: datetime,
            start: datetime, end: datetime,
            half_window_s: float = REFINE_HALF_WINDOW_S,
            ) -> tuple[datetime, float, float] | None:
    """1 s fine grid around the coarse minimum, clamped to the window.

    The discrete minimum is sharpened by parabolic interpolation of d^2
    through its neighbors: near TCA the encounter is locally rectilinear, so
    d^2(t) is quadratic and the vertex gives sub-second TCA and the true miss
    distance (the raw 1 s grid overstates it by up to |v_rel| * 0.5 s).

    Returns (tca, miss_distance_m, relative_velocity_ms) or None if both
    objects fail to propagate anywhere on the fine grid.
    """
    lo = max(start, t_coarse - timedelta(seconds=half_window_s))
    hi = min(end, t_coarse + timedelta(seconds=half_window_s))
    n = int((hi - lo).total_seconds() / REFINE_STEP_S) + 1
    times = [lo + timedelta(seconds=REFINE_STEP_S * i) for i in range(n)]
    jd, fr = jday_arrays(times)
    ep, rp, vp = sat_p.sgp4_array(jd, fr)
    es, rs, vs = sat_s.sgp4_array(jd, fr)
    ok = (ep == 0) & (es == 0)
    if not ok.any():
        return None
    d_km = np.linalg.norm(rp - rs, axis=1)
    d_km[~ok] = np.inf
    i = int(np.argmin(d_km))
    tca = times[i]
    miss_km = float(d_km[i])
    if 0 < i < n - 1 and np.isfinite(d_km[i - 1]) and np.isfinite(d_km[i + 1]):
        # Quadratic through d^2 at (i-1, i, i+1); i is the argmin so the
        # vertex offset is within +/-0.5 steps of i.
        y0 = float(d_km[i - 1]) ** 2
        y1 = miss_km ** 2
        y2 = float(d_km[i + 1]) ** 2
        denom = y0 - 2.0 * y1 + y2
        if denom > 0.0:
            x = 0.5 * (y0 - y2) / denom          # vertex, in units of steps
            x = max(-1.0, min(1.0, x))
            m2 = y1 - (y0 - y2) ** 2 / (8.0 * denom)
            miss_km = max(m2, 0.0) ** 0.5
            tca = times[i] + timedelta(seconds=REFINE_STEP_S * x)
    rel_v_kms = float(np.linalg.norm(vp[i] - vs[i]))
    return tca, miss_km * 1000.0, rel_v_kms * 1000.0


def screen(conn, primary_norad_ids: set[int], window_hours=None, threshold_km=None,
           step_s=None, min_rel_vel_ms=None, extra=None, *, start=None,
           catalog=None) -> list[ConjunctionEvent]:
    """Screen primaries against the whole LEO catalog over the look-ahead window.

    primary_norad_ids: norad ids fetched from config.PRIMARY_GROUPS this run
    (pipeline passes them through - membership is not stored in the DB).
    Keeps the deepest approach per (primary, secondary) pair; when both objects
    are primaries only the norad-ascending ordering is kept. Pairs whose refined
    encounter velocity is under min_rel_vel_ms (default
    config.SCREENING_MIN_REL_VEL_MS) are co-moving - docked station modules,
    formation flyers - and are excluded, not conjunctions.

    extra: optional [{norad_id, line1, line2}] injected into the propagation
    set after the catalog TLEs load - used by the user-fleet on-demand screen,
    which injects a user object under a synthetic norad (900000+id) and passes
    that norad as the sole primary. Injected objects exist only for this call
    (never written to the DB), replace a same-norad catalog entry if one
    exists, and bypass the LEO apogee skip (they are explicitly requested
    screening subjects, not bulk catalog secondaries). Unparseable extra TLEs
    are skipped like unparseable catalog rows.

    start / catalog (Time Machine replays): `start` pins the window start
    (default: now, floored to the second) and `catalog` replaces the DB
    catalog with in-memory [{norad_id, line1, line2}] rows (conn is then
    unused and may be None). Everything else - gates, refinement, filters -
    is the identical live code path.
    """
    window_hours = window_hours if window_hours is not None else config.SCREENING_WINDOW_HOURS
    threshold_km = threshold_km if threshold_km is not None else config.SCREENING_THRESHOLD_KM
    step_s = step_s if step_s is not None else config.SCREENING_COARSE_STEP_S
    min_rel_vel_ms = (min_rel_vel_ms if min_rel_vel_ms is not None
                      else config.SCREENING_MIN_REL_VEL_MS)

    sats: list[Satrec] = []
    norads: list[int] = []
    for row in (catalog if catalog is not None else _latest_tles(conn)):
        try:
            sat = Satrec.twoline2rv(row["line1"], row["line2"])
        except Exception:
            continue
        if sat.no_kozai <= 0 or _apogee_km(sat) > config.SCREENING_MAX_LEO_ALT_KM:
            continue
        sats.append(sat)
        norads.append(row["norad_id"])

    if extra:
        index_by_norad = {norad: i for i, norad in enumerate(norads)}
        for rec in extra:
            try:
                sat = Satrec.twoline2rv(rec["line1"], rec["line2"])
            except Exception:
                continue
            if sat.no_kozai <= 0:
                continue
            norad = int(rec["norad_id"])
            at = index_by_norad.get(norad)
            if at is None:
                index_by_norad[norad] = len(norads)
                sats.append(sat)
                norads.append(norad)
            else:
                sats[at] = sat

    nsat = len(sats)
    primary_set = set(primary_norad_ids)
    p_idx = np.array([i for i, nid in enumerate(norads) if nid in primary_set], dtype=int)
    if nsat < 2 or p_idx.size == 0:
        return []
    p_norads = [norads[i] for i in p_idx]
    norads_arr = np.array(norads)
    sat_by_norad = dict(zip(norads, sats))

    start = (start if start is not None
             else datetime.now(timezone.utc)).replace(microsecond=0)
    end = start + timedelta(hours=window_hours)
    n_steps = int(round(window_hours * 3600.0 / step_s)) + 1
    times = [start + timedelta(seconds=step_s * i) for i in range(n_steps)]
    jd, fr = jday_arrays(times)

    arr = SatrecArray(sats)
    thr2 = threshold_km * threshold_km
    capture_km = threshold_km + COARSE_CAPTURE_MARGIN_KM
    cap2 = capture_km * capture_km
    dropped = np.zeros(nsat, dtype=bool)
    # (p_norad, s_norad) -> [min_estimated_dist2_km2, global_time_index]
    best: dict[tuple[int, int], list] = {}

    def note(a: int, b: int, metric2: float, ti: int) -> None:
        pn = p_norads[a]
        sn = int(norads_arr[b])
        if sn in primary_set and pn > sn:
            return  # primary-primary pair: keep norad-ascending only
        key = (pn, sn)
        cur = best.get(key)
        if cur is None:
            best[key] = [metric2, ti]
        elif metric2 < cur[0]:
            cur[0] = metric2
            cur[1] = ti

    prev = None  # (d2, rv, vv) at the previous global step
    for c0 in range(0, n_steps, TIME_CHUNK_STEPS):
        e, r, v = arr.sgp4(jd[c0:c0 + TIME_CHUNK_STEPS], fr[c0:c0 + TIME_CHUNK_STEPS])
        bad = e != 0
        dropped |= bad.any(axis=1)   # mark objects that error and skip them
        r[bad] = np.nan
        r[dropped] = np.nan
        v[bad] = np.nan
        v[dropped] = np.nan
        for t in range(r.shape[1]):
            gi = c0 + t
            rt = r[:, t, :]                                   # (nsat, 3)
            vt = v[:, t, :]                                   # (nsat, 3) km/s
            norms2 = np.einsum("ij,ij->i", rt, rt)
            rv_dot = np.einsum("ij,ij->i", rt, vt)
            v2 = np.einsum("ij,ij->i", vt, vt)
            rp = rt[p_idx]                                    # (P, 3)
            vp = vt[p_idx]
            # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b  (BLAS-friendly); same identity
            # gives dr.dv and |dv|^2 for every (primary, secondary) pair.
            d2 = norms2[p_idx][:, None] + norms2[None, :] - 2.0 * (rp @ rt.T)
            rv = rv_dot[p_idx][:, None] + rv_dot[None, :] - rp @ vt.T - vp @ rt.T
            vv = v2[p_idx][:, None] + v2[None, :] - 2.0 * (vp @ vt.T)
            d2[np.arange(p_idx.size), p_idx] = np.inf         # exclude self-pairs

            # gate 1: sampled distance already under the reporting threshold
            for a, b in zip(*np.nonzero(d2 < thr2)):
                note(a, b, max(float(d2[a, b]), 0.0), gi)

            # Gate 2: rectilinear minimum over the segment [gi-1, gi] dips
            # under the capture radius. min_tau |dr + dv*tau|^2 from the
            # relative state at the segment start; catches fast crossings the
            # grid samples never see (endpoint minima are covered by gate 1).
            if prev is not None:
                pd2, prv, pvv = prev
                with np.errstate(invalid="ignore", divide="ignore"):
                    tau = -prv / pvv
                    dmin2 = pd2 - prv * prv / pvv
                interior = (tau > 0.0) & (tau < step_s) & (dmin2 < cap2)
                for a, b in zip(*np.nonzero(interior)):
                    ti = gi - 1 if tau[a, b] <= step_s / 2.0 else gi
                    note(a, b, max(float(dmin2[a, b]), 0.0), ti)
            prev = (d2, rv, vv)

    events: list[ConjunctionEvent] = []
    half_window_s = max(float(REFINE_HALF_WINDOW_S), float(step_s))
    for (pn, sn), (_m2, ti) in best.items():
        refined = _refine(sat_by_norad[pn], sat_by_norad[sn], times[ti], start, end,
                          half_window_s=half_window_s)
        if refined is None:
            continue
        tca, miss_m, rel_ms = refined
        if miss_m >= threshold_km * 1000.0:
            continue  # captured wider than the threshold; filter on refined miss
        if rel_ms < min_rel_vel_ms:
            continue  # co-moving pair (docked/formation), not a conjunction
        events.append(ConjunctionEvent(
            primary_norad=pn, secondary_norad=sn, tca=tca,
            miss_distance_m=miss_m, relative_velocity_ms=rel_ms,
        ))
    events.sort(key=lambda ev: ev.miss_distance_m)
    return events


def _compute_pc(miss_distance_m: float, relative_velocity_ms: float,
                propagation_age_days: float):
    """Lazy bridge to models.pc.compute_pc; None if it is missing."""
    try:
        from backend.app.models import pc as pc_mod
    except Exception:
        return None
    try:
        return pc_mod.compute_pc(miss_distance_m, relative_velocity_ms,
                                 propagation_age_days=propagation_age_days)
    except Exception:
        return None


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


def record_events(conn, events, now_iso: str) -> dict:
    """Upsert conjunctions and append one screening CDM per event.

    Dedupe: the UNIQUE (primary, secondary, tca) key handles exact repeats;
    additionally, an existing active conjunction for the same pair whose TCA is
    within 30 minutes is treated as the same event and has its TCA updated.
    """
    conjunctions_new = 0
    cdms_added = 0
    now_dt = _parse_iso(now_iso)
    for ev in events:
        prow = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                            (ev.primary_norad,)).fetchone()
        srow = conn.execute("SELECT id FROM satellites WHERE norad_id = ?",
                            (ev.secondary_norad,)).fetchone()
        if prow is None or srow is None:
            continue
        p_id, s_id = prow["id"], srow["id"]
        tca = ev.tca if ev.tca.tzinfo else ev.tca.replace(tzinfo=timezone.utc)
        tca_iso = tca.strftime(ISO_FMT)

        conj_id = None
        # Either object order: Space-Track-minted events store (SAT_1, SAT_2),
        # which may be the reverse of our (primary, secondary).
        for row in conn.execute(
            "SELECT id, primary_object_id, secondary_object_id,"
            " time_of_closest_approach FROM conjunctions"
            " WHERE ((primary_object_id = ? AND secondary_object_id = ?)"
            "     OR (primary_object_id = ? AND secondary_object_id = ?))"
            " AND status = 'active'",
            (p_id, s_id, s_id, p_id),
        ).fetchall():
            if abs((_parse_iso(row["time_of_closest_approach"]) - tca).total_seconds()) \
                    < TCA_MATCH_TOLERANCE_S:
                conj_id = row["id"]
                if row["time_of_closest_approach"] != tca_iso:
                    # Skip the update when another row already holds this exact
                    # TCA (pre-checked rather than caught: an IntegrityError
                    # would abort the whole transaction on Postgres).
                    dup = conn.execute(
                        "SELECT 1 FROM conjunctions WHERE primary_object_id = ?"
                        " AND secondary_object_id = ?"
                        " AND time_of_closest_approach = ?",
                        (row["primary_object_id"], row["secondary_object_id"],
                         tca_iso)).fetchone()
                    if dup is None:
                        conn.execute(
                            "UPDATE conjunctions SET time_of_closest_approach = ? WHERE id = ?",
                            (tca_iso, conj_id))
                break
        if conj_id is None:
            # INSERT OR IGNORE is dialect-portable (db.py translates it to
            # ON CONFLICT DO NOTHING on Postgres); rowcount==0 means the
            # UNIQUE (primary, secondary, tca) row already existed.
            cur = conn.execute(
                "INSERT OR IGNORE INTO conjunctions (primary_object_id,"
                " secondary_object_id, time_of_closest_approach, first_seen_at,"
                " status) VALUES (?, ?, ?, ?, 'active')",
                (p_id, s_id, tca_iso, now_iso))
            if cur.rowcount > 0:
                conj_id = cur.lastrowid
                conjunctions_new += 1
            else:
                conj_id = conn.execute(
                    "SELECT id FROM conjunctions WHERE primary_object_id = ?"
                    " AND secondary_object_id = ? AND time_of_closest_approach = ?",
                    (p_id, s_id, tca_iso)).fetchone()["id"]

        age_days = max((tca - now_dt).total_seconds() / 3600.0, 0.0) / 24.0
        pc = _compute_pc(ev.miss_distance_m, ev.relative_velocity_ms, age_days)
        conn.execute(
            "INSERT INTO cdm_messages (conjunction_id, message_time, miss_distance_m,"
            " relative_velocity_ms, probability_of_collision, source)"
            " VALUES (?, ?, ?, ?, ?, 'screening')",
            (conj_id, now_iso, ev.miss_distance_m, ev.relative_velocity_ms, pc))
        cdms_added += 1
    conn.commit()
    return {"conjunctions_new": conjunctions_new, "cdms_added": cdms_added}
