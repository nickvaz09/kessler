"""Feature engineering for the risk model.

Turns a CDM sequence for one conjunction into the fixed feature vector defined
by FEATURE_NAMES (the contract in docs/CONTRACTS.md). Pure and deterministic;
robust to sequences of length 1 (deltas/slope -> 0.0), to null
``probability_of_collision`` entries, and to empty sequences (all zeros).
"""
from datetime import datetime, timezone

import numpy as np

FEATURE_NAMES = ["miss_distance_m", "relative_velocity_ms", "pc_latest",
                 "time_to_tca_h", "cdm_count", "md_delta", "pc_log_delta",
                 "md_slope",
                 # vigil-0.3 additions - selected by held-out ablation:
                 # risk-trend rate, peak observed risk, message cadence
                 "pc_log_slope", "pc_log_max", "msg_gap_last_h"]

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 UTC 'Z' timestamp (fractional seconds tolerated)."""
    try:
        return datetime.strptime(value, _ISO_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_features(cdms: list[dict], tca_iso: str,
                   as_of_iso: str | None = None) -> dict:
    """Build the FEATURE_NAMES feature dict from a CDM sequence.

    ``cdms``: dicts with keys message_time, miss_distance_m,
    relative_velocity_ms, probability_of_collision (nullable), ascending by
    message_time (sorted defensively here). ``as_of_iso`` defaults to the
    latest CDM's message_time; time_to_tca_h = (tca - as_of) in hours.

    - miss_distance_m / relative_velocity_ms: from the latest CDM.
    - pc_latest: latest non-null probability_of_collision, else 0.0.
    - md_delta: latest minus first miss distance (0.0 when len < 2).
    - pc_log_delta: log10(latest positive pc) - log10(first positive pc)
      across the sequence; 0.0 with fewer than two positive values.
    - md_slope: per-day least-squares slope of miss distance vs time
      (0.0 when len < 2 or the time span is zero).
    - pc_log_slope (vigil-0.3): per-day least-squares slope of log10 pc
      over positive-pc messages (0.0 with < 2 positives or zero span).
    - pc_log_max (vigil-0.3): max log10 positive pc seen (0.0 when none -
      distinguishable from real values, which are far below 0 in log10).
    - msg_gap_last_h (vigil-0.3): hours between the last two messages
      (0.0 when len < 2).
    """
    feats = {name: 0.0 for name in FEATURE_NAMES}
    cdms = sorted(cdms, key=lambda c: c["message_time"])
    n = len(cdms)
    feats["cdm_count"] = float(n)

    tca = _parse_iso(tca_iso)
    if n == 0:
        if as_of_iso is not None:
            feats["time_to_tca_h"] = (tca - _parse_iso(as_of_iso)).total_seconds() / 3600.0
        return feats

    latest = cdms[-1]
    feats["miss_distance_m"] = float(latest["miss_distance_m"])
    feats["relative_velocity_ms"] = float(latest["relative_velocity_ms"])

    as_of = _parse_iso(as_of_iso) if as_of_iso is not None else _parse_iso(latest["message_time"])
    feats["time_to_tca_h"] = (tca - as_of).total_seconds() / 3600.0

    pcs = [c.get("probability_of_collision") for c in cdms]
    non_null = [float(p) for p in pcs if p is not None]
    if non_null:
        feats["pc_latest"] = max(non_null[-1], 0.0)
    positive = [float(p) for p in pcs if p is not None and p > 0.0]
    if len(positive) >= 2:
        feats["pc_log_delta"] = float(np.log10(positive[-1]) - np.log10(positive[0]))

    t0 = _parse_iso(cdms[0]["message_time"])
    days = np.array([(_parse_iso(c["message_time"]) - t0).total_seconds() / 86400.0
                     for c in cdms], dtype=float)

    if n >= 2:
        feats["md_delta"] = float(latest["miss_distance_m"]) - float(cdms[0]["miss_distance_m"])
        mds = np.array([float(c["miss_distance_m"]) for c in cdms], dtype=float)
        if float(days.max() - days.min()) > 0.0:
            feats["md_slope"] = float(np.polyfit(days, mds, 1)[0])
        feats["msg_gap_last_h"] = float(days[-1] - days[-2]) * 24.0

    # vigil-0.3: log10-pc trajectory over positive-pc messages
    pos = [(float(d), float(np.log10(float(p))))
           for d, p in zip(days, pcs) if p is not None and float(p) > 0.0]
    if pos:
        feats["pc_log_max"] = max(v for _, v in pos)
    if len(pos) >= 2:
        pdays = np.array([p[0] for p in pos], dtype=float)
        pvals = np.array([p[1] for p in pos], dtype=float)
        if float(pdays.max() - pdays.min()) > 0.0:
            feats["pc_log_slope"] = float(np.polyfit(pdays, pvals, 1)[0])

    return feats
