"""Kelvins data loading and the official challenge protocol.

The Collision Avoidance Challenge (ESA Kelvins, 2019; Uriot et al.,
Astrodynamics 2022) defines each event's task as: from the CDMs received at
least ``INPUT_CUTOFF_DAYS`` (2) days before TCA, predict the ``risk``
(log10 Pc) of the event's FINAL CDM. The official test set only contains
events that satisfy all of:

    - at least 2 CDMs in the event;
    - the last CDM within ``TARGET_MAX_DAYS`` (1) day of TCA;
    - the first CDM at least ``INPUT_CUTOFF_DAYS`` (2) days before TCA.

Training events were not filtered, so a validation set built from raw
training events does not resemble the test set. ``test_eligible`` applies
the official filters so validation can mimic the test distribution.

Everything downstream consumes ``Event`` objects: ``inputs`` holds only the
CDMs a model may see (time_to_tca >= cutoff), ``target`` the final risk.
Late CDMs are kept separately (``late``) solely for the ceiling diagnostic
and are never visible to a benchmark-track model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INPUT_CUTOFF_DAYS = 2.0
TARGET_MAX_DAYS = 1.0
HIGH_RISK = -6.0            # official high-risk threshold on log10 Pc
RISK_FLOOR = -30.0          # Kelvins sentinel for "Pc effectively zero"
LOW_PREDICTION = -6.001     # official convention for "not high risk"


@dataclass
class Event:
    event_id: str
    inputs: list[dict]              # CDM rows with time_to_tca >= cutoff, ascending time
    target: float                   # final CDM risk (log10 Pc)
    final_tta: float                # time_to_tca of the final CDM [days]
    n_cdm_total: int
    first_tta: float
    source: str = "train"           # 'train' | 'test'
    late: list[dict] = field(default_factory=list)  # CDMs inside the cutoff (diagnostics only)

    @property
    def latest_risk(self) -> float:
        return float(self.inputs[-1]["risk"]) if self.inputs else RISK_FLOOR

    @property
    def test_eligible(self) -> bool:
        return (self.n_cdm_total >= 2
                and self.final_tta <= TARGET_MAX_DAYS
                and self.first_tta >= INPUT_CUTOFF_DAYS
                and len(self.inputs) >= 1)

    @property
    def high_risk(self) -> bool:
        return self.target >= HIGH_RISK

    @property
    def escalation_candidate(self) -> bool:
        """VIGIL-E domain: still below the high-risk line at the cutoff."""
        return self.latest_risk < HIGH_RISK

    @property
    def escalates(self) -> bool:
        return self.escalation_candidate and self.high_risk


def _records(df) -> list[dict]:
    return df.to_dict("records")


def events_from_frame(df, source: str = "train",
                      cutoff_days: float = INPUT_CUTOFF_DAYS) -> list[Event]:
    """Group a Kelvins CDM frame into Events (target = the final CDM).

    Rows are ordered by descending time_to_tca (ascending message time).
    Events without any CDM at or before the cutoff have no admissible input
    and are dropped (a model could not have been asked about them).
    """
    required = {"event_id", "time_to_tca", "risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kelvins frame lacks columns: {sorted(missing)}")
    events: list[Event] = []
    for eid, group in df.groupby("event_id", sort=True):
        g = group.sort_values("time_to_tca", ascending=False, kind="mergesort")
        rows = _records(g)
        final = rows[-1]
        inputs = [r for r in rows if float(r["time_to_tca"]) >= cutoff_days]
        late = [r for r in rows if float(r["time_to_tca"]) < cutoff_days]
        if not inputs:
            continue
        events.append(Event(
            event_id=f"{source}:{eid}", inputs=inputs,
            target=float(final["risk"]), final_tta=float(final["time_to_tca"]),
            n_cdm_total=len(rows), first_tta=float(rows[0]["time_to_tca"]),
            source=source, late=late))
    return events


def attach_test_targets(test_df, targets_df):
    """Complete the official test inputs with the separately published final
    CDMs. ``targets_df`` may hold full CDM rows (appended as the final CDM of
    each event) or just ``event_id`` + ``risk`` (+ optional ``time_to_tca``;
    absent -> 0.0, i.e. "at TCA", which satisfies the <=1-day filter the test
    set was built with)."""
    import pandas as pd

    t = targets_df.copy()
    if "risk" not in t.columns:
        # some releases name the target column differently
        for alt in ("true_risk", "final_risk", "target"):
            if alt in t.columns:
                t = t.rename(columns={alt: "risk"})
                break
    if "event_id" not in t.columns or "risk" not in t.columns:
        raise ValueError("test targets need event_id and risk columns")
    if "time_to_tca" not in t.columns:
        t["time_to_tca"] = 0.0
    # keep one COMPLETE final row per event: the one closest to TCA
    # (groupby().first() would mix first-non-null values across rows)
    t = t.sort_values("time_to_tca", kind="mergesort").drop_duplicates("event_id", keep="first")
    # event ids must match in type, or events silently lose their target
    t["event_id"] = t["event_id"].astype(test_df["event_id"].dtype)
    missing = set(test_df["event_id"].unique()) - set(t["event_id"].unique())
    if missing:
        raise ValueError(f"{len(missing)} test events have no final CDM in the targets file "
                         f"(e.g. {sorted(missing)[:5]}) — check the file and event_id types")
    extra = [c for c in t.columns if c not in test_df.columns]
    t = t.drop(columns=extra)
    return pd.concat([test_df, t], ignore_index=True, sort=False)


def load(train_csv: str | Path, test_csv: str | Path | None = None,
         test_targets: str | Path | None = None) -> tuple[list[Event], list[Event]]:
    """(train_events, test_events). test_events is empty unless both the test
    inputs and their final CDMs are available."""
    import pandas as pd

    train = events_from_frame(pd.read_csv(train_csv), "train")
    test: list[Event] = []
    if test_csv:
        tdf = pd.read_csv(test_csv)
        if test_targets:
            tdf = attach_test_targets(tdf, pd.read_csv(test_targets))
            test = events_from_frame(tdf, "test")
        else:
            # without targets the test inputs cannot be scored
            test = []
    return train, test


def eligible_mask(events: list[Event]) -> np.ndarray:
    return np.array([e.test_eligible for e in events], dtype=bool)
