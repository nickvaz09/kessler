"""Train the XGBoost risk model on the ESA Kelvins collision-avoidance dataset.

CLI (run from the repo root):
    .venv/bin/python -m backend.app.models.train --data data/kelvins/train_data.csv

The Kelvins collision-avoidance-challenge ``train_data.csv`` contains CDM
sequences grouped by ``event_id`` with (among others) columns ``time_to_tca``
[days], ``miss_distance`` [m], ``relative_speed`` [m/s] and ``risk`` - the
log10 collision probability of that CDM; the target of the challenge is the
``risk`` of each event's *final* CDM.

Pipeline:
1. Group rows by event_id, sort by ascending time (descending time_to_tca).
2. Per event (>= 2 CDMs): features from all but the final CDM via
   ``features.build_features`` (message times synthesized from time_to_tca
   against a fixed reference TCA); target = the final CDM's ``risk``.
3. Event-level 80/20 split (one feature row per event, seeded shuffle).
4. XGBoost regressor on log10 risk. Report overall held-out MSE and the
   competition-style MSE restricted to held-out events with final
   risk >= -6, alongside the naive last-risk carry-forward baseline computed
   on the same split.
5. Save the booster to config.RISK_MODEL_FILE and meta json (version,
   trained_at, metrics) to config.RISK_MODEL_META_FILE.

Exits 0 with download instructions when the CSV is absent.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from backend.app import config
from backend.app.models import risk_model
from backend.app.models.features import FEATURE_NAMES, build_features

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
# Fixed synthetic TCA reference: Kelvins gives relative times (time_to_tca in
# days), so absolute message times are reconstructed against this epoch.
_TCA_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_TCA_ISO = _TCA_EPOCH.strftime(_ISO_FMT)

REQUIRED_COLUMNS = ["event_id", "time_to_tca", "miss_distance",
                    "relative_speed", "risk"]
HIGH_RISK_THRESHOLD = -6.0  # competition metric: events with final risk >= -6

MISSING_CSV_MESSAGE = """\
Kelvins training data not found at: {path}

To train the model, download the ESA Kelvins Collision Avoidance Challenge
dataset from its public Zenodo archive (no account needed):
    https://zenodo.org/record/4463683
unzip it so train_data.csv sits at data/kelvins/train_data.csv (or pass
--data <path>), then re-run:
    .venv/bin/python -m backend.app.models.train --data data/kelvins/train_data.csv

Until then the API serves the heuristic model ({heuristic}).\
"""


def build_dataset(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event (X, y, baseline) arrays from the Kelvins dataframe.

    One row per event with >= 2 CDMs: X = FEATURE_NAMES vector built from all
    but the final CDM, y = final CDM's risk (log10 Pc), baseline = last known
    (non-final) risk carried forward.
    """
    X, y, baseline = [], [], []
    for _, group in df.groupby("event_id"):
        group = group.sort_values("time_to_tca", ascending=False)  # ascending time
        if len(group) < 2:
            continue
        rows = group.to_dict("records")
        final, prior = rows[-1], rows[:-1]
        cdms = []
        for r in prior:
            message_time = _TCA_EPOCH - timedelta(days=float(r["time_to_tca"]))
            cdms.append({
                "message_time": message_time.strftime(_ISO_FMT),
                "miss_distance_m": float(r["miss_distance"]),
                "relative_velocity_ms": float(r["relative_speed"]),
                "probability_of_collision": 10.0 ** float(r["risk"]),
            })
        feats = build_features(cdms, _TCA_ISO)
        X.append([feats[name] for name in FEATURE_NAMES])
        y.append(float(final["risk"]))
        baseline.append(float(prior[-1]["risk"]))
    return (np.array(X, dtype=float), np.array(y, dtype=float),
            np.array(baseline, dtype=float))


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        return float("nan")
    return float(np.mean((pred - truth) ** 2))


def _fmt(value: float) -> str:
    return "n/a (no such events in split)" if np.isnan(value) else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Kessler risk model on the ESA Kelvins dataset.")
    parser.add_argument("--data", default=str(config.KELVINS_DIR / "train_data.csv"),
                        help="path to the Kelvins train_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high-weight", type=float, default=0.0,
                        help="training weight of events already high-risk at "
                             "prediction time; 0 excludes them — the hybrid "
                             "never applies corrections there, so train on "
                             "the domain where the model actually acts")
    args = parser.parse_args(argv)

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(MISSING_CSV_MESSAGE.format(
            path=csv_path, heuristic=config.HEURISTIC_MODEL_VERSION))
        return 0

    import pandas as pd
    import xgboost as xgb

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"error: {csv_path} lacks required columns: {missing}")
        return 1
    df = df.dropna(subset=REQUIRED_COLUMNS)

    print(f"loading {csv_path} — {len(df)} CDM rows, "
          f"{df['event_id'].nunique()} events")
    X, y, baseline = build_dataset(df)
    n_events = len(y)
    if n_events < 10:
        print(f"error: only {n_events} usable events (>= 2 CDMs each); "
              "not enough to train")
        return 1

    # Event-level 80/20 split (one row per event => row split is event split).
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(n_events)
    n_train = int(round(n_events * 0.8))
    train_idx, test_idx = order[:n_train], order[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    base_test = baseline[test_idx]
    print(f"events: {n_events} usable ({len(train_idx)} train / "
          f"{len(test_idx)} held out)")

    # vigil-0.4 configuration - selected by random search on three validation
    # carves inside the train split (never the held-out test; a config had to
    # win on the carve MEAN), then judged once here. The 0.3 feature set is
    # unchanged: eleven further candidates were tried and none survived the
    # greedy ablation - the gain is configuration, not features.
    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=7, learning_rate=0.015,
        min_child_weight=3, subsample=1.0, colsample_bytree=0.6,
        reg_lambda=5.0, reg_alpha=0.0, gamma=0.1,
        objective="reg:squarederror", random_state=args.seed, n_jobs=4,
    )
    # Hybrid residual target. For events already at/above the high-risk
    # threshold, carry-forward proved near-unbeatable in the published
    # challenge (conditioning the metric on the final value punishes any
    # learned decay), so the model defers to it there. Below the threshold - 
    # where risk decay dominates and carry-forward is weakest - the model
    # predicts the log10 correction to the last-known risk, and it is trained
    # only on that domain (high-last events weighted --high-weight, default 0
    # = excluded).
    base_train = baseline[train_idx]
    resid_train = y_train - base_train
    weights = np.where(base_train >= HIGH_RISK_THRESHOLD, args.high_weight, 1.0)
    model.fit(X_train, resid_train, sample_weight=weights)
    resid_pred = np.clip(model.predict(X_test),
                         -risk_model.RESID_CLIP_DECADES,
                         risk_model.RESID_CLIP_DECADES)
    resid_applied = np.where(base_test < HIGH_RISK_THRESHOLD, resid_pred, 0.0)
    pred_test = base_test + resid_applied

    high = y_test >= HIGH_RISK_THRESHOLD
    metrics = {
        "mse_test_all": _mse(pred_test, y_test),
        "mse_test_high_risk": _mse(pred_test[high], y_test[high]),
        "baseline_mse_all": _mse(base_test, y_test),
        "baseline_mse_high_risk": _mse(base_test[high], y_test[high]),
        "n_events": int(n_events),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_high_risk": int(high.sum()),
    }

    print()
    print("held-out MSE on log10 risk:")
    print(f"  xgboost   (all events):            {_fmt(metrics['mse_test_all'])}")
    print(f"  xgboost   (final risk >= -6):      {_fmt(metrics['mse_test_high_risk'])}")
    print(f"  last-risk carry-forward (all):     {_fmt(metrics['baseline_mse_all'])}")
    print(f"  last-risk carry-forward (>= -6):   {_fmt(metrics['baseline_mse_high_risk'])}")
    print()
    print("comparison vs published Kelvins results: the winning challenge")
    print("solutions clustered around ~0.5 on the official competition metric,")
    print("and the naive last-risk carry-forward above proved a strong baseline")
    print("that most entries struggled to beat — compare the two high-risk MSE")
    print("lines above to judge whether the learned model adds value.")

    # Ship gate (fixed before the hybrid was evaluated): high-risk MSE within
    # 2% of carry-forward (the hybrid defers there; the tolerance covers
    # low-last events that rise across the threshold) and overall MSE at
    # least 10% better (the model must add real value where it acts).
    # Otherwise remove any artifact so the API keeps serving the heuristic.
    ship = (metrics["mse_test_high_risk"] <= metrics["baseline_mse_high_risk"] * 1.02
            and metrics["mse_test_all"] <= metrics["baseline_mse_all"] * 0.9)
    print()
    if not ship:
        for stale in (config.RISK_MODEL_FILE, config.RISK_MODEL_META_FILE):
            Path(stale).unlink(missing_ok=True)
        print("NOT SHIPPED: model fails the ship gate against the last-risk "
              "carry-forward baseline (high-risk MSE <= 1.02x baseline AND "
              "overall MSE <= 0.9x baseline). Any previous artifact was "
              "removed; the API "
              f"serves {config.HEURISTIC_MODEL_VERSION}.")
        return 0

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(config.RISK_MODEL_FILE))
    meta = {
        "version": "vigil-0.4",
        "target": "residual_below_threshold",
        "trained_at": datetime.now(timezone.utc).strftime(_ISO_FMT),
        "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in metrics.items()},
    }
    config.RISK_MODEL_META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"saved model -> {config.RISK_MODEL_FILE}")
    print(f"saved meta  -> {config.RISK_MODEL_META_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
