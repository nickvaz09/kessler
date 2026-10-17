"""Competition mode: a leaderboard entry under the 2019 Kelvins rules.

The 2019 leaderboard was built with test-set feedback: teams submitted
predictions for the official test events and got their score back, up to
two submissions a day for about two months (Uriot et al. 2022, §4.2). This
module reproduces those conditions for VIGIL, so its leaderboard number is
comparable to the teams' - and keeps the claim honest:

- The official test set is split once (fixed seed, stratified on the label)
  into a PUBLIC and a PRIVATE half. ``--make-split`` writes the split before
  anything is scored, so it can be committed first.
- Models train on the training file only. Test labels never enter training.
- Development may see only the AGGREGATE score on the public half, logged
  per submission in ``ledger.jsonl``, capped at ``BUDGET`` submissions. No
  per-event test result is ever inspected.
- The chosen entry is scored once on the untouched private half and on the
  full test. Both numbers are reported; only the full-test score is
  comparable to the leaderboard, as the teams' were.

This is a benchmark entry, not evidence about live performance: like the
2019 entries it is tuned to the official test's distribution.

    .venv/bin/python -m backend.app.models.lab.compete --make-split \\
        --train data/kelvins/train_data.csv --test <test_data.csv> \\
        --test-targets <final-CDM file> --out reports/compete
    .venv/bin/python -m backend.app.models.lab.compete --run ... --out reports/compete
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.models.lab import candidates as C
from backend.app.models.lab import data, metric, protocol
from backend.app.models.lab.data import HIGH_RISK, LOW_PREDICTION

# kelvins.esa.int/collision-avoidance-challenge/results - final scores of the
# twelve teams at or below the latest-risk baseline (97 ranked teams)
LEADERBOARD = [("sesc", 0.555310), ("dietmarw", 0.574456), ("Magpies", 0.584892),
               ("Vidente", 0.609829), ("DeCRA", 0.614875), ("Valis", 0.627369),
               ("DunderMifflin", 0.627583), ("Spacemeister", 0.651616),
               ("GrandpaBem", 0.651736), ("TS", 0.672828), ("AndrewLouw", 0.689373),
               ("ANTwork", 0.693964)]
N_TEAMS = 97
SPLIT_SEED = 2019
BUDGET = 60          # public submissions; the 2019 rules allowed two a day for ~2 months

# the decision family: every combination below is one possible entry; the
# latest-risk baseline is its member with no catches, no overrides, alpha 0
GRID = {
    "w": (1.0, 0.5, 0.0),                      # benchmark vs serving classifier (logit blend)
    "tau": tuple(np.round(np.geomspace(0.002, 0.3, 10), 5)) + (1.01,),   # 1.01 = never
    "band": (None, -6.4, -7.0, -7.3),          # also catch every event at or above this risk
    "c": (None, -5.6, -5.3, -5.0),             # catch value (None = the model's m(x))
    "alpha": (0.0, 0.5, 1.0),                  # base-high value: alpha*m + (1-alpha)*latest
    "cap": (None, -4.0, -3.5, -3.0),           # clip high values (risk tends to fall)
    "tau_drop": (0.0, 0.01, 0.03, 0.1),        # drop a base-high call below this probability
    "o_payload": (False, True),                # low if the chaser is a payload
    "o_tspan": (False, True),                  # low if the target is small (t_span < 0.5)
    "o_miss": (False, True),                   # low if miss distance > 30 km
}
LRP_CFG = {"w": 1.0, "tau": 1.01, "band": None, "c": None, "alpha": 0.0, "cap": None,
           "tau_drop": 0.0, "o_payload": False, "o_tspan": False, "o_miss": False}


def make_split(test_events, seed: int = SPLIT_SEED) -> np.ndarray:
    """Boolean mask of the PUBLIC half, stratified on the final label."""
    hr = np.array([e.high_risk for e in test_events])
    rng = np.random.RandomState(seed)
    pub = np.zeros(len(test_events), bool)
    for cls in (True, False):
        idx = np.flatnonzero(hr == cls)
        rng.shuffle(idx)
        pub[idx[: len(idx) // 2]] = True
    return pub


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def decide(cfg: dict, s: dict) -> np.ndarray:
    """Predicted final risk for every event in ``s`` (a side: arrays p_b,
    p_s, m_b, m_s, latest, payload, t_span, miss) under config ``cfg``."""
    w = cfg["w"]
    p = 1.0 / (1.0 + np.exp(-(w * _logit(s["p_b"]) + (1 - w) * _logit(s["p_s"]))))
    m = w * s["m_b"] + (1 - w) * s["m_s"]
    latest = s["latest"]
    base = latest >= HIGH_RISK
    v_hi = cfg["alpha"] * m + (1 - cfg["alpha"]) * latest
    if cfg["cap"] is not None:
        v_hi = np.minimum(v_hi, cfg["cap"])
    v_c = m if cfg["c"] is None else np.full(len(latest), float(cfg["c"]))
    catch = p >= cfg["tau"]
    if cfg["band"] is not None:
        catch |= latest >= cfg["band"]
    call = np.where(base, p >= cfg["tau_drop"], catch)
    pred = np.where(call, np.maximum(np.where(base, v_hi, v_c), HIGH_RISK), LOW_PREDICTION)
    over = np.zeros(len(latest), bool)
    if cfg["o_payload"]:
        over |= s["payload"]
    if cfg["o_tspan"]:
        over |= s["t_span"] < 0.5
    if cfg["o_miss"]:
        over |= s["miss"] > 30000
    return np.where(over, LOW_PREDICTION, pred)


def sesc_cascade(s: dict) -> np.ndarray:
    """Team sesc's published rule cascade (Uriot et al. 2022, Eq. 2, steps
    0-6) - a credited reference row, not a VIGIL entry."""
    r = s["latest"]
    pred = metric.lrp(r)
    pred = np.where((r >= -6.04) & (r < -6.00), -5.95, pred)       # step 0
    pred = np.where((r >= -6.40) & (r < -6.04), -5.60, pred)       # step 1
    pred = np.where((r >= -7.30) & (r < -6.40), -5.00, pred)       # step 2
    pred = np.where((r >= -4.00) & (r < -3.50), -4.00, pred)       # step 6
    pred = np.where(r >= -3.50, -3.50, pred)                       # step 6
    low = s["payload"] | (s["t_span"] < 0.5) | (s["miss"] > 30000)  # steps 3-5
    return np.where(low, -6.00001, pred)


def markdown(final: dict, ledger_rows: list) -> str:
    f, pv = final["full"], final["private"]
    L = ["# VIGIL competition entry (2019 Kelvins rules)\n",
         "Built the way the 2019 leaderboard was: models trained on the training file only, "
         "tuned with score-only feedback from a PUBLIC half of the official test "
         f"({final['public_submissions']} of {final['budget']} allowed submissions, every one in "
         "`ledger.jsonl`), then scored once on the untouched PRIVATE half and on the full test. "
         "A benchmark entry tuned to the official test's distribution, like the 2019 entries — "
         "not evidence about live performance.\n",
         "| | Public half | Private half (scored once) | Full test (leaderboard) |",
         "|---|---|---|---|",
         f"| VIGIL entry | {final['public_L']:.3f} | {pv['L']:.3f} | **{f['L']:.3f}** |",
         f"| Latest-risk baseline | — | {pv['lrp_L']:.3f} | {f['lrp_L']:.3f} |",
         f"| sesc's published cascade (reference) | — | {pv['sesc_cascade_L']:.3f} | "
         f"{f['sesc_cascade_L']:.3f} |", "",
         f"Full test: L {f['L']:.4f} (MSE_HR {f['mse_hr']:.3f}, F2 {f['f2']:.3f}), "
         f"{f['delta_L_pct_vs_lrp']:+.1f}% vs the baseline — **rank {final['placement_full']} of "
         f"{N_TEAMS}** on the final 2019 leaderboard (1st: sesc {LEADERBOARD[0][1]:.3f}).\n",
         f"Private half, never used for feedback: L {pv['L']:.4f} "
         f"({pv['delta_L_pct_vs_lrp']:+.1f}% vs the baseline on the same events).\n",
         f"Training file, out-of-fold, at the test's class mix: L "
         f"{final['train_oof_test_mix_L']:.4f}.\n",
         "Entry configuration:", "```", json.dumps(final["config"], indent=1, default=str), "```",
         f"Code: `{(final.get('code') or {}).get('commit')}`.\n"]
    return "\n".join(L) + "\n"


def fit_sides(train_events, test_events, seed: int = 42, log=print):
    """Fit the benchmark and serving hurdles once on the training file.
    Returns (train side: cross-fitted out-of-fold arrays on eligible training
    events, test side: predictions for every test event)."""
    tr = protocol.Bundle(train_events)
    te = protocol.Bundle(test_events, tr.columns)
    idx = np.flatnonzero(tr.trainable)
    rows = idx[tr.eligible[idx]]
    train_side = {"latest": tr.latest[rows], "y": tr.y[rows]}
    test_side = {"latest": te.latest, "y": te.y}
    for key, track in (("b", "benchmark"), ("s", "serving")):
        t0 = time.time()
        h = C.Hurdle(f"compete_{track}", mode="lrp", track=track)
        h.fit(protocol._ctx(tr, idx, track, seed))
        train_side[f"p_{key}"], train_side[f"m_{key}"] = h._tune["p"], h._tune["m"]
        test_side[f"p_{key}"] = h.prob_high(te.X[track], te.latest)
        test_side[f"m_{key}"] = h._cond(te.X[track], te.latest)
        log(f"  fitted {track} hurdle ({time.time() - t0:.0f}s)")
    for side, b, r in ((train_side, tr, rows), (test_side, te, np.arange(len(test_events)))):
        def col(name, default):
            j = b.colindex.get(name)
            return b.X["benchmark"][r, j] if j is not None else np.full(len(r), default)
        side["payload"] = col("l_ctype_payload", 0.0) > 0.5
        side["t_span"] = col("l_t_span", np.nan)
        side["miss"] = col("l_miss_distance", np.nan)
    return train_side, test_side


def _subset(s: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in s.items()}


def _configs():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def _key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


class Ledger:
    """Every look at the public half, in order, with only its aggregate score."""

    def __init__(self, path: Path, public: dict, budget: int = BUDGET):
        self.path, self.public, self.budget = path, public, budget
        self.rows, self.seen = [], {}
        path.write_text("")

    def submit(self, cfg: dict, stage: str, pred=None) -> float:
        k = _key(cfg) if pred is None else stage
        if k in self.seen:
            return self.seen[k]
        if len(self.rows) >= self.budget:
            raise RuntimeError("public submission budget exhausted")
        p = decide(cfg, self.public) if pred is None else pred
        s = metric.official(p, self.public["y"])
        row = {"n": len(self.rows) + 1, "stage": stage, "config": cfg if pred is None else None,
               "public_L": s["L"], "public_mse_hr": s["mse_hr"], "public_f2": s["f2"]}
        self.rows.append(row)
        self.seen[k] = s["L"]
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return s["L"]

    @property
    def left(self):
        return self.budget - len(self.rows)


def _neighbours(cfg):
    for k, vals in GRID.items():
        i = vals.index(cfg[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                yield {**cfg, k: vals[j]}


def search(train_side: dict, ledger: Ledger, log=print) -> dict:
    """Stage 1 ranks every config on the training file's out-of-fold
    predictions at the official test's class mix (no test data). Stage 2
    submits the best config per (classifier blend, override set) to the
    public half. Stage 3 spends the rest of the budget on one-step
    neighbours of the best public configs. Returns the best public config."""
    w = metric.label_shift_weights(train_side["y"] >= HIGH_RISK, C.BENCH_HR_PREVALENCE)
    t0 = time.time()
    best_per = {}
    n = 0
    for cfg in _configs():
        L = metric.official(decide(cfg, train_side), train_side["y"], weights=w)["L"]
        g = (cfg["w"], cfg["o_payload"], cfg["o_tspan"], cfg["o_miss"])
        if g not in best_per or L < best_per[g][0]:
            best_per[g] = (L, cfg)
        n += 1
    log(f"  stage 1: {n} configs ranked on training out-of-fold at the test mix "
        f"({time.time() - t0:.0f}s)")
    for L, cfg in sorted(best_per.values(), key=lambda t: t[0]):
        if ledger.left <= 0:
            break
        ledger.submit(cfg, "stage2")
    log(f"  stage 2: {len(ledger.rows)} public submissions")
    while ledger.left > 0:
        top = sorted((r for r in ledger.rows if r["config"]), key=lambda r: r["public_L"])[:3]
        fresh = [c for r in top for c in _neighbours(r["config"]) if _key(c) not in ledger.seen]
        if not fresh:
            break
        for c in fresh:
            if ledger.left <= 0:
                break
            ledger.submit(c, "stage3")
    log(f"  stage 3: {len(ledger.rows)} public submissions in total")
    best = min((r for r in ledger.rows if r["config"]), key=lambda r: r["public_L"])
    return best["config"]


def placement(L: float) -> int:
    """Rank this score would have taken on the final 2019 leaderboard."""
    return 1 + sum(1 for _, s in LEADERBOARD if s < L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VIGIL competition mode (2019 Kelvins rules)")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--test-targets", required=True)
    p.add_argument("--out", default="reports/compete")
    p.add_argument("--seed", type=int, default=42)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-split", action="store_true", help="write the public/private split only")
    g.add_argument("--run", action="store_true", help="fit, search on the public half, score once")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = data.load(a.train, a.test, a.test_targets)
    split_path = out / "split.json"
    if a.make_split:
        if split_path.exists():
            p.error(f"{split_path} exists; the split is made once")
        pub = make_split(test)
        split_path.write_text(json.dumps({
            "seed": SPLIT_SEED, "n_public": int(pub.sum()), "n_private": int((~pub).sum()),
            "public_event_ids": [e.event_id for e, b in zip(test, pub) if b]}, indent=1))
        print(f"split written: {int(pub.sum())} public / {int((~pub).sum())} private events")
        return 0
    if not split_path.exists():
        p.error("make (and commit) the split first: --make-split")
    if (out / "final.json").exists():
        p.error("this competition run already produced a final score; it is scored once")
    ids = set(json.loads(split_path.read_text())["public_event_ids"])
    pub = np.array([e.event_id in ids for e in test])
    t0 = time.time()
    train_side, test_side = fit_sides(train, test, a.seed)
    public, private = _subset(test_side, pub), _subset(test_side, ~pub)
    ledger = Ledger(out / "ledger.jsonl", public)
    ledger.submit(LRP_CFG, "reference: latest-risk baseline")
    ledger.submit(None, "reference: sesc published cascade", pred=sesc_cascade(public))
    best = search(train_side, ledger)
    # the one scoring of the untouched private half and of the full test
    final = {"config": best, "public_submissions": len(ledger.rows), "budget": BUDGET,
             "code": None, "runtime_s": None}
    for name, side in (("private", private), ("full", test_side)):
        s = metric.official(decide(best, side), side["y"])
        b = metric.official(metric.lrp(side["latest"]), side["y"])
        c = metric.official(sesc_cascade(side), side["y"])
        final[name] = {"L": s["L"], "mse_hr": s["mse_hr"], "f2": s["f2"],
                       "lrp_L": b["L"], "sesc_cascade_L": c["L"],
                       "delta_L_pct_vs_lrp": 100 * (s["L"] / b["L"] - 1)}
    w = metric.label_shift_weights(train_side["y"] >= HIGH_RISK, C.BENCH_HR_PREVALENCE)
    final["train_oof_test_mix_L"] = metric.official(decide(best, train_side), train_side["y"],
                                                    weights=w)["L"]
    final["public_L"] = ledger.seen[_key(best)]
    final["placement_full"] = placement(final["full"]["L"])
    from backend.app.models.lab.__main__ import _code_version
    final["code"], final["runtime_s"] = _code_version(), round(time.time() - t0, 1)
    (out / "final.json").write_text(json.dumps(final, indent=2, default=str))
    (out / "report.md").write_text(markdown(final, ledger.rows))
    print(json.dumps(final, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
