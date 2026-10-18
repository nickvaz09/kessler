# VIGIL Lab — how Kessler finds its best possible risk models

`backend/app/models/lab/` is a self-contained research harness. It is not
imported by the API and never overwrites the shipped models. It answers one
question as rigorously as public data allows: **which model, if any, is
genuinely better than the simple "latest risk" rule, and by how much?**

## Why a lab was needed

The research review (reports/Advanced ML for conjunction risk.md) found that
VIGIL-0.4's numbers could not be compared with the 2019 ESA Kelvins
leaderboard:

1. **Inputs.** VIGIL-0.4 predicted from every CDM except the last, including
   messages inside the final two days. The official task only allows CDMs
   received at least 2 days before closest approach (TCA). That is a
   different, easier problem.
2. **Scoring.** It reported plain MSE. The official score is
   `L = MSE_HR / F2`: regression error on events that END high-risk (final
   log10 Pc >= -6), divided by the F2 score of the high-risk call. It also
   omitted the organisers' convention of predicting -6.001 for "not high
   risk".
3. **Test events.** The official test set only contains events with at
   least 2 CDMs, a final CDM within 1 day of TCA and a first CDM at least 2
   days out. VIGIL-0.4 validated on unfiltered training events.
4. **Statistics.** It used one random 80/20 split. The spread across seeds
   (13.77–14.93) was as large as the claimed gain.

For VIGIL-E, a 20% split held only about 37 escalations (under VIGIL-E's
old all-event definition). At that size only a PR-AUC gap of roughly
0.05–0.08 is detectable, so the earlier gate could not tell a real 0.01–0.02
improvement from noise. Under the official protocol the real counts are
smaller still: 19 eligible escalations in the training file, 35 in the
official test set, 54 pooled.

## What the lab does

**Protocol (`data.py`, `metric.py`)**
- Inputs are only CDMs with time_to_tca >= 2 days, and the target is the
  final CDM's risk.
- Validation and test scores cover only events that pass the official test
  filters.
- The score is the official L, with the organisers' latest-risk baseline
  (LRP) and the -6.001 convention.
- The official test set, completed with its separately published final
  CDMs, is **sealed**. It is scored once, after the candidate pool and the
  gate are fixed. It judges the pre-registered primaries (see "Pre-run
  amendment"), but it never selects a model.

**Two feature tracks (`features.py`)**
- *Benchmark*: the latest admissible CDM's columns, plus the sequence
  history, plus encounter-plane physics (below). Used for the leaderboard
  comparison.
- *Serving*: exactly the production feature vectors (VIGIL's
  `FEATURE_NAMES`, VIGIL-E's `ESCALATION_FEATURE_NAMES`) built by the
  production functions from what the live app stores: risk, miss distance,
  speed and timing. **The app may only claim serving-track numbers**,
  because the app never sees the Kelvins covariance columns.

**Physics features (`physics.py`)**
- *Dilution index* `log(m²/2)`: whether a low Pc is reassuring or merely
  diluted by a large covariance (Alfano / NASA CARA).
- *Pc-max gap*: how far Pc could rise if the covariance shrank.
- *Shrink-projected risk*: the closed-form Pc after the covariance shrinks at
  the event's own observed rate. This is the physical mechanism of
  escalation.
- Also: covariance-shrink rate, a CARA-style parabolic risk trend, headroom
  to `max_risk_estimate`, and tracking quality.
- The short-encounter formulas are exact for the model stated in
  `physics.py`, and a test pins them. The RTN covariance sum is a documented
  approximation.

**Candidate pool (`candidates.py`)**
- *Controls*:
  - `lrp`, the baseline to beat;
  - `sesc`, the organisers' post-competition LightGBM control (Uriot et
    al. 2022, §6.1, without its quantile-transformed target): last
    admissible CDM, numerical columns only, a delta target, 25 trees with
    min_data_in_leaf 20, trained on test-like events. Despite the key, it is
    **not** team sesc's winning entry, which was a hand-tuned threshold
    cascade partly fitted against the test leaderboard (§5.3.1). In §6.1
    only 1 of 5,000 such models beat the baseline on both train and test;
  - ordinary regressors (`gbm_resid*`, `gbm_quant`, `ridge_phys`).
- *Metric-aligned hurdle models*, the core idea. MSE_HR only counts events
  whose TRUE final risk is at least -6. So, given a high-risk call, the
  optimal point forecast is `E[final | x, final >= -6]`, not `E[final | x]`.
  Ordinary regressors estimate the latter, which the ~70% of events that
  fade to -30 drag down.
  - A **hurdle** model pairs a calibrated classifier for P(final >= -6) with
    a regressor trained only on high-risk-final events.
  - **Cross-fitting**: every training event gets an out-of-fold prediction
    from 5 inner models, and Platt calibration and the decision (threshold
    τ, blend α ∈ {0, 0.5, 1}) are tuned on all eligible training events, not
    on a carve holding a dozen high-risk events. The final predictor
    averages the 5 inner models. With fewer than 40 high-risk events it
    falls back to a fixed decision near the L-optimal operating point.
  - **`hurdle_lrp`** keeps the baseline's high-risk calls, preserving its
    F2, and adds only confident early catches. It moves away from
    carry-forward only as far as held-out data supports.
- *Early warning*:
  - baselines: the logistic baseline and no-ML physics scores;
  - **`e_hurdle`**: P(final >= -6) from the hurdle classifier, which learns
    from every high-risk-final event rather than only the escalations;
  - a class-weighted classifier, ranked by its Platt probability;
  - a label-free rank blend;
  - the serving-feature classifier.

**Pre-registered primaries**
Four candidates were fixed before any real-data run: `hurdle_lrp`,
`hurdle_lrp_serving`, `e_hurdle` and `e_serving`. The gate tests each as a
single hypothesis, so it never picks the best of many and there is no
winner's curse. One non-primary is also gated: `serving_resid`, the only
VIGIL the app's current artifact format can load, which decides the VIGIL
`--export`. It is judged on its max-T multiplicity-adjusted interval.
Every other row in the report is exploratory, with max-T
multiplicity-adjusted intervals.

**Evaluation (`protocol.py`, `gate.py`)**
- R×K stratified cross-validation (default 5×5), with one row per event,
  so a row split is an event split. Earlier-cutoff augmentation rows stay
  with their parent event in every inner split.
- Training uses only events whose final CDM falls inside the cutoff. For
  the others the target equals the latest input by construction.
- Scoring uses only test-eligible events, for VIGIL and VIGIL-E alike, so CV
  and the sealed test apply the same filters. Their class mix differs,
  because the organisers enriched the test set with high-risk events
  (Uriot et al. 2022, §4.2): 0.80% high-risk and 0.24% escalating in CV,
  versus 6.92% and 1.76% on the test. CV magnitudes are therefore not
  leaderboard proxies, and not replicates of the sealed ones.
- The paired bootstrap uses **shared resample indices**: one draw of events
  per replicate is used for every candidate and every repeat, averaged over
  repeats. So the interval describes the same quantity as the reported mean.
- **Which population decides** (amended 2026-09-24, before any real-data
  run; see "Pre-run amendment" below): when a sealed test exists, every
  check except VIGIL-E's calibration slope is judged on the sealed test,
  and the other CV checks are reported as non-blocking support. The
  calibration slope always comes from CV and stays blocking, because only
  CV computes it. Without a sealed test (pooled runs), every check is
  judged on CV.
- **VIGIL gate** for the primary, on the deciding population:
  - at least 100 high-risk eligible events, otherwise "underpowered", with
    no claim either way;
  - a mean gain of at least 2%;
  - a 95% CI entirely below 0.
- **VIGIL-E gate**, on the deciding population:
  - at least 50 escalations;
  - a ΔPR-AUC CI above 0;
  - recall at the alert budget not worse;
  - a calibration slope between 0.6 and 1.6 (always from CV).
- Every verdict states the **minimum detectable effect**.
- **Data-fidelity check**: on the sealed test the latest-risk baseline must
  reproduce the published score (L 0.694, MSE_HR 0.513, F2 0.739). If it
  doesn't, the data handling is wrong and nothing else can be trusted.
- A **ceiling diagnostic** (not admissible) reruns the primary model with
  CDMs inside the cutoff, to measure how much of the outcome depends on late
  tracking updates.

**Independent review**
An adversarial statistics review of the first version found:
- a crash on near-duplicate CDM times;
- VIGIL-E scored on a population different from the test;
- winner's-curse optimism from picking the best of 13 candidates;
- unstable threshold tuning on about 18 high-risk events;
- a monotone constraint on the wrong quantity;
- augmentation leaking into early stopping;
- untrainable events;
- an unfaithful sesc;
- tie-order-dependent metrics;
- flat quantile tails;
- in-sample export thresholds.

All of them are fixed, with tests. It confirmed the protocol, the metric,
the physics formulas and the absence of outer-fold leakage.

**Second independent review (2026-09-24, before any real-data model run)**

A six-dimension audit read the harness against the real Kelvins files:
metric and protocol fidelity, leakage, statistics and gate, serving
faithfulness, real-data robustness, and VIGIL-E definitions. Each finding
was checked by three independent verifiers. It confirmed that:
- `metric.py` matches the official L, MSE_HR, F2 and latest-risk (LRP)
  definitions exactly;
- no path leads from targets, late CDMs or the test set into features or
  fitting;
- the eligibility filters reproduce the paper's counts (8,293 eligible / 66
  high-risk in train, 2,167 / 150 in test);
- LRP on the train-eligible events reproduces Uriot et al.'s Table 4 train
  column (0.804 / 0.330 / 0.411).

Fixed from it:
- **Crash on the real files.** The `*_covariance_det` columns reach 6.7e46,
  beyond float32, so every benchmark-track xgboost fit refused the data.
  They now enter as a signed log10 of the whole column (`*_slog10`). That is
  monotone, so tree splits are unchanged. `Bundle` also fails fast, naming
  the column, on any future float32 overflow. The synthetic test data had no
  such column.
- **Exported boosters were not the evaluated models.** Early-stopped models
  were saved with up to 100 trees past `best_iteration`, while the app's
  `Booster.predict` uses every tree. They are now sliced like
  `train_escalation.py` does, and a test pins served margin = lab margin.
- **Sealed-test hygiene.** `--final` is required, the commit is recorded,
  and the report now says so when there is no sealed test (pooled, or no
  targets).
- **Exploratory sealed rows** now show max-T adjusted intervals.
- **Wording.** `sesc` attribution; "same population" (see above);
  serving-claim scope (predictions ≥ 2 days before TCA; `hurdle_lrp_serving`
  needs a RiskModel extension); the export metas carry the decision point,
  and gate checks in the format the Model card reads.
- **Disclosure.** The share of CV bootstrap replicates dropped for an
  undefined ratio (F2 = 0) is now reported.

**Sealed-label looks before the final run (disclosed):**
- the test set's label counts (150 high-risk, 35 escalations), for power
  planning;
- the model-free latest-risk fidelity check: L 0.6940, MSE_HR 0.5129,
  F2 0.7391, reproducing the published score;
- during the audit, the no-ML `e_physics` score's PR-AUC and recall@5%, and
  the organisers' Eq. 2 rule cascade, which reproduces Table 4.

No trained model, and no primary candidate, had been scored on the sealed
test when this was written.

## Pre-run amendment (2026-09-24)

This amendment was made and committed before any real-data model run. The
only real-data results that existed were the label counts, the model-free
baselines listed above, and a check that every candidate fits and predicts
on the real files, which computed no metric.

**Problem.** The shipped gate counted power on CV alone. But the organisers
put 150 of the 216 eligible high-risk events in the official test set
(Uriot et al. 2022, §4.2). That leaves the training file with 66 high-risk
events (the gate needs 100) and 19 escalations (it needs 50). Under the
original rule, every verdict would have been "underpowered" by
construction, whatever the models did, and `--export` could never fire.

**Change.**
1. **VIGIL.** When a sealed test exists, the gate is judged on it: power
   from its 150 high-risk events, a mean ΔL of at most −2%, and a 95% CI
   entirely below 0. The CV mean and CI are still reported, as
   non-blocking support. Judged there, a pass *is* the sealed
   confirmation.
2. **VIGIL-E.** The same rule applies, so in the sealed run it reads
   "underpowered" (35 sealed escalations, below 50). Its deciding analysis
   is declared now: a separate pooled run (`--pool-test-into-cv`, 54
   escalations). That run is CV-only, has no sealed veto, is not
   comparable to the leaderboard, and says so in its report.
3. **Who decides what.** VIGIL's verdict comes only from the sealed run,
   and the pooled run's VIGIL rows are exploratory. VIGIL-E's verdict comes
   only from the pooled run. Neither run's result may override the other's
   role.

**Unchanged:** `candidates.PRIMARY`, the candidate pool, `MIN_HR_EVENTS`
(100), `MIN_ESCALATIONS` (50), `MIN_GAIN_PCT` (2%) and the calibration band.

VIGIL-E's calibration slope comes from CV in both runs and stays a
blocking check.

**Run plan**, one job at a time, each in its own output folder:
1. **The sealed run, once:**
   `--test ... --test-targets ... --final --repeats 5 --folds 5
   --bootstrap 500 --out reports/vigil-lab-sealed --export
   --export-dir data/models/lab`.
   The CLI refuses to overwrite a report that holds sealed results. Keep
   the report before starting step 2.
2. **The pooled run:**
   `--test ... --test-targets ... --final --pool-test-into-cv --repeats 5
   --folds 5 --bootstrap 500 --no-ceiling --out reports/vigil-lab-pooled
   --export --export-dir data/models/lab/pooled`.
   - Its ceiling diagnostic would be diluted, because test events carry no
     late CDMs.
   - The CLI never exports VIGIL from it (VIGIL's verdict is the sealed
     run's).
   - A VIGIL-E that passes lands in `data/models/lab/pooled/`, labelled as
     pooled. Promotion stays a separate, deliberate step.

**Before freezing**, a second adversarial review checked these changes: 4
reviewers, with 2 skeptics per finding. It found, and these were fixed:
- the pooled run would have overwritten the sealed report, so each run now
  gets its own `--out`, and the CLI refuses to replace sealed results;
- a pooled `--export` could have exported VIGIL;
- the adjusted verdict's MDE ignored the max-T critical value;
- the wording above did not yet match the amended gate.

The synthetic rehearsal below predates this amendment and used the old
CV-plus-veto gate.

## Results on the real ESA data (2026-09-24)

**Sealed run** (`reports/vigil-lab-sealed/`): code 034c879, scored once.
- **Fidelity.** The latest-risk baseline reproduces the published score
  exactly: L 0.694, MSE_HR 0.513, F2 0.739.
- **VIGIL primaries.** Neither passes on the sealed test (150 high-risk
  events, MDE ≈ 5–7 pp):

  | Primary | Sealed L | ΔL vs LRP [95% CI] |
  |---|---|---|
  | `hurdle_lrp` | 0.694 | +0.1% [−4.6, +4.7] |
  | `hurdle_lrp_serving` | 0.682 | −1.7% [−5.4, +1.8] |

- **App's current artifact format.** `serving_resid` is far worse than LRP
  (sealed L 41.0). Nothing was exported, and vigil-0.4 stays as shipped.
- **VIGIL-E** is underpowered here (35 escalations), as pre-declared.
- **Ceiling (CV).** The admissible primary scores 0.772 against LRP's 0.804.
  With CDMs to 1 day before TCA visible, it is 0.398 against 0.387. Most of
  the outcome is decided after the 2-day cutoff.

**Pooled VIGIL-E run** (`reports/vigil-lab-pooled/`): the pre-declared
deciding run, with code unchanged.
- **`e_hurdle` passes.** PR-AUC is 0.126 against the logistic baseline's
  0.065, a gain of +0.061 [+0.014, +0.128] on 54 escalations. Recall at the
  alert budget is not worse, and the calibration slope is 1.07. It is
  benchmark-track (it needs the CDM covariance that Kessler's live feed
  lacks), so it is a research result, not an app claim.
- **`e_serving` does not pass:** −0.013 [−0.037, +0.003].
- The run is CV-only, with no sealed test, and is not comparable to the
  leaderboard.

**What may be quoted** (Round 2, recorded 2026-10-08):
- "Locked in before testing, VIGIL scored L 0.661 on ESA's official test.
  That beats ESA's own model, 0.684: of the 5,000 machine-learning models
  ESA trained after the competition, the only one to beat the baseline on
  both the training and test data, which ESA says would have ranked 11th
  (Uriot et al. 2022, §6.1). VIGIL would rank 10th of the 97 teams."
  (Corrected 2026-10-15: this line first called 0.684 "the best of the
  5,000". ESA reports only that it was the one to beat the baseline on both.)
  Always add that the decision was set for the test's published class mix,
  and that the 4.7% gain over the baseline is not statistically
  significant at the pre-set 97.5% level.

**What may be quoted** (Round 1):
- "On ESA's official sealed test, our best deployable model scored L 0.682
  against the latest-risk baseline's 0.694 (−1.7%, not significant)."
- "With full CDM data, our pre-registered early-warning model nearly doubled
  the baseline's PR-AUC at flagging quiet events that later turned
  high-risk (0.126 vs 0.065; 54 escalations; cross-validated)."
  *(Superseded 2026-10-15 by "Round 4 correction 2": that baseline is
  weakened by pooling its per-fold fits; against the latest risk itself the
  gain is not significant. Never quote "nearly doubled" or "1.9 times".)*

## Round 2 pre-registration (2026-10-08)

This section was written and committed before any Round 2 real-data run.
Round 1 is complete (results above), so the sealed test has had one look.

**Goals**, set by the project owner, in this order:
- **A.** Make the models the app can run better. These are claims the app
  can make.
- **B.** A leaderboard entry for the official benchmark, disclosed as such.

**Primaries.** There is one per analysis, and each is a single hypothesis.

1. **`e_hurdle_eserving` (A, VIGIL-E on the app's own features).**
   - *Model:* P(final ≥ −6) from a hurdle classifier on
     `ESCALATION_FEATURE_NAMES`, trained on every high-risk-final event. It
     ports the mechanism of Round 1's passing `e_hurdle` to features the app
     computes.
   - *Judged by* a second pooled-CV analysis of the same data as Round 1's
     pooled run.
   - *Passes if:*
     - at least 50 escalations;
     - the ΔPR-AUC vs the logistic baseline has a **97.5%** CI entirely
       above 0;
     - recall at the alert budget is not worse (95% CI);
     - the calibration slope is 0.6–1.6.
2. **`hurdle_lrp_serving_bench` (B, VIGIL leaderboard entry).**
   - *Model:* Round 1's `hurdle_lrp_serving`, with the same features,
     boosters and calibration. Only its decision (τ, α) is re-tuned, by the
     same cross-fitted search, under label-shift importance weights for the
     official test's **published** class mix. That mix is 150 of 2,167
     events high-risk (Uriot et al. 2022, §4.2 and Fig. 10).
   - *Judged on* the sealed test, in its second and last look.
   - *Passes if:*
     - at least 100 high-risk events;
     - the mean ΔL is at most −2%;
     - the **97.5%** CI lies entirely below 0.

**Go/no-go before the last look** (enforced in code by `--require-cv-go`).
The sealed test is scored only if `hurdle_lrp_serving_bench` beats LRP in
the training file's 5×5 CV, with the predictions re-weighted to the official
test's class mix (ΔL < 0). Otherwise the look is saved, and the report says
so.

**Exploratory rows** (reported, never claimed):
- `e_hurdle_eserving_aux` and `hurdle_eserving_aux` use near-miss labels:
  - For events below −6 at the cutoff, the classifier learns "crossed −6 at
    any CDM inside the cutoff". Late CDMs are labels only, never inputs.
  - Among trainable candidates that gives 88 positives instead of 23.
  - Calibration stays on the true label.
- The `hurdle_eserving` regression row.
- A "CV ΔL at test mix" column for every candidate.

**What B is and is not.** B's decision is a scoring adjustment for the test
set's enriched class mix: 6.9% high-risk, against 0.8% among the training
file's eligible events. It is standard label-shift decision theory with a
published prior. But it does not make live predictions better. The real sky
has the training file's mix, where the same operating point raises false
alarms. A live deployment would use the same model's natural-prior decision.
B is always quoted as "on the official benchmark's class mix".

**Order and outputs.** One job at a time. The sealed run goes first, so
pooled-CV numbers (which include test events) cannot inform the sealed look.
1. **Sealed run:**
   `--test … --test-targets … --final --require-cv-go --repeats 5 --folds 5
   --bootstrap 1000 --no-ceiling --out reports/vigil-lab-r2-sealed`.
2. **Pooled run:**
   `--test … --test-targets … --final --pool-test-into-cv --repeats 5
   --folds 5 --bootstrap 1000 --no-ceiling --out reports/vigil-lab-r2-pooled`.

There is no `--export`: a Round 2 VIGIL-E primary has no exporter in the
app's artifact format yet. Promotion would be a separate, deliberate step.

**Unchanged:**
- `MIN_HR_EVENTS` 100, `MIN_ESCALATIONS` 50, `MIN_GAIN_PCT` 2% and the
  calibration band;
- the amendment's rule that a sealed test, when present, is the
  confirmatory population.

**Disclosures.**
- Round 2 was designed with Round 1's sealed and pooled results known. The
  stricter 97.5% bar and this note account for that.
- The design for B came from training-file counts and the published class
  mix. No per-candidate sealed number was used to choose B's model or its
  decision rule.
- A label-free check confirmed that every Round 2 candidate fits and
  predicts on the real files. It computed no score.

## Competition mode: a leaderboard entry under the 2019 rules (2026-10-08)

Round 2's pre-registered leaderboard entry scored L 0.661 on the sealed
test, which is rank 10 of 97. That was its last *scientific* look. The 2019
leaderboard itself was not made that way: teams submitted predictions for
the test events and got their score back, up to two submissions a day for
about two months. The winner's cascade was tuned that way. When the
organisers re-ran it on random splits, its F2 fell about 14% below the
baseline's (Uriot et al. 2022, footnote 4).

To compete for the leaderboard on the teams' own terms, without inventing
any advantage, `backend/app/models/lab/compete.py` runs under these rules,
fixed before any scoring:

- **Split.** The official test is split once (seed 2019, stratified on the
  label) into a PUBLIC and a PRIVATE half. The split is written by
  `--make-split` and committed before `--run`.
- **Training.** Models train on the training file only. Test labels never
  enter training, and no per-event test result is ever inspected.
- **Feedback.** The only feedback is the *aggregate* score on the public
  half. Every submission is logged in `ledger.jsonl`, capped at 60
  submissions, well under what the 2019 rules allowed.
- **Entry family.** The benchmark and serving hurdle classifiers
  (logit-blended) decide which sub-threshold events to call high-risk, with
  an optional latest-risk band. Learned or fixed catch values, a high-value
  cap, a drop rule for weak base-high calls and the published "safe" rules
  (payload chaser, small target, miss > 30 km) are switches. The
  latest-risk baseline is a member of the family.
- **Search.**
  1. Stage 1 ranks every configuration on the training file's out-of-fold
     predictions, at the test's published class mix. It uses no test data.
  2. Stage 2 submits the best configuration per classifier blend and
     override set.
  3. Stage 3 spends the rest of the budget on one-step neighbours of the
     best public configurations.
- **Final score.** The best public configuration is scored once on the
  untouched private half and on the full test.

**Reporting.** Only the full-test score is comparable to the leaderboard,
as the teams' were. It is always quoted with how it was made: "competition
rules, N public submissions". The private-half score is reported beside it.
Team sesc's published cascade is scored as a credited reference row, not as
a VIGIL entry. The entry is a benchmark result tuned to the official test's
distribution, like the 2019 entries; it says nothing about live
performance.

### Competition-mode result (2026-10-08) and why it is not a fair placement

The run is in `reports/compete/` (code 5e4a9e6). It used 60 of 60 public
submissions, every one in `ledger.jsonl`.

| | Public half | Private half (scored once) | Full test |
|---|---|---|---|
| VIGIL entry | 0.542 | 0.670 | 0.605 |
| Latest-risk baseline | 0.677 | 0.711 | 0.694 |
| sesc's published cascade (reference) | 0.507 | 0.634 | 0.569 |

A full-test score of 0.605 would rank 4th of 97. **It must not be quoted as
VIGIL's placement**, for two reasons:
- **Hindsight.** The entry family included constants the 2019 winner
  published after tuning them on the leaderboard: the −6.4 and −7.3 risk
  bands, the payload and t_span < 0.5 overrides, the 30 km rule, and the
  −5.6 catch value and −4/−3.5 caps. The selected entry uses several of
  them. The 2019 teams did not have this information.
- **The reference row beats it.** Those published rules alone score 0.569
  on the full test and 0.634 on the private half, better than the entry on
  every subset. VIGIL's classifiers did not add value on top of the
  winner's recipe.

The honest leaderboard-comparable number therefore remains **Round 2's
pre-registered sealed result**: L 0.661 (rank 10 of 97). It used no test
feedback and no winner's rules.

A 1st place (below 0.5553) is not reachable without tuning on the full test
set or adopting the winner's published, test-tuned answer. Neither would be
a real result.

## Round 3 pre-registration (2026-10-15): analyses for the research paper

This section was written and committed before any Round 3 computation.
Rounds 1–2 and competition mode are complete, and their results are known.

**The sealed test is spent, so no Round 3 analysis trains a model on the
training file and scores it on the official test as a held-out set.** (P4
pools the test events into cross-validation, as Round 1's deciding VIGIL-E
run did; this clarification was added before any computation.) The other
analyses that use the test's labels apply only these:
- the latest-risk baseline (LRP);
- team sesc's published rules;
- label-free random perturbations.

They measure the benchmark, not a model.

New code lives in `backend/app/models/lab/paper.py`; the frozen lab modules
are not edited. Outputs go to `reports/paper/`.

**P1. The information horizon** (training file only; the test files hold no
CDMs between the cutoff and the final message).
- *Cutoffs:* c ∈ {2.0, 1.75, 1.5, 1.25, 1.0} days. At cutoff c a model sees
  every CDM with time_to_tca ≥ c, and never the final CDM.
- *Events:* scoring uses the training events that pass the official test
  filters, the same events at every cutoff. Training uses the events whose
  final CDM lies inside the cutoff.
- *Models:*
  - LRP.
  - `hurdle_lrp`, the ceiling diagnostic's model, with its natural-mix
    decision.
  - The same fitted model with its decision re-tuned for the test's
    published class mix (`with_prior`), scored with label-shift weights.
- *Protocol:* 3 repeats × 5 folds, seeds as in `protocol.ceiling`, and a
  paired bootstrap with 1,000 shared-index replicates.
- *Reported:*
  - L, MSE_HR and F2 for each model and cutoff;
  - ΔL vs LRP with a 95% CI at each cutoff;
  - LRP's L(c)/L(2.0) with a 95% CI.
- *Question:* how much of the outcome is decided by information that
  arrives after the 2-day cutoff, and does any model close that gap at
  c = 2.0? There is no pass/fail.

**P2. What the official test can resolve** (test labels; LRP and published
rules only).
- (a) A bootstrap of LRP's L on the full test (2,000 replicates over events),
  with its 95% interval.
- (b) The minimum detectable effect of every primary in the committed sealed
  reports. MDE = 2.8 × the bootstrap SD of ΔL. Also the number of high-risk
  test events needed to detect a 2% and a 5% gain, assuming the SD scales as
  1/√n_HR.
- (c) The 2019 design: 2,000 random visible subsets with p_high = p_low =
  0.9 (Uriot et al. 2022, §4.2). Reported:
  - LRP's visible-subset L relative to the full test;
  - sesc's published cascade's ΔL vs LRP on the full test, on the visible
    subset and on the 10% hold-out.
- (d) The winner's rules out of sample. sesc's published cascade
  (`compete.sesc_cascade`, whose constants were tuned on the test
  leaderboard) is scored on the training file's eligible events, at the
  natural mix and at the test's published mix. Reported: ΔL vs LRP with a
  95% bootstrap CI.

**P3. Zero-information leaderboard climbing** (test labels; a simulation of
the 2019 protocol).
- *The simulated team* has no information beyond the LRP, and starts from
  the LRP's predictions.
- *Each submission:*
  1. Draw k events at random from the whole test set (a team cannot know
     which events are visible).
  2. Perturb them. With probability 1/2, toggle each chosen event's call
     (low → −5.5, high → −6.001). Otherwise, add N(0, 0.3²) decades to each
     chosen event that is currently called high.
  3. Keep the change only if the visible-subset score improves.
- *Visible subset:* a stratified random 90% of each class, the organisers'
  p_high = p_low = 0.9.
- *Budget:* 90 submissions (two a day for a month, then one a day for a
  month).
- *Subset size:* k = 20 is primary; k ∈ {5, 50} is a sensitivity check.
- *Runs:* 500 independent runs, each with a fresh visible subset and fresh
  moves.
- *Reported:*
  - the median and 5–95% range of ΔL vs LRP on the visible subset, on the
    full test (the 2019 final ranking used the full test) and on the 10%
    hold-out;
  - the 2019 rank of the median full-test score.
- The moves carry no information, so any full-test gain is adaptive
  overfitting. There is no pass/fail.

**P4. What carries VIGIL-E's early-warning signal** (pooled CV: the data and
protocol of Round 1's deciding run, i.e. train + test events, 5 × 5 folds,
1,000 shared-index bootstrap replicates, seed 42).
- *Covariance-derived features*, by a fixed name rule:
  - Kelvins columns containing `_sigma_`, `covariance_det`, `mahalanobis` or
    `max_risk_`, or matching `^[tc]_c(t|n|rdot|tdot|ndot)_` (the
    correlation terms);
  - every `p_*` encounter-plane feature;
  - `h_log_sigma_rate`, `h_c_log_sigma_rate`, `h_sigma_shrink_to_eval`,
    `h_maha_slope` and `h_headroom_mre`.
- *Variants* of `e_hurdle` (the same `Hurdle("hurdle")` classifier,
  P(final ≥ −6)):
  - A0: all benchmark features (Round 1's model, refit here);
  - A1: A0 minus the covariance-derived features;
  - A2: the serving features plus the covariance-derived features.
- *Baseline:* `e_logit`.
- *Primary contrast:* ΔPR-AUC(A0 − A1), with a 95% CI. *Secondary:*
  A1 − e_logit, A2 − e_logit and A2 − A0.
- *Reading:*
  - Covariance carries the signal if A0 − A1 has a CI above 0 and
    A1 − e_logit has a CI that includes 0.
  - A CI for A0 − A1 that includes 0 is inconclusive, never "no effect".
    There are 54 escalations, and the MDE is about 0.05–0.07.

**P5. The organisers' significance test** (published numbers only).
- Uriot et al. (2022, §6.3) compare LightGBM with LRP over 505 Monte Carlo
  splits (test size 0.2) with a paired t-test (t = −10.23).
- The splits overlap, so the test's independence assumption fails.
- We apply the corrected resampled t-test (Nadeau & Bengio, 2003):
  t_corr = t · sqrt((1/J) / (1/J + n_test/n_train)), with J = 505 and
  n_test/n_train = 0.25. We report the two-sided p (df = 504).

**Reporting.**
- Every number in the paper comes from a committed report, through a
  script.
- Confidence intervals are 95% unless stated.
- Rounds 1–2 keep their claim rules (above).

**Disclosure.** Round 3 was designed with every earlier result known. None
of its analyses re-tests a Round 1–2 conclusion, and none can change one.

## Round 3 results (2026-10-15)

All five analyses ran once, with code `948ad3a` (clean), into `reports/paper/`.
The paper built from them is in `paper/`.

- **P1, information horizon** (training file, 8,293 eligible events, 66
  high-risk).
  - LRP's L falls from 0.804 at a 2-day cutoff to 0.387 at 1 day: −52%,
    95% CI 27–74%. At the test's class mix the drop is 39%.
  - VIGIL's edge shrinks from −3.2% [−9.1, +3.4] (−12.7% [−26.2, +2.7] at
    the test's mix) at 2 days to +0.2% [−2.1, +2.7] at 1 day.
  - At c = 2.0, repeat 1 reproduces the Round 1 ceiling exactly (0.772).
- **P2, test resolution.**
  - LRP's own L under bootstrap re-draws of the test spans 0.46–0.95.
  - The MDE is 5.2% (live-feed VIGIL), 6.5% (benchmark VIGIL) and 11.4%
    (Round 2 primary). Detecting 2% would need about 1,000–1,600 high-risk
    events.
  - On 2019-style visible subsets, LRP's score moves 4.8% on average. On the
    10% hold-out, sesc's cascade ranges from −64% to +39%.
  - sesc's cascade on the training file: −7.2% [−25.0, +14.5] at the test's
    mix, and +30.7% [+4.5, +64.8] (worse) at the natural mix. On the test it
    scores −18.0%.
- **P3, zero-information climbing** (k = 20, 90 submissions, 500 runs).
  - The median full-test ΔL is −4.6% (5–95%: −10.8% to +1.5%), and the
    median L of 0.662 would rank 10th.
  - The hidden 10% gets worse: +8.0% median.
  - k = 5 and k = 50 give −4.2% and −4.5%.
- **P4, early-warning ablation** (pooled CV, 54 escalations). A0 reproduces
  Round 1's pooled e_hurdle exactly (0.1261).
  - A1 (no covariance) scores 0.130 and A2 (serving + covariance) 0.125,
    against e_logit's 0.065.
  - A0 − A1 = −0.004 [−0.025, +0.023]. The pre-registered reading is
    **inconclusive**, but the point estimate is about zero.
  - A1 − logit = +0.065 [+0.015, +0.134] and A2 − logit = +0.060
    [+0.013, +0.119].
  - **So the covariance is not what makes early warning work.** The
    covariance and the other full-CDM fields each carry the signal; the
    live-feed fields alone do not. This corrects an earlier informal
    claim that the gain needed the covariance.
- **P5, corrected t-test.** The organisers' t = −10.23 becomes −0.91
  (p = 0.36). The 95% CI of their +1.28% F2 gain is [−1.5, +4.0]%.

## Round 4 pre-registration (2026-10-15): coherence, and early warning from public data

This section was written and committed before any Round 4 confirmatory
computation. Rounds 1–3 are complete, and their results are known.

**What was looked at first (exploratory, training file only).** Before this
was written, exploratory analyses ran on the training file's events, and on
no test event:
- *Coherence of the CDM risk* (eligible events, latest admissible CDM vs
  later CDMs; the bound and the tests are defined in P6):
  - Candidates with −29.9 < r < −12: 18 of 1,855 later reach ≥ −6 at some
    CDM, where a coherent probability allows at most 7.3 × 10⁻⁵. With
    −12 ≤ r < −8: 14 of 815 against at most 1.07. With −8 ≤ r < −6: 45 of
    634 against at most 134 (consistent).
  - LRP's calls (r ≥ −6), reweighted to the natural class mix: the mean
    final Pc is 0.46 [0.24, 0.72] times the mean latest Pc for
    −6 ≤ r < −4.
  - Consecutive pre-cutoff updates (rotated covariance, P6c): median
    d² 3.03; 36% exceed 11.34 (a coherent filter allows at most 1%).
  - Recomputing Pc from the CDM fields: the median |Δlog10 Pc| against
    ESA's risk is 0.51 decades if the two covariances are added in their own
    frames (as `physics.py` does), and 0.12 if the chaser's covariance is
    first rotated into the target's frame (4,000 CDMs with risk > −20).
- *Model screen* (3 repeats × 5 folds of training-file CV): about a dozen
  variants of the hurdle classifier, including different trees, monotone
  constraints, seed bagging, near-miss labels, earlier-cutoff augmentation,
  a ranking objective, a continuous target and the rotated physics features.
  None improved log loss, early-warning PR-AUC or the fade AUC by more than
  the noise; some were worse. **No new forecasting model is proposed, and VIGIL is unchanged.**
- *Public-counterpart features*: one run of an earlier rule (70 features)
  gave an early-warning PR-AUC of 0.116 against A0's 0.125, live feed's
  0.037 and 0.050 for a logistic baseline on the latest risk (the P8
  baseline, not `e_logit`). The rule was then tightened
  (P7) to drop quantities without a reliable public source, and run once:
  0.100. The tightened rule is fixed from here on, whatever the result.

The confirmatory analyses below use data that the exploration did not touch
(P6 uses only the test events) or the pooled design of Round 1's deciding
run and P4 (P7–P8, whose test events are new to these questions).

**The sealed test stays spent.** No Round 4 analysis trains a model on the
training file and scores it on the official test as a held-out set. P6 uses
the test's labels with no model at all. New code lives in
`backend/app/models/lab/round4.py`; the frozen lab modules are not edited.
Outputs go to `reports/paper/`.

**P6. Is the CDM risk a coherent forecast of itself?** (test events only; no
model.)
- *Theory.* If each CDM's Pc were a calibrated probability given what was
  known at that message, the sequence of Pc values would be a martingale:
  the expected value of any later Pc equals the current one. Then, for the
  latest admissible CDM with Pc = 10^r:
  - P(the final Pc ≥ 10⁻⁶) ≤ 10^(r+6) (Markov's inequality), and
  - P(any later Pc ≥ 10⁻⁶) ≤ 10^(r+6) (Ville's maximal inequality);
  - the mean later Pc equals the mean current Pc.
- *Data.* The 2,167 official test events. Their CDMs after the cutoff come
  from ESA's raw file (`raw_data_2015-2019.gz`), matched to each test event
  by its pre-cutoff CDMs (mission, time_to_tca, risk, miss distance) and
  accepted only if exactly one raw event matches and its final CDM equals
  the published final CDM. Unmatched events enter (a) through their final
  CDM only.
- *Selection correction.* The organisers split events by outcome (150 of 216
  high-risk eligible events, but 2,017 of 10,244 others, went to the test).
  Within the test set the coherent bounds become
  b = min(1, ρ · 10^(r+6)) with ρ = (150/216)/(2,017/10,244) = 3.53, and
  means are weighted by 1/s, with s = 150/216 for events that end high-risk
  and 2,017/10,244 otherwise.
- (a) *Low end (primary).* Escalation candidates with −29.9 < r < −8 (the
  −30 sentinel is reported separately). Count the events whose any later CDM
  reaches ≥ −6, and compare with M = Σ b. The Chernoff bound
  P(N ≥ k) ≤ e^(−M)(eM/k)^k gives a one-sided p. **Incoherent** if
  p < 0.001. Also reported: the same with the final CDM only, and the bands
  −29.9 < r < −12 and −12 ≤ r < −8.
- (b) *High end.* LRP's calls with −6 ≤ r < −4: the ratio of the weighted
  mean final Pc to the weighted mean latest Pc, with a 95% CI from 2,000
  bootstrap replicates over events. **Drifts down** if the CI lies below 1.
  Also reported: the same at the first CDM after the cutoff.
- (c) *Update realism.* Every consecutive pair of pre-cutoff CDMs:
  d² = Δxᵀ C⁻¹ Δx, where Δx is the change in the predicted relative position
  at TCA and C is the earlier message's combined position covariance, with
  the chaser's covariance rotated into the target's RTN frame (both objects
  at the conjunction point, target velocity (0, √(μ/a), 0); pairs whose
  target eccentricity exceeds 0.1 are skipped). A coherent sequence keeps
  d² below 11.34 (the χ²₃ 99th percentile) at least 99% of the time.
  **Unrealistic** if the lower limit of the 95% CI (bootstrap over events)
  of the share above 11.34 exceeds 1%.
- (d) *Methods check* (descriptive). Median |Δlog10 Pc| between ESA's risk
  and the Pc recomputed from the same fields, unrotated and rotated, over the
  test events' pre-cutoff CDMs with risk > −20.

**P7. Early warning from public-data counterparts** (pooled CV: the data and
protocol of P4, i.e. train + test events, 5 × 5 folds, seed 42, 1,000
shared-index bootstrap replicates).
- *Public-counterpart rule* (fixed): the 11 live-feed features; every `h_*`
  history feature except `h_log_sigma_rate`, `h_c_log_sigma_rate`,
  `h_sigma_shrink_to_eval`, `h_maha_slope` and `h_headroom_mre`; the chaser
  object type (`l_ctype_*`); from the latest CDM, `time_to_tca`, `risk`,
  `miss_distance`, `relative_speed`, `relative_velocity_r/t/n`,
  `geocentric_latitude`, `azimuth`, `elevation`, both objects' `j2k_sma`,
  `j2k_ecc`, `j2k_inc`, `h_apo`, `h_per` and `cd_area_over_mass` (public
  through the catalogue's element sets and drag term), and `F10`, `F3M`,
  `SSN`, `AP`; and both objects' `rcs_estimate` only as the public catalogue's
  size class (small < 0.1 m² ≤ medium < 1 m² ≤ large). No covariance, no
  orbit-determination quality, no relative-position components and no object
  span. These are ESA's recorded values of quantities with public sources,
  not values computed from public data.
- *Variants* of `e_hurdle` (the same `Hurdle("hurdle")` classifier):
  - A0: all benchmark features (P4's A0);
  - PUB: the public-counterpart features;
  - LIVE: the 11 live-feed features.
- *Baseline:* `e_logit`.
- *Primary:* ΔPR-AUC(PUB − e_logit), with a 95% CI. **Supported** if the CI
  lies above 0; a CI that includes 0 is inconclusive.
- *Secondary:* A0 − PUB and PUB − LIVE; recall within the 5% alert budget;
  and the same metrics restricted to the test events' candidates.

**P8. VIGIL's classifier as a probability forecast and a false-alarm
filter** (the same pooled run; A0 and PUB).
- *Baseline:* a calibrated latest-risk forecast: logistic regression of
  1[final ≥ −6] on (r, 1[r ≥ −6], max(r, −12)), with C = 10, fit on the
  training fold's trainable events.
- *Primary:* on all pooled eligible events, Δ log loss of P(final ≥ −6),
  A0 − baseline, with a 95% CI (bootstrap over events). **Better** if the CI
  lies below 0.
- *Secondary:* the Brier score; and among LRP's calls (r ≥ −6), the ROC AUC
  for "stays high-risk" and the share of LRP's false alarms each model
  removes while keeping 95% of its true calls.
- No claim about the official loss (Round 1's pooled run reported it:
  hurdle_lrp +1.0% [−1.7, +4.0]).

**Reporting.** Every number in the paper comes from a committed report,
through a script. Confidence intervals are 95% unless stated. Earlier rounds
keep their claim rules.

**Disclosure.** Round 4 was designed after the exploration above, with every
earlier result known. P6 confirms on events the exploration did not use. P7
and P8 reuse the pooled design, whose training-file part the exploration
used.

## Round 4 results (2026-10-15)

Both analyses ran once, with code `6e8b6e2` (clean), into `reports/paper/`
(`coherence.json`, `public.json`).

- **P6, coherence of the CDM risk** (official test events; 2,142 of 2,167
  matched to a unique raw sequence; ρ = 3.53).
  - (a) **Incoherent** (primary). Candidates with −29.9 < r < −8: 16 of 623
    later reach ≥ −6, where a calibrated Pc allows at most 0.98
    (Chernoff p ≤ 1.4 × 10⁻¹³). Final CDM only: 9 (p ≤ 6.8 × 10⁻⁶). Below
    −12: 7 of 444 against at most 6.4 × 10⁻⁵. Between −12 and −8: 9 of 179
    against 0.98. Between −8 and −6: 36 of 210 against at most 106
    (consistent). At the −30 floor: 6 of 1,131.
  - (b) **Not shown.** LRP's calls with −6 ≤ r < −4 (155): weighted mean
    final Pc / mean latest Pc = 1.13 [0.45, 2.29]; at the next CDM, 0.92
    [0.73, 1.08]. The downward drift seen in the training file did not
    replicate.
  - (c) **Unrealistic.** 35.1% [33.7, 36.6] of 22,311 consecutive
    pre-cutoff updates exceed 11.34 (coherent: at most 1%); median d² 2.94.
  - (d) ESA's risk recomputed from the CDM fields: median |Δlog10 Pc| 0.41
    unrotated, 0.075 rotated (15,668 test CDMs).
- **P7, public-data counterparts** (pooled CV, 54 escalations; A0
  reproduces P4's 0.1261 exactly). PUB (65 features) scores 0.114 against
  e_logit's 0.065: +0.049 [+0.009, +0.114], **supported**. A0 − PUB = +0.012
  [−0.024, +0.050]; PUB − LIVE = +0.032 [−0.005, +0.080]; LIVE − logit =
  +0.017 [−0.014, +0.067]. On the test events' 35 escalations alone,
  PUB − logit = +0.048 [−0.009, +0.124].
- **P8, VIGIL's classifier as a probability forecast** (same run).
  - Log loss over 10,460 events: A0 0.0503 against the calibrated latest
    risk's 0.0556, Δ = −0.0053 [−0.0077, −0.0028], **better** (−10%). PUB
    0.0518, Δ = −0.0038 [−0.0062, −0.0013].
  - LRP's 486 calls (162 stay high-risk): ROC area A0 0.727, PUB 0.683,
    latest risk 0.581. At 95% of true calls kept, A0 removes 28% [20, 35]
    of the false alarms, PUB 20% [13, 29], the latest risk 6% [4, 11].

**What may be quoted** (Round 4, recorded 2026-10-15):
- "Below a Pc of 10⁻⁸, ESA's CDMs later reach 10⁻⁶ about 16 times as often
  as any calibrated probability allows; just below the high-risk line they
  behave as a calibrated forecast." Say that this is a model-free test on
  the official test events. *(Superseded by the correction below: the second
  half overstates what the test shows.)*
- "As a probability forecast, VIGIL beats the calibrated latest risk (10%
  lower log loss) and removes 28% of its false alarms while keeping 95% of
  its true calls; quantities with public counterparts keep most of each
  gain." Always add: pooled cross-validation, not a sealed test; the
  official loss shows no gain.
- Never say VIGIL beats the baseline on the official loss.

## Round 4 correction (2026-10-15, post hoc)

A review of the paper, after the results above were recorded, found one flaw
in P6(c) and two overstatements. The registered results stay as recorded.
The checks below were computed first in an audit script, then by
`round4 posthoc` (commit `6a3bd02`, clean) into
`reports/paper/coherence_posthoc.json`. They are **not pre-registered**.

- **P6(c) as registered is inflated.** Each CDM gives the relative position
  at TCA, which is perpendicular to the relative velocity: an error along
  the direction of motion shows up as a change of TCA, not of that position.
  So Δx lies in the encounter plane (median |cos(x, v)| = 0.0001 over 24,484
  test CDMs). For a vector in that plane, the 3-D length Δxᵀ C⁻¹ Δx is at
  least the planar one, and much larger when C correlates the plane with the
  direction of motion, as propagated covariances do. Here it is larger by a
  median factor of 60.
- **The corrected test.** In the earlier message's encounter plane, with C
  projected onto it, a coherent sequence's d² is no larger than χ²₂ (99th
  percentile 9.21). Over the same 22,311 updates: median d² 0.051 (χ²₂: 1.39,
  so most updates are small, which coherence allows); **3.5% [3.2, 3.9]**
  exceed 9.21, where a coherent sequence allows at most 1%. The covariance
  still understates the rare large update, but by a factor of about 3.5, not
  35. The registered statistic reproduces 35.1%.
- **Robustness of P6(a)** (descriptive). The 16 crossings come from 7
  missions. The Chernoff p treats encounters as independent; without that,
  Markov's inequality bounds the chance of 16 or more by 0.98 / 16 = 0.06.
  The test events were also filtered on their later messages (final CDM
  within a day of TCA). In the training file, 48% (−12 to −8) to 67% (−29.9
  to −12) of the candidates pass those filters, so the filter could plausibly
  double the allowed rate, not multiply it sixteenfold.
- **Overstatements.** That Pc stays within the bound between 10⁻⁸ and 10⁻⁶
  (36 against at most 106) is consistency, not proof of calibration: so close
  to the line the bound is loose. And a martingale fixes only the mean of the
  later Pc, so a coherent Pc near the line does not mean the message holds
  nothing more (P8 shows it does).

**What may be quoted instead:**
- "Below a Pc of 10⁻⁸, ESA's CDMs later reach 10⁻⁶ about 16 times as often
  as a calibrated probability allows (p ≤ 1.4 × 10⁻¹³, treating encounters as
  independent); just below the high-risk line the test finds no violation."
- "Measured in the encounter plane, 3.5% of message-to-message updates fall
  outside the earlier message's 99% uncertainty region, where a coherent
  sequence allows 1%." Say that this corrects the registered statistic, post
  hoc. Never quote the 35%.

## Round 4 correction 2 (2026-10-15, post hoc): the latest-risk baselines

An independent re-check of the paper (fresh code, no lab imports, from ESA's
files) reproduced its numbers, and found one more flaw: the registered
latest-risk baselines are weaker than the latest risk itself. Computed by
`round4 ewposthoc` (commit `edee703`, clean; `reports/paper/ew_posthoc.json`)
and `round4 posthoc` (commit `f709951`; `coherence_posthoc.json`). **Not
pre-registered**; the registered results stay as recorded.

- **Why.** `e_logit` (and P8's calibrated latest risk) is fitted in each
  fold. Each fold's curve ranks events exactly by the latest risk, but the
  curves differ (slopes 0.145–0.198 here), so pooling the out-of-fold scores
  into one ranking reorders events of similar risk. Ranking by the latest
  risk itself needs no fit.
- **Early warning** (54 escalations; ewposthoc repeats public() and ablation()
  exactly; every registered PR-AUC is reproduced). The latest-risk ranking
  scores PR-AUC 0.098 (ROC 0.862, recall@5% 0.54), against e_logit's 0.065:
  −0.033 [−0.099, −0.003]. Against that ranking, no model's PR-AUC gain is
  significant: A0 +0.028 [−0.027, +0.077], A1 +0.032 [−0.025, +0.080], A2
  +0.027 [−0.023, +0.069], PUB +0.016 [−0.031, +0.060], LIVE −0.017
  [−0.070, +0.014]. Recall@5% is no better than the ranking's (A0 0.52). The
  ROC area is better: A0 0.920, +0.058 [+0.016, +0.107]; A1 +0.064; A2
  +0.048; PUB +0.060 (all CIs above 0); LIVE +0.037 [−0.004, +0.084].
- **False alarms.** Ranking LRP's calls by the latest risk itself: ROC 0.592
  and 9% [5, 15] of false alarms removed at 95% of true calls (the
  calibrated baseline's pooled 0.581 and 6%). A0 still leads: ROC +0.135
  [+0.073, +0.194]; PUB +0.091 [+0.027, +0.155]. Log loss is unaffected
  (each event is scored by its own fold's model).
- **Coherence, robustness.** The plane share is 3.4–3.8% under other
  treatments of the 0.2% of correlations beyond ±0.999 (3.5% registered
  clip). The rotated combined covariance reproduces ESA's
  mahalanobis_distance to within 1% for 92% of test CDMs (unrotated: 4%). A
  tolerance-based matcher (rounded keys can split two copies of one value)
  matches 2,144 test events, not 2,142; both extra events are at the −30
  floor and do not cross, so no other P6 count changes.
- **Leaderboard.** On ESA's final results page, 11 teams scored below LRP;
  the twelfth, ANTwork (0.6939636), equals LRP (0.6939613) to five decimals.
  Say "12 of 97 matched or beat it".

**What may be quoted instead** (early warning):
- "VIGIL-E ranks quiet encounters that later turn dangerous better overall
  than their latest risk (ROC area 0.920 against 0.862), but not at the top of
  the list: its precision-recall gain is not significant." Say pooled CV, post
  hoc. Never say VIGIL-E beats the latest risk at early warning.
- "Quantities with public counterparts keep most of the probability-forecast
  and false-alarm gains." Not "most of the early-warning gain".

## Review checks (2026-10-18, post hoc)

An outside review of the paper asked for five more checks. All are post hoc, none looks at
the sealed test again, and every registered result stays as recorded. Computed by
`review coherence` and `review cv` (`backend/app/models/lab/review.py`; reports
`reports/paper/review_coherence.json` and `review_cv.json`). The cv run repeats the folds and
seeds of `public()` and `ew_posthoc()`, and every registered pooled number is reproduced.

- **P6(a) by mission.** The 16 crossings come from 7 of the 16 missions with candidates
  between -29.9 and -8. A mission's chance of any crossing is at most the sum of its
  encounters' bounds (a union bound), so at most 0.98 missions are expected to cross;
  7 did: p <= 4.5e-04, assuming only that missions are independent. The
  encounter-level p, 1.4e-13, assumes independent encounters.
- **Maneuvers and dilution.** By the chaser's object type the crossings are 8 debris,
  7 unknown and 1 payload, so maneuvers by the chaser explain at most one. By ESA's
  `max_risk_scaling`, 12 sit outside the dilution region (scaling of at least 1: the miss
  is many sigma wide, so the covariance is too small) and 4 inside it.
- **Corrected CV intervals.** The registered bootstrap holds each fold's model fixed. The
  corrected repeated k-fold t-test (Nadeau and Bengio 2003; Bouckaert and Frank 2004) on
  the 25 fold-level differences gives:
  - P8 log loss: A0 against the calibrated latest risk -9.6% [-16, -3]%
    (p 0.005); against an isotonic latest risk -9.8% [-19, -1]% (p 0.035);
    isotonic and logistic do not differ. PUB against isotonic -7.0% [-16, +2]%
    (p 0.11): not significant.
  - Early warning against the latest risk: no PR-area gain is significant (A0 +0.036 [-0.048, +0.120]).
    ROC gains hold for A0 (+0.059 [+0.003, +0.115], p 0.04), A1 and PUB, and not for
    A2 (p 0.10) or LIVE.
- **False alarms, deployable.** With the 95% threshold set inside the training folds (nested
  3-fold CV), A0 removes 24% of the held-out false alarms and keeps 96% of the true calls;
  the latest risk, thresholded the same way, removes 10%. A0 minus latest: +13.8 points [+5.1, +22.5]
  (p 0.003); PUB 18%, +8.5 points [+0.0, +17.0] (p 0.049). The registered 28%
  set its threshold on the events it scored.
- **The frame error.** Refitting with the chaser's covariance rotated into the target's frame
  in the physics features (all 28,209 rows) changes nothing: log loss
  -0.00004 [-0.00039, +0.00032], false alarms +0.5 points [-3.2, +4.3], PR area -0.011 [-0.034, +0.013].
- **Class mix.** Pooling the training and test events restores ESA's full population, so P8
  is already at the operational class mix.

**What may be quoted instead** (false alarms): "With thresholds set inside the training
folds, VIGIL's classifier removes about a quarter (24%) of the baseline's false alarms
while keeping 96% of its true calls; the latest risk alone removes 10%." Say pooled CV,
post hoc. Quantities with public counterparts keep part of the gain, not most of it
significantly.

## Results so far (synthetic rehearsal only)

This is a full CLI run on **synthetic** Kelvins-shaped data (5,650 training
events, 1,414 sealed-test events, 2×5 CV, about 12 minutes). It shows the
harness works end to end. It says nothing about real performance, and the
fidelity check correctly flags that this is not the ESA data.

| Candidate | CV ΔL vs LRP [95% CI] | Sealed-test ΔL [95% CI] | Gate |
|---|---|---|---|
| **hurdle_lrp** (primary) | −34% [−46, −20] | −44% [−58, −23] | passes, confirmed |
| **hurdle_lrp_serving** (primary) | −28% [−40, −13] | −42% [−55, −26] | passes, confirmed |
| sesc / ordinary regressors | +100% … +3000% or ∞ | similar | — |

- The ordinary regressors lose badly to the baseline, as they did in the
  real 2019 challenge, where only 12 of 97 teams matched or beat it. The metric-aligned
  hurdle design is what turns that around.
- VIGIL-E's primary `e_hurdle` improved PR-AUC by +0.10 in CV (CI above 0)
  and +0.11 on the sealed test. It was correctly marked **underpowered**,
  because only 28 escalations were scorable.
- On the real file, if too few escalations are test-eligible, re-run with
  `--pool-test-into-cv` to score VIGIL-E on all of them.
- The ceiling diagnostic: letting the model also see CDMs 1–2 days out would
  improve L from 1.86 to 1.51. That gap is what no admissible model can
  recover.

## Handoff: how to run it on the real ESA data

From the repo root, with the Kelvins files in `data/kelvins/`:

```bash
# quick smoke run (a few minutes) — WITHOUT the sealed targets
.venv/bin/python -m backend.app.models.lab \
    --train data/kelvins/train_data.csv \
    --repeats 1 --folds 3 --bootstrap 100 --out reports/vigil-lab-smoke

# the real run (longer; use more repeats if time allows)
.venv/bin/python -m backend.app.models.lab \
    --train data/kelvins/train_data.csv \
    --test data/kelvins/test_data.csv --test-targets <final-CDM file> --final \
    --repeats 5 --folds 5 --bootstrap 500 --out reports/vigil-lab-sealed \
    --export --export-dir data/models/lab
```

(and the pooled VIGIL-E run from the "Pre-run amendment" run plan)

- The sealed test's numbers are final the first time they are seen:
  `sealed_test` ignores `--repeats`, `--folds` and `--bootstrap`, so a smoke
  run with targets would already produce them. The CLI therefore refuses
  `--test-targets` without `--final`. Each report records the commit it ran
  on, and whether `backend/app/models` had uncommitted changes.

- The full run takes a few hours on a laptop (about 2 h for 5×5 on 4 cores).
  Don't run two lab jobs at once: xgboost threads oversubscribe and
  everything slows about 10×.
- `--test-targets` accepts either full CDM rows or just `event_id` plus a
  risk column (`risk` / `true_risk` / `final_risk` / `target`).
- Read `reports/vigil-lab-sealed/lab_report.md` (VIGIL's verdict) and
  `reports/vigil-lab-pooled/lab_report.md` (VIGIL-E's). Only the **sealed
  test** column is comparable to the 2019 leaderboard, where LRP scores
  ≈ 0.694 and the top entry ≈ 0.556 (Uriot et al. 2022). That entry was a
  rule cascade partly tuned against the test leaderboard.
- `--export` writes models to `data/models/lab/`, and only models that
  passed their gate. VIGIL's export is decided by `serving_resid`'s gate
  on the sealed test, and VIGIL-E's by `e_serving`'s gate. Promoting one
  means copying it over `data/models/`,
  then updating the Model page copy, which still describes vigil-0.4's
  metrics.
- The best serving-track VIGIL may be a hurdle model (`hurdle_lrp_serving`).
  The app's current artifact format holds one residual booster, so serving a
  hurdle would need a small `RiskModel` extension: two boosters, a
  threshold and alpha. The report shows whether that is worth doing.
- `--pool-test-into-cv` pools the official test events into CV. That
  maximises VIGIL-E's positives, but leaves no sealed test.

## Honest limits
- The winner's leaderboard gain partly reflected its particular test set. When the
  organisers re-ran it on 505 random splits, its F2 fell about 14% below the
  baseline's. Expect realistic real-data gains of about 0–20% on L, and trust
  only the CIs.
- Many escalations are caused by future tracking updates that are invisible
  at the 2-day cutoff. The ceiling diagnostic measures how much.
- Public data offers nothing better than Kelvins for CDM sequences with
  covariance. Space-Track's public CDMs lack covariance.
