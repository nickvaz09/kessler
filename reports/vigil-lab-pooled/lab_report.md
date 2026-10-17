# VIGIL Lab report

> **NO SEALED TEST** — the 2167 official test events were pooled into cross-validation. Verdicts are CV-only, not comparable to the 2019 leaderboard, and the data-fidelity check did not run.

> Pre-declared role (docs/v2/vigil-lab.md, "Pre-run amendment"): this run decides **VIGIL-E only**. Its VIGIL rows and verdicts are exploratory; VIGIL's verdict comes from the sealed run.

Code: `02428e7f0e13`.

CV pool (training file + 2167 official test events): 14109 events, 11059 usable for training, 10460 meeting the official test filters (216 of them high-risk, 2.07%). Early warning is scored on 9974 eligible escalation candidates with 54 escalations (0.54%; 58 escalations in the whole file). Protocol: 5×5 stratified CV, 500 shared-index paired bootstrap replicates, 28505 cutoff-augmentation rows.

Pre-registered primary candidates: `hurdle_lrp`, `hurdle_lrp_serving`, `e_hurdle`, `e_serving`. Each is tested by the ship gate as a single hypothesis. `serving_resid`, the only VIGIL the app's current artifact format can load, is also gated (it decides `--export`), on its max-T adjusted interval. Every other row is exploratory, and its *adjusted* interval accounts for comparing many candidates at once.

## VIGIL — official score L = MSE_HR / F2 (lower is better)

| Candidate | Track | L (CV mean ± sd) | ΔL vs LRP | 95% CI (%) | adjusted CI (%) | MDE (pp) | MSE_HR | F2 |
|---|---|---|---|---|---|---|---|---|
| **hurdle_lrp_serving** | serving | 0.761 ± 0.013 | -0.1% | [-3.6, 3.7] | [-5.6, 5.4] | 5.4 | 0.453 | 0.595 |
| lrp | benchmark | 0.762 ± 0.000 | +0.0% | [0.0, 0.0] | [0.0, 0.0] | — | 0.457 | 0.600 |
| hurdle | benchmark | 0.769 ± 0.013 | +0.9% | [-7.0, 9.7] | [-11.2, 13.0] | 11.8 | 0.463 | 0.603 |
| **hurdle_lrp** | benchmark | 0.770 ± 0.005 | +1.0% | [-1.7, 4.0] | [-3.4, 5.5] | 4.3 | 0.461 | 0.599 |
| gbm_quant | benchmark | 2.484 ± 0.041 | +226.1% | [139.7, 340.1] | [79.7, 372.5] | 143.4 | 1.493 | 0.601 |
| serving_resid | serving | 9.366 ± 0.721 | +1129.5% | [712.3, 1861.1] | [293.0, 1966.1] | 819.0 | 1.383 | 0.148 |
| gbm_resid_hr | benchmark | 17.891 ± 1.591 | +2248.5% | [1412.1, 3886.3] | [436.5, 4060.6] | 1774.1 | 1.507 | 0.085 |
| gbm_resid | benchmark | 19.970 ± 3.022 | +2521.4% | [1640.8, 4609.4] | [267.1, 4775.7] | 2207.1 | 1.511 | 0.077 |
| gbm_resid_aug | benchmark | 21.588 ± 4.807 | +2733.9% | [1792.9, 5098.6] | [69.3, 5398.5] | 2608.8 | 1.524 | 0.074 |
| sesc | last | ∞ ± — | +25840.0% | [7731.0, 44308.1] | [-3204.9, 54885.0] | 28437.1 | 1.610 | 0.008 |
| ridge_phys (never predicts high risk) | benchmark | ∞ ± — | —% | — | — | — | 1.613 | 0.000 |

CV bootstrap replicates where ΔL is undefined (F2 = 0) are dropped, which flatters those rows: `sesc` 2%, `ridge_phys` 100% of replicates.

**VIGIL (benchmark): `hurdle_lrp` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 4.345 percentage points.
- ✓ 216 high-risk eligible events in CV (need >= 100)
- ✗ CV mean ΔL +1.01% (need <= -2.0%)
- ✗ CV 95% CI [-1.71, +4.04]% (need upper < 0)

**VIGIL (serving-track features; needs a RiskModel hurdle extension to deploy): `hurdle_lrp_serving` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 5.419 percentage points.
- ✓ 216 high-risk eligible events in CV (need >= 100)
- ✗ CV mean ΔL -0.11% (need <= -2.0%)
- ✗ CV 95% CI [-3.57, +3.67]% (need upper < 0)

**The only app-loadable VIGIL result (`serving_resid`, current artifact format): `serving_resid` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 1082.713 percentage points.
- ✓ 216 high-risk eligible events in CV (need >= 100)
- ✗ CV mean ΔL +1129.53% (need <= -2.0%)
- ✗ CV max-T adjusted 95% CI [+293.00, +1966.07]% (need upper < 0)

## VIGIL-E — early warning (eligible events still below −6 at the cutoff; one decision per event, at the last CDM ≥ 2 days before TCA)

| Candidate | PR-AUC | ROC-AUC | pAUC@FPR≤0.1 | Recall@budget | ΔPR-AUC vs logit | 95% CI | adjusted CI | MDE |
|---|---|---|---|---|---|---|---|---|
| **e_hurdle** | 0.1261 ± 0.0165 | 0.920 | 0.742 | 0.519 | +0.0611 | [+0.0141, +0.1281] | [-0.0205, +0.1427] | 0.0849 |
| e_quant | 0.0802 ± 0.0060 | 0.821 | 0.737 | 0.511 | +0.0152 | [-0.0149, +0.0604] | [-0.0348, +0.0652] | 0.0520 |
| e_logit | 0.0650 ± 0.0156 | 0.853 | 0.720 | 0.515 | — | — | — | — |
| **e_serving** | 0.0518 ± 0.0087 | 0.881 | 0.688 | 0.441 | -0.0131 | [-0.0366, +0.0029] | [-0.0413, +0.0150] | 0.0293 |
| e_rank_blend | 0.0482 ± 0.0108 | 0.882 | 0.689 | 0.441 | -0.0168 | [-0.0462, +0.0046] | [-0.0515, +0.0179] | 0.0361 |
| e_clf | 0.0378 ± 0.0047 | 0.881 | 0.661 | 0.363 | -0.0272 | [-0.0595, -0.0067] | [-0.0635, +0.0092] | 0.0378 |
| e_physics | 0.0364 ± 0.0000 | 0.733 | 0.631 | 0.333 | -0.0286 | [-0.0706, +0.0235] | [-0.0911, +0.0339] | 0.0651 |
| e_max | 0.0122 ± 0.0000 | 0.658 | 0.518 | 0.093 | -0.0528 | [-0.0949, -0.0246] | [-0.1005, -0.0050] | 0.0497 |

**VIGIL-E (benchmark): `e_hurdle` passes the CV gate (no sealed test, so no veto was possible).** Minimum detectable effect with this data ≈ 0.085.
- ✓ 54 escalations among eligible candidates in CV (need >= 50)
- ✓ CV ΔPR-AUC 95% CI [+0.0141, +0.1281] (need lower > 0)
- ✓ CV Δrecall@budget 95% CI [-0.086, +0.094] (need upper >= 0)
- ✓ CV calibration slope 1.07 (need 0.6–1.6)

**VIGIL-E (serving track): `e_serving` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 0.029.
- ✓ 54 escalations among eligible candidates in CV (need >= 50)
- ✗ CV ΔPR-AUC 95% CI [-0.0366, +0.0029] (need lower > 0)
- ✗ CV Δrecall@budget 95% CI [-0.134, -0.013] (need upper >= 0)
- ✓ CV calibration slope 0.99 (need 0.6–1.6)

## How to read this

- Only the **sealed test** columns are comparable to the 2019 leaderboard (published: latest-risk baseline L ≈ 0.694; the top entry, ≈ 0.556, was a hand-tuned rule cascade partly fitted by trial against the test leaderboard — Uriot et al. 2022, §5.3.1).
- `sesc` here is the organisers' post-competition LightGBM control (Uriot et al. 2022, §6.1, without its quantile-transformed target), not team sesc's winning entry.
- CV rows use test-like events (same filters, different class mix); the sealed test can judge a pre-registered primary but never select one.
- `serving` rows use only what the live app can compute, for predictions made ≥ 2 days before TCA; those are the only numbers the app may claim, and only for that decision point. `hurdle_lrp_serving` cannot be loaded by the current RiskModel.
- MDE = minimum detectable effect (80% power). A difference smaller than the MDE can be real but is invisible to this much data.
- CV intervals hold the fitted out-of-fold predictions fixed, so they leave out refit variance (Bates et al. 2023): read them as lower bounds on the uncertainty.
