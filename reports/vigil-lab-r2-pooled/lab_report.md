# VIGIL Lab report

> **NO SEALED TEST** — the 2167 official test events were pooled into cross-validation. Verdicts are CV-only, not comparable to the 2019 leaderboard, and the data-fidelity check did not run.

> Pre-declared role (docs/v2/vigil-lab.md, "Pre-run amendment"): this run decides **VIGIL-E only**. Its VIGIL rows and verdicts are exploratory; VIGIL's verdict comes from the sealed run.

Code: `ad7074930724`.

CV pool (training file + 2167 official test events): 14109 events, 11059 usable for training, 10460 meeting the official test filters (216 of them high-risk, 2.07%). Early warning is scored on 9974 eligible escalation candidates with 54 escalations (0.54%; 58 escalations in the whole file). Protocol: 5×5 stratified CV, 1000 shared-index paired bootstrap replicates, 28505 cutoff-augmentation rows.

Pre-registered primary candidates: `hurdle_lrp_serving_bench`, `e_hurdle_eserving`. Each is tested by the ship gate as a single hypothesis. `serving_resid`, the only VIGIL the app's current artifact format can load, is also gated (it decides `--export`), on its max-T adjusted interval. Every other row is exploratory, and its *adjusted* interval accounts for comparing many candidates at once.

**Round 2** (pre-registered 2026-10-08, docs/v2/vigil-lab.md): a second look at data Round 1 already judged, so the primaries must clear a **97.5%** interval (half of Round 1's error rate). Round 1's primaries are not re-tested.

## VIGIL — official score L = MSE_HR / F2 (lower is better)

| Candidate | Track | L (CV mean ± sd) | ΔL vs LRP | 95% CI (%) | adjusted CI (%) | MDE (pp) | MSE_HR | F2 | CV ΔL at test mix |
|---|---|---|---|---|---|---|---|---|---|
| hurdle_lrp_serving | serving | 0.761 ± 0.013 | -0.1% | [-3.7, 3.8] | [-5.8, 5.6] | 5.5 | 0.453 | 0.595 | -3.5% |
| lrp | benchmark | 0.762 ± 0.000 | +0.0% | [0.0, 0.0] | [0.0, 0.0] | — | 0.457 | 0.600 | +0.0% |
| hurdle | benchmark | 0.769 ± 0.013 | +0.9% | [-6.4, 9.6] | [-11.2, 13.0] | 11.5 | 0.463 | 0.603 | -2.9% |
| hurdle_lrp | benchmark | 0.770 ± 0.005 | +1.0% | [-1.8, 4.0] | [-3.4, 5.4] | 4.2 | 0.461 | 0.599 | -1.8% |
| hurdle_eserving | serving | 0.795 ± 0.022 | +4.4% | [-4.0, 13.7] | [-9.1, 17.9] | 12.8 | 0.455 | 0.572 | -3.3% |
| hurdle_eserving_aux | serving | 0.818 ± 0.011 | +7.4% | [-3.4, 19.2] | [-9.3, 24.0] | 15.8 | 0.453 | 0.554 | -2.9% |
| **hurdle_lrp_serving_bench** | serving | 0.840 ± 0.022 | +10.2% | [1.4, 19.7] | [-3.4, 23.8] | 12.9 | 0.435 | 0.519 | -5.8% |
| gbm_quant | benchmark | 2.484 ± 0.041 | +226.1% | [141.9, 342.1] | [76.0, 376.2] | 142.7 | 1.493 | 0.601 | +233.6% |
| serving_resid | serving | 9.366 ± 0.721 | +1129.5% | [713.4, 1842.5] | [280.3, 1978.8] | 807.3 | 1.383 | 0.148 | +1362.2% |
| gbm_resid_hr | benchmark | 17.891 ± 1.591 | +2248.5% | [1419.4, 3948.6] | [321.9, 4175.2] | 1831.5 | 1.507 | 0.085 | +2725.2% |
| gbm_resid | benchmark | 19.970 ± 3.022 | +2521.4% | [1616.2, 4583.3] | [238.5, 4804.3] | 2170.2 | 1.511 | 0.077 | +3055.9% |
| gbm_resid_aug | benchmark | 21.588 ± 4.807 | +2733.9% | [1781.7, 5158.7] | [-178.2, 5646.0] | 2768.3 | 1.524 | 0.074 | +3310.5% |
| sesc | last | ∞ ± — | +25840.0% | [7377.1, 44403.3] | [-3685.1, 55365.2] | 28067.2 | 1.610 | 0.008 | +31202.4% |
| ridge_phys (never predicts high risk) | benchmark | ∞ ± — | —% | — | — | — | 1.613 | 0.000 | —% |

CV ΔL at test mix: the same out-of-fold predictions scored with label-shift weights that give the official test's published class mix (150 of 2,167 high-risk; Uriot et al. 2022, §4.2) — what CV says about a sealed look. Point estimate, mean over repeats.

CV bootstrap replicates where ΔL is undefined (F2 = 0) are dropped, which flatters those rows: `sesc` 2%, `ridge_phys` 100% of replicates.

**VIGIL leaderboard entry (serving features; decision set for the official test's published class mix — a scoring adjustment for the benchmark, not a live-app improvement): `hurdle_lrp_serving_bench` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 14.251 percentage points.
- ✓ 216 high-risk eligible events in CV (need >= 100)
- ✗ CV mean ΔL +10.23% (need <= -2.0%)
- ✗ CV 97.5% CI [+0.10, +21.21]% (need upper < 0)

**The only app-loadable VIGIL result (`serving_resid`, current artifact format): `serving_resid` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 1091.901 percentage points.
- ✓ 216 high-risk eligible events in CV (need >= 100)
- ✗ CV mean ΔL +1129.53% (need <= -2.0%)
- ✗ CV max-T adjusted 95% CI [+280.29, +1978.78]% (need upper < 0)

## VIGIL-E — early warning (eligible events still below −6 at the cutoff; one decision per event, at the last CDM ≥ 2 days before TCA)

| Candidate | PR-AUC | ROC-AUC | pAUC@FPR≤0.1 | Recall@budget | ΔPR-AUC vs logit | 95% CI | adjusted CI | MDE |
|---|---|---|---|---|---|---|---|---|
| e_hurdle | 0.1261 ± 0.0165 | 0.920 | 0.742 | 0.519 | +0.0611 | [+0.0140, +0.1318] | [-0.0272, +0.1493] | 0.0849 |
| **e_hurdle_eserving** | 0.0868 ± 0.0132 | 0.898 | 0.710 | 0.470 | +0.0218 | [-0.0113, +0.0738] | [-0.0416, +0.0853] | 0.0611 |
| e_quant | 0.0802 ± 0.0060 | 0.821 | 0.737 | 0.511 | +0.0152 | [-0.0159, +0.0625] | [-0.0410, +0.0714] | 0.0541 |
| e_hurdle_eserving_aux | 0.0694 ± 0.0072 | 0.910 | 0.723 | 0.519 | +0.0044 | [-0.0178, +0.0306] | [-0.0324, +0.0412] | 0.0354 |
| e_logit | 0.0650 ± 0.0156 | 0.853 | 0.720 | 0.515 | — | — | — | — |
| e_serving | 0.0518 ± 0.0087 | 0.881 | 0.688 | 0.441 | -0.0131 | [-0.0367, +0.0029] | [-0.0436, +0.0173] | 0.0293 |
| e_rank_blend | 0.0482 ± 0.0108 | 0.882 | 0.689 | 0.441 | -0.0168 | [-0.0469, +0.0063] | [-0.0557, +0.0221] | 0.0374 |
| e_clf | 0.0378 ± 0.0047 | 0.881 | 0.661 | 0.363 | -0.0272 | [-0.0626, -0.0066] | [-0.0675, +0.0131] | 0.0388 |
| e_physics | 0.0364 ± 0.0000 | 0.733 | 0.631 | 0.333 | -0.0286 | [-0.0716, +0.0238] | [-0.0984, +0.0413] | 0.0672 |
| e_max | 0.0122 ± 0.0000 | 0.658 | 0.518 | 0.093 | -0.0528 | [-0.0959, -0.0243] | [-0.1054, -0.0001] | 0.0507 |

**VIGIL-E (serving track: features the app computes): `e_hurdle_eserving` does not pass the CV gate (no sealed test).** Minimum detectable effect with this data ≈ 0.067.
- ✓ 54 escalations among eligible candidates in CV (need >= 50)
- ✗ CV ΔPR-AUC 97.5% CI [-0.0142, +0.0854] (need lower > 0)
- ✓ CV Δrecall@budget 95% CI [-0.129, +0.024] (need upper >= 0)
- ✓ CV calibration slope 1.02 (need 0.6–1.6)

## How to read this

- Only the **sealed test** columns are comparable to the 2019 leaderboard (published: latest-risk baseline L ≈ 0.694; the top entry, ≈ 0.556, was a hand-tuned rule cascade partly fitted by trial against the test leaderboard — Uriot et al. 2022, §5.3.1).
- `sesc` here is the organisers' post-competition LightGBM control (Uriot et al. 2022, §6.1, without its quantile-transformed target), not team sesc's winning entry.
- CV rows use test-like events (same filters, different class mix); the sealed test can judge a pre-registered primary but never select one.
- `serving` rows use only what the live app can compute, for predictions made ≥ 2 days before TCA; those are the only numbers the app may claim, and only for that decision point. `hurdle_lrp_serving` cannot be loaded by the current RiskModel.
- MDE = minimum detectable effect (80% power). A difference smaller than the MDE can be real but is invisible to this much data.
- CV intervals hold the fitted out-of-fold predictions fixed, so they leave out refit variance (Bates et al. 2023): read them as lower bounds on the uncertainty.
