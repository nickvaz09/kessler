# VIGIL Lab report

Code: `280ce55c212c`.

Training file: 11942 events, 8892 usable for training, 8293 meeting the official test filters (66 of them high-risk, 0.80%). Early warning is scored on 7985 eligible escalation candidates with 19 escalations (0.24%; 23 escalations in the whole file). Protocol: 5×5 stratified CV, 1000 shared-index paired bootstrap replicates, 22809 cutoff-augmentation rows.

Sealed official test: 2167 events, 150 high-risk (6.92%); 1989 escalation candidates with 35 escalations (1.76%). Same eligibility filters as CV, different class mix: the organisers enriched the test set with high-risk events (Uriot et al. 2022, §4.2), so CV and sealed magnitudes are not replicates of each other.

Pre-registered primary candidates: `hurdle_lrp_serving_bench`, `e_hurdle_eserving`. Each is tested by the ship gate as a single hypothesis. `serving_resid`, the only VIGIL the app's current artifact format can load, is also gated (it decides `--export`), on its max-T adjusted interval. Every other row is exploratory, and its *adjusted* interval accounts for comparing many candidates at once.

**Round 2** (pre-registered 2026-10-08, docs/v2/vigil-lab.md): a second look at data Round 1 already judged, so the primaries must clear a **97.5%** interval (half of Round 1's error rate). Round 1's primaries are not re-tested.

**Data fidelity:** on the sealed test the latest-risk baseline reproduces the published score (lab L 0.694 vs 0.694, MSE_HR 0.513 vs 0.513, F2 0.739 vs 0.739). If it does not, fix the data/target handling before trusting anything below (on synthetic test data this check is expected to fail).

## VIGIL — official score L = MSE_HR / F2 (lower is better)

| Candidate | Track | L (CV mean ± sd) | ΔL vs LRP | 95% CI (%) | adjusted CI (%) | MDE (pp) | MSE_HR | F2 | CV ΔL at test mix | Sealed L | Sealed ΔL [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hurdle_lrp | benchmark | 0.779 ± 0.013 | -3.1% | [-9.6, 3.5] | [-13.5, 7.3] | 9.2 | 0.328 | 0.420 | -4.6% | 0.694 | +0.1% adj [-7.1, 7.2] |
| lrp | benchmark | 0.804 ± 0.000 | +0.0% | [0.0, 0.0] | [0.0, 0.0] | — | 0.330 | 0.411 | +0.0% | 0.694 | +0.0% adj [0.0, 0.0] |
| hurdle_lrp_serving | serving | 0.807 ± 0.013 | +0.4% | [-2.5, 3.9] | [-4.5, 5.3] | 4.4 | 0.328 | 0.406 | -1.2% | 0.682 | -1.7% adj [-7.3, 4.0] |
| hurdle | benchmark | 0.876 ± 0.107 | +9.0% | [-8.8, 39.6] | [-27.9, 45.9] | 32.9 | 0.351 | 0.402 | +10.0% | 0.864 | +24.5% adj [-27.6, 76.6] |
| **hurdle_lrp_serving_bench** | serving | 0.971 ± 0.056 | +20.8% | [2.8, 41.4] | [-10.3, 52.0] | 27.7 | 0.305 | 0.314 | -11.0% | 0.661 | -4.7% 97.5% [-14.0, 3.6] |
| hurdle_eserving_aux | serving | 1.027 ± 0.184 | +27.8% | [-3.6, 80.9] | [-39.6, 95.3] | 60.1 | 0.310 | 0.300 | -8.6% | 0.671 | -3.3% adj [-20.6, 14.1] |
| hurdle_eserving | serving | 1.046 ± 0.233 | +30.2% | [-5.3, 98.2] | [-54.7, 115.1] | 75.6 | 0.379 | 0.362 | +11.0% | 0.649 | -6.5% adj [-21.6, 8.6] |
| gbm_quant | benchmark | 2.880 ± 0.129 | +258.3% | [115.2, 515.6] | [-63.9, 580.5] | 287.0 | 1.258 | 0.438 | +314.4% | 2.791 | +302.1% adj [74.0, 530.3] |
| gbm_resid_hr | benchmark | 46.557 ± 17.120 | +5692.4% | [2086.1, 10750.5] | [-1368.1, 12753.0] | 6288.8 | 1.242 | 0.030 | +9650.4% | 208.186 | +29899.7% adj [-1556.3, 61355.8] |
| gbm_resid | benchmark | 57.372 ± 17.440 | +7038.1% | [1859.3, 11761.6] | [-1189.7, 15265.8] | 7328.5 | 1.248 | 0.026 | +11916.8% | ∞ | —% — |
| sesc (never predicts high risk) | last | ∞ ± — | —% | — | — | — | 1.299 | 0.000 | —% | 210.420 | +30221.6% adj [-1437.7, 61880.8] |
| gbm_resid_aug | benchmark | ∞ ± — | +6981.7% | [2002.1, 12627.3] | [-1901.6, 15865.0] | 7912.4 | 1.289 | 0.015 | +11819.7% | 69.802 | +9958.5% adj [-17710.4, 37627.4] |
| ridge_phys (never predicts high risk) | benchmark | ∞ ± — | —% | — | — | — | 1.299 | 0.000 | —% | ∞ | —% — |
| serving_resid | serving | ∞ ± — | +5613.1% | [1248.8, 11836.5] | [-2849.0, 14075.2] | 7537.2 | 1.287 | 0.015 | +9368.8% | 40.980 | +5805.2% adj [-14575.3, 26185.8] |

Sealed ΔL: primaries show the 97.5% CI their gate judges; exploratory rows show the max-T multiplicity-adjusted 95% CI ("adj").

CV ΔL at test mix: the same out-of-fold predictions scored with label-shift weights that give the official test's published class mix (150 of 2,167 high-risk; Uriot et al. 2022, §4.2) — what CV says about a sealed look. Point estimate, mean over repeats.

CV bootstrap replicates where ΔL is undefined (F2 = 0) are dropped, which flatters those rows: `gbm_resid_hr` 0%, `gbm_resid` 4%, `sesc` 100%, `gbm_resid_aug` 13%, `ridge_phys` 100%, `serving_resid` 6% of replicates.

**VIGIL leaderboard entry (serving features; decision set for the official test's published class mix — a scoring adjustment for the benchmark, not a live-app improvement): `hurdle_lrp_serving_bench` does not pass the ship gate on the sealed test.** Minimum detectable effect with this data ≈ 12.575 percentage points.
- ✓ 150 high-risk events in the sealed test (need >= 100)
- ✓ sealed-test ΔL -4.73% (need <= -2.0%)
- ✗ sealed-test 97.5% CI [-14.02, +3.58]% (need upper < 0)
- (✗ supporting, not blocking) CV mean ΔL +20.83% (need <= -2.0%)
- (✗ supporting, not blocking) CV 97.5% CI [+1.00, +44.41]% (need upper < 0)

**The only app-loadable VIGIL result (`serving_resid`, current artifact format): `serving_resid` does not pass the ship gate on the sealed test.** Minimum detectable effect with this data ≈ 25991.971 percentage points.
- ✓ 150 high-risk events in the sealed test (need >= 100)
- ✗ sealed-test ΔL +5805.25% (need <= -2.0%)
- ✗ sealed-test max-T adjusted 95% CI [-14575.35, +26185.85]% (need upper < 0)
- (✗ supporting, not blocking) CV mean ΔL +5613.13% (need <= -2.0%)
- (✗ supporting, not blocking) CV max-T adjusted 95% CI [-2848.97, +14075.22]% (need upper < 0)

## VIGIL-E — early warning (eligible events still below −6 at the cutoff; one decision per event, at the last CDM ≥ 2 days before TCA)

| Candidate | PR-AUC | ROC-AUC | pAUC@FPR≤0.1 | Recall@budget | ΔPR-AUC vs logit | 95% CI | adjusted CI | MDE | Sealed ΔPR-AUC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| e_hurdle | 0.1307 ± 0.0453 | 0.932 | 0.741 | 0.526 | +0.0859 | [+0.0027, +0.2166] | [-0.0847, +0.2564] | 0.1566 | +0.0356 adj [-0.0925, +0.1636] |
| e_quant | 0.1031 ± 0.0141 | 0.890 | 0.773 | 0.621 | +0.0582 | [+0.0016, +0.1880] | [-0.0912, +0.2076] | 0.1372 | -0.0800 adj [-0.1992, +0.0391] |
| e_rank_blend | 0.0663 ± 0.0267 | 0.882 | 0.674 | 0.421 | +0.0214 | [-0.0288, +0.0931] | [-0.0763, +0.1192] | 0.0897 | -0.0777 adj [-0.2272, +0.0719] |
| e_logit | 0.0449 ± 0.0195 | 0.907 | 0.734 | 0.558 | — | — | — | — | — |
| e_hurdle_eserving_aux | 0.0411 ± 0.0115 | 0.918 | 0.760 | 0.600 | -0.0037 | [-0.0396, +0.0437] | [-0.0663, +0.0589] | 0.0575 | -0.0796 adj [-0.1928, +0.0337] |
| **e_hurdle_eserving** | 0.0394 ± 0.0076 | 0.907 | 0.730 | 0.537 | -0.0054 | [-0.0290, +0.0314] | [-0.0510, +0.0402] | 0.0419 | -0.0100 97.5% [-0.1183, +0.0879] |
| e_physics | 0.0190 ± 0.0000 | 0.744 | 0.656 | 0.421 | -0.0259 | [-0.0618, -0.0049] | [-0.0724, +0.0206] | 0.0427 | -0.0971 adj [-0.2715, +0.0773] |
| e_clf | 0.0165 ± 0.0068 | 0.810 | 0.609 | 0.232 | -0.0284 | [-0.0685, -0.0022] | [-0.0802, +0.0235] | 0.0476 | -0.1123 adj [-0.2700, +0.0454] |
| e_serving | 0.0152 ± 0.0039 | 0.819 | 0.628 | 0.346 | -0.0296 | [-0.0713, -0.0051] | [-0.0819, +0.0226] | 0.0479 | -0.0941 adj [-0.2329, +0.0447] |
| e_max | 0.0125 ± 0.0000 | 0.695 | 0.533 | 0.105 | -0.0324 | [-0.0861, +0.0288] | [-0.1202, +0.0554] | 0.0806 | -0.1669 adj [-0.3196, -0.0143] |

**VIGIL-E (serving track: features the app computes): `e_hurdle_eserving` is underpowered — too little data for a claim either way.** Minimum detectable effect with this data ≈ 0.137.
- ✗ 35 escalations among eligible candidates in the sealed test (need >= 50)
- ✗ sealed-test ΔPR-AUC 97.5% CI [-0.1183, +0.0879] (need lower > 0)
- ✓ sealed-test Δrecall@budget 95% CI [-0.194, +0.100] (need upper >= 0)
- ✓ CV calibration slope 1.09 (need 0.6–1.6)
- (✗ supporting, not blocking) CV ΔPR-AUC 97.5% CI [-0.0319, +0.0408] (need lower > 0)
- (✓ supporting, not blocking) CV Δrecall@budget 95% CI [-0.063, +0.048] (need upper >= 0)

## How to read this

- Only the **sealed test** columns are comparable to the 2019 leaderboard (published: latest-risk baseline L ≈ 0.694; the top entry, ≈ 0.556, was a hand-tuned rule cascade partly fitted by trial against the test leaderboard — Uriot et al. 2022, §5.3.1).
- `sesc` here is the organisers' post-competition LightGBM control (Uriot et al. 2022, §6.1, without its quantile-transformed target), not team sesc's winning entry.
- CV rows use test-like events (same filters, different class mix); the sealed test can judge a pre-registered primary but never select one.
- `serving` rows use only what the live app can compute, for predictions made ≥ 2 days before TCA; those are the only numbers the app may claim, and only for that decision point. `hurdle_lrp_serving` cannot be loaded by the current RiskModel.
- MDE = minimum detectable effect (80% power). A difference smaller than the MDE can be real but is invisible to this much data.
- CV intervals hold the fitted out-of-fold predictions fixed, so they leave out refit variance (Bates et al. 2023): read them as lower bounds on the uncertainty.
