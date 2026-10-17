# VIGIL Lab report

Code: `034c879b4048`.

Training file: 11942 events, 8892 usable for training, 8293 meeting the official test filters (66 of them high-risk, 0.80%). Early warning is scored on 7985 eligible escalation candidates with 19 escalations (0.24%; 23 escalations in the whole file). Protocol: 5×5 stratified CV, 500 shared-index paired bootstrap replicates, 22809 cutoff-augmentation rows.

Sealed official test: 2167 events, 150 high-risk (6.92%); 1989 escalation candidates with 35 escalations (1.76%). Same eligibility filters as CV, different class mix: the organisers enriched the test set with high-risk events (Uriot et al. 2022, §4.2), so CV and sealed magnitudes are not replicates of each other.

Pre-registered primary candidates: `hurdle_lrp`, `hurdle_lrp_serving`, `e_hurdle`, `e_serving`. Each is tested by the ship gate as a single hypothesis. `serving_resid`, the only VIGIL the app's current artifact format can load, is also gated (it decides `--export`), on its max-T adjusted interval. Every other row is exploratory, and its *adjusted* interval accounts for comparing many candidates at once.

**Data fidelity:** on the sealed test the latest-risk baseline reproduces the published score (lab L 0.694 vs 0.694, MSE_HR 0.513 vs 0.513, F2 0.739 vs 0.739). If it does not, fix the data/target handling before trusting anything below (on synthetic test data this check is expected to fail).

## VIGIL — official score L = MSE_HR / F2 (lower is better)

| Candidate | Track | L (CV mean ± sd) | ΔL vs LRP | 95% CI (%) | adjusted CI (%) | MDE (pp) | MSE_HR | F2 | Sealed L | Sealed ΔL [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| **hurdle_lrp** | benchmark | 0.779 ± 0.013 | -3.1% | [-9.8, 3.9] | [-13.1, 7.0] | 9.5 | 0.328 | 0.420 | 0.694 | +0.1% [-4.6, 4.7] |
| lrp | benchmark | 0.804 ± 0.000 | +0.0% | [0.0, 0.0] | [0.0, 0.0] | — | 0.330 | 0.411 | 0.694 | +0.0% adj [0.0, 0.0] |
| **hurdle_lrp_serving** | serving | 0.807 ± 0.013 | +0.4% | [-2.6, 4.1] | [-4.3, 5.2] | 4.5 | 0.328 | 0.406 | 0.682 | -1.7% [-5.4, 1.8] |
| hurdle | benchmark | 0.876 ± 0.107 | +9.0% | [-8.1, 43.3] | [-27.4, 45.5] | 34.3 | 0.351 | 0.402 | 0.864 | +24.5% adj [-27.3, 76.3] |
| gbm_quant | benchmark | 2.880 ± 0.129 | +258.3% | [119.6, 518.1] | [-44.8, 561.4] | 285.5 | 1.258 | 0.438 | 2.791 | +302.1% adj [75.0, 529.2] |
| gbm_resid_hr | benchmark | 46.557 ± 17.120 | +5692.4% | [2010.5, 11414.8] | [-1368.8, 12753.6] | 6651.4 | 1.242 | 0.030 | 208.186 | +29899.7% adj [-1410.9, 61210.4] |
| gbm_resid | benchmark | 57.372 ± 17.440 | +7038.1% | [1893.1, 11682.7] | [-908.6, 14984.7] | 7485.5 | 1.248 | 0.026 | ∞ | —% — |
| sesc (never predicts high risk) | last | ∞ ± — | —% | — | — | — | 1.299 | 0.000 | 210.420 | +30221.6% adj [-1291.4, 61734.5] |
| gbm_resid_aug | benchmark | ∞ ± — | +6981.7% | [2036.4, 13362.9] | [-1952.7, 15916.1] | 8415.9 | 1.289 | 0.015 | 69.802 | +9958.5% adj [-17582.6, 37499.5] |
| ridge_phys (never predicts high risk) | benchmark | ∞ ± — | —% | — | — | — | 1.299 | 0.000 | ∞ | —% — |
| serving_resid | serving | ∞ ± — | +5613.1% | [1479.1, 10912.9] | [-2270.0, 13496.2] | 7425.6 | 1.287 | 0.015 | 40.980 | +5805.2% adj [-14481.2, 26091.7] |

Sealed ΔL: primaries show their single-test 95% CI; exploratory rows show the max-T multiplicity-adjusted CI ("adj").

CV bootstrap replicates where ΔL is undefined (F2 = 0) are dropped, which flatters those rows: `gbm_resid_hr` 0%, `gbm_resid` 5%, `sesc` 100%, `gbm_resid_aug` 14%, `ridge_phys` 100%, `serving_resid` 6% of replicates.

**VIGIL (benchmark): `hurdle_lrp` does not pass the ship gate on the sealed test.** Minimum detectable effect with this data ≈ 6.539 percentage points.
- ✓ 150 high-risk events in the sealed test (need >= 100)
- ✗ sealed-test ΔL +0.05% (need <= -2.0%)
- ✗ sealed-test 95% CI [-4.59, +4.67]% (need upper < 0)
- (✓ supporting, not blocking) CV mean ΔL -3.09% (need <= -2.0%)
- (✗ supporting, not blocking) CV 95% CI [-9.78, +3.92]% (need upper < 0)

**VIGIL (serving-track features; needs a RiskModel hurdle extension to deploy): `hurdle_lrp_serving` does not pass the ship gate on the sealed test.** Minimum detectable effect with this data ≈ 5.167 percentage points.
- ✓ 150 high-risk events in the sealed test (need >= 100)
- ✗ sealed-test ΔL -1.66% (need <= -2.0%)
- ✗ sealed-test 95% CI [-5.44, +1.78]% (need upper < 0)
- (✗ supporting, not blocking) CV mean ΔL +0.42% (need <= -2.0%)
- (✗ supporting, not blocking) CV 95% CI [-2.64, +4.14]% (need upper < 0)

**The only app-loadable VIGIL result (`serving_resid`, current artifact format): `serving_resid` does not pass the ship gate on the sealed test.** Minimum detectable effect with this data ≈ 25897.791 percentage points.
- ✓ 150 high-risk events in the sealed test (need >= 100)
- ✗ sealed-test ΔL +5805.25% (need <= -2.0%)
- ✗ sealed-test max-T adjusted 95% CI [-14481.17, +26091.67]% (need upper < 0)
- (✗ supporting, not blocking) CV mean ΔL +5613.13% (need <= -2.0%)
- (✗ supporting, not blocking) CV max-T adjusted 95% CI [-2269.95, +13496.21]% (need upper < 0)

## VIGIL-E — early warning (eligible events still below −6 at the cutoff; one decision per event, at the last CDM ≥ 2 days before TCA)

| Candidate | PR-AUC | ROC-AUC | pAUC@FPR≤0.1 | Recall@budget | ΔPR-AUC vs logit | 95% CI | adjusted CI | MDE | Sealed ΔPR-AUC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| **e_hurdle** | 0.1307 ± 0.0453 | 0.932 | 0.741 | 0.526 | +0.0859 | [+0.0050, +0.2167] | [-0.0745, +0.2463] | 0.1604 | +0.0356 [-0.0543, +0.1458] |
| e_quant | 0.1031 ± 0.0141 | 0.890 | 0.773 | 0.621 | +0.0582 | [+0.0030, +0.1960] | [-0.0820, +0.1984] | 0.1402 | -0.0800 adj [-0.1959, +0.0359] |
| e_rank_blend | 0.0663 ± 0.0267 | 0.882 | 0.674 | 0.421 | +0.0214 | [-0.0252, +0.0890] | [-0.0692, +0.1120] | 0.0906 | -0.0777 adj [-0.2231, +0.0678] |
| e_logit | 0.0449 ± 0.0195 | 0.907 | 0.734 | 0.558 | — | — | — | — | — |
| e_physics | 0.0190 ± 0.0000 | 0.744 | 0.656 | 0.421 | -0.0259 | [-0.0651, -0.0062] | [-0.0705, +0.0188] | 0.0447 | -0.0971 adj [-0.2668, +0.0726] |
| e_clf | 0.0165 ± 0.0068 | 0.810 | 0.609 | 0.232 | -0.0284 | [-0.0735, -0.0031] | [-0.0773, +0.0206] | 0.0490 | -0.1123 adj [-0.2658, +0.0412] |
| **e_serving** | 0.0152 ± 0.0039 | 0.819 | 0.628 | 0.346 | -0.0296 | [-0.0773, -0.0052] | [-0.0792, +0.0199] | 0.0495 | -0.0941 [-0.2119, -0.0015] |
| e_max | 0.0125 ± 0.0000 | 0.695 | 0.533 | 0.105 | -0.0324 | [-0.0904, +0.0224] | [-0.1132, +0.0484] | 0.0808 | -0.1669 adj [-0.3155, -0.0184] |

**VIGIL-E (benchmark): `e_hurdle` is underpowered — too little data for a claim either way.** Minimum detectable effect with this data ≈ 0.141.
- ✗ 35 escalations among eligible candidates in the sealed test (need >= 50)
- ✗ sealed-test ΔPR-AUC 95% CI [-0.0543, +0.1458] (need lower > 0)
- ✓ sealed-test Δrecall@budget 95% CI [-0.069, +0.200] (need upper >= 0)
- ✓ CV calibration slope 1.14 (need 0.6–1.6)
- (✓ supporting, not blocking) CV ΔPR-AUC 95% CI [+0.0050, +0.2167] (need lower > 0)
- (✓ supporting, not blocking) CV Δrecall@budget 95% CI [-0.127, +0.070] (need upper >= 0)

**VIGIL-E (serving track): `e_serving` is underpowered — too little data for a claim either way.** Minimum detectable effect with this data ≈ 0.153.
- ✗ 35 escalations among eligible candidates in the sealed test (need >= 50)
- ✗ sealed-test ΔPR-AUC 95% CI [-0.2119, -0.0015] (need lower > 0)
- ✓ sealed-test Δrecall@budget 95% CI [-0.293, +0.030] (need upper >= 0)
- ✓ CV calibration slope 0.84 (need 0.6–1.6)
- (✗ supporting, not blocking) CV ΔPR-AUC 95% CI [-0.0773, -0.0052] (need lower > 0)
- (✗ supporting, not blocking) CV Δrecall@budget 95% CI [-0.361, -0.058] (need upper >= 0)

## Ceiling diagnostic (not admissible)

The primary model, allowed to also see CDMs between 2 and 1 day before TCA:
- admissible: model L 0.772 vs latest-risk 0.804
- with_late_cdms: model L 0.398 vs latest-risk 0.387
The gap between the two rows is the part of the outcome decided by late tracking updates that no admissible model can see.

## How to read this

- Only the **sealed test** columns are comparable to the 2019 leaderboard (published: latest-risk baseline L ≈ 0.694; the top entry, ≈ 0.556, was a hand-tuned rule cascade partly fitted by trial against the test leaderboard — Uriot et al. 2022, §5.3.1).
- `sesc` here is the organisers' post-competition LightGBM control (Uriot et al. 2022, §6.1, without its quantile-transformed target), not team sesc's winning entry.
- CV rows use test-like events (same filters, different class mix); the sealed test can judge a pre-registered primary but never select one.
- `serving` rows use only what the live app can compute, for predictions made ≥ 2 days before TCA; those are the only numbers the app may claim, and only for that decision point. `hurdle_lrp_serving` cannot be loaded by the current RiskModel.
- MDE = minimum detectable effect (80% power). A difference smaller than the MDE can be real but is invisible to this much data.
- CV intervals hold the fitted out-of-fold predictions fixed, so they leave out refit variance (Bates et al. 2023): read them as lower bounds on the uncertainty.
