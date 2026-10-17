# VIGIL competition entry (2019 Kelvins rules)

Built the way the 2019 leaderboard was: models trained on the training file only, tuned with score-only feedback from a PUBLIC half of the official test (60 of 60 allowed submissions, every one in `ledger.jsonl`), then scored once on the untouched PRIVATE half and on the full test. A benchmark entry tuned to the official test's distribution, like the 2019 entries — not evidence about live performance.

| | Public half | Private half (scored once) | Full test (leaderboard) |
|---|---|---|---|
| VIGIL entry | 0.542 | 0.670 | **0.605** |
| Latest-risk baseline | — | 0.711 | 0.694 |
| sesc's published cascade (reference) | — | 0.634 | 0.569 |

Full test: L 0.6045 (MSE_HR 0.443, F2 0.733), -12.9% vs the baseline — **rank 4 of 97** on the final 2019 leaderboard (1st: sesc 0.555).

Private half, never used for feedback: L 0.6698 (-5.8% vs the baseline on the same events).

Training file, out-of-fold, at the test's class mix: L 0.3754.

Entry configuration:
```
{
 "w": 0.5,
 "tau": 0.3,
 "band": -7.3,
 "c": -5.6,
 "alpha": 0.5,
 "cap": null,
 "tau_drop": 0.01,
 "o_payload": true,
 "o_tspan": true,
 "o_miss": false
}
```
Code: `5e4a9e6ad5432e0c35d774d5236f4d2a0cab7935`.

