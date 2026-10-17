# One-click avoidance plan, simulator and assistant (v2)

## `POST /conjunctions/{id}/avoidance-plan`

Body (optional, JSON): `{"target_miss_m": number?, "max_dv_ms": number?}`

- `target_miss_m`: 0 < x <= 100000, default **1000 m**.
- `max_dv_ms`: 0 < x <= 5, default 5 (the maneuver sandbox bound).

Errors: 404 unknown id; **409** when closest approach has passed, or when every
candidate burn time is already past or less than 5 minutes away; 422 for bad
input or an unusable TLE.

### Method

Reuses the what-if maneuver sandbox (`api/conjunctions.load_maneuver_context`
+ `evaluate_maneuver`, two-body differential post-burn model, assumed-sigma Pc).
Candidate burns sit **1 orbit** (primary TLE mean motion), **3 h, 6 h and
12 h** before the stored TCA; only burns still >= 5 min in the future are
searched, the rest are listed in `skipped`. For each, a per-sign golden-section
search (`models/maneuver.min_dv_for_miss`, 0.001 m/s tolerance) finds the
smallest along-track dv (prograde or retrograde) whose post-burn miss reaches
the **required miss**:

    required_miss_m = max(target_miss_m, pc_required_miss_m)

where `pc_required_miss_m` is the miss at which the assumed-sigma Pc drops
below 95 % of `config.PC_ELEVATED` (1e-6), found by bisection on the Pc model.
So a "safe" option clears both the distance target and the elevated threshold;
the 5 % margin absorbs the sub-second post-burn TCA shift.

**Recommendation rule:** the feasible option with the smallest |dv|; options
within 0.01 m/s of it tie, and the tie goes to the later burn (less
extrapolation, more time for fresh tracking).

### Response

```json
{
  "conjunction_id": 5756,
  "tca": "2026-09-24T12:53:11Z",
  "computed_at": "2026-09-24T01:15:03Z",
  "status": "burn_recommended | no_burn_needed | unreachable | geometry_mismatch",
  "note": "string | null",
  "baseline": {"miss_m": 284.2, "pc": 9.25e-5, "tca": "...", "safe": false,
               "screened_miss_m": 284.2},
  "criteria": {"target_miss_m": 1000.0, "pc_threshold": 1e-6,
               "pc_required_miss_m": 2155.3, "required_miss_m": 2155.3,
               "max_dv_ms": 5.0, "rule": "..."},
  "recommended": { "...one of options..." },
  "options": [{
    "lead_label": "6 h", "lead_s": 21600.0, "burn_iso": "2026-09-24T06:53:11Z",
    "dv_ms": 0.0429, "direction": "prograde", "new_miss_m": 2190.7,
    "delta_miss_m": 1906.5, "new_tca": "...", "new_pc": 8.1e-7,
    "reachable": true, "meets_miss": true, "meets_pc": true,
    "propellant_g_per_100kg": 1.99, "evaluations": 24
  }],
  "skipped": [{"lead_label": "12 h", "lead_s": 43200.0, "burn_iso": "...", "reason": "..."}],
  "assumptions": {"isp_s": 220.0, "reference_mass_kg": 100.0,
                  "primary_period_s": 5709.3, "pc_model": "..."},
  "method": "two-body differential, along-track, screening-grade; not an operational maneuver plan",
  "method_detail": "<the maneuver endpoint's method string>"
}
```

- `geometry_mismatch`: the latest-TLE geometry already clears the criteria but
  the latest screening message's miss does not - never reported as an
  all-clear.
- `propellant_g_per_100kg`: Tsiolkovsky at Isp 220 s (hydrazine class) per
  100 kg of spacecraft - fuel context, labelled as an assumption.
- Applying `recommended.dv_ms` at `recommended.burn_iso` through
  `POST /conjunctions/{id}/maneuver` reproduces `new_miss_m` (tested).

## Assistant

- Data tool `plan_avoidance(conjunction_id, target_miss_m?)` returns the plan
  trimmed for the model (no search bookkeeping).
- UI tool `open_simulator` gains `apply_plan: boolean` (upcoming events only);
  the action `{type: "open_simulator", conjunction_id, apply_plan: true}` makes
  the browser call `KSim.open({conjunctionId, applyPlan: true})`.

## Simulator (frontend)

- `KSim.open({conjunctionId, applyPlan?})`, `KSim.planAvoidance({apply?})`
  (resolves to the plan payload), `KSim.close()`.
- AUTO zoom scales by the in-plane (encounter-plane) separation.
- Space camera: right-handed basis, camera on the hemisphere the Earth texture
  paints (the v11 view was mirrored and showed the antipodal hemisphere).
- `KSim.open` is safe right after page load: before `DOMContentLoaded` the
  request is queued (last call wins). A script that runs before
  `simulator.js` (e.g. `dashboard.js` handling `/?sim=<id>`) can instead set
  `window.KSIM_PENDING = {conjunctionId, applyPlan?}`; it opens as soon as the
  simulator loads. While open, `body.ksim-open` is set (the assistant
  launcher hides).
- Narrate/Voice need Kessler AI: a `503` from `/assistant/narrate` (or the
  assistant's shared `sessionStorage['kessler.ai.nokey'] = '1'`) disables both
  with the tooltip "Needs Kessler AI to be set up". User-facing no-key copy is
  one sentence everywhere: "Kessler AI isn't set up on this server yet."

## Geometry check on the ephemeris (QA A2)

`GET /conjunctions/{id}/ephemeris` adds two fields:

- `latest_message_miss_m` -- screened miss of the latest message (0.1 m), or
  `null` without a message.
- `geometry_consistent` -- `false` when the latest-TLE miss (`tca_state.sep_m`)
  already clears the plan's required miss while the latest message does not:
  the same rule that makes the avoidance plan return `geometry_mismatch`
  (required = max(1 km default target, miss where the assumed-sigma Pc drops
  below `PC_ELEVATED × 0.95`)). `true` otherwise; `null` when either miss is
  unknown. The simulator shows a calm notice at open when it is `false`.
