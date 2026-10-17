# Time Machine

`/replay.html` replays the 10 February 2009 collision of Iridium 33
(NORAD 24946) and Cosmos 2251 (NORAD 22675) through Kessler's own pipeline.
Five chapters on one story clock:

| # | Chapter | What the page shows | Where the numbers come from |
|---|---------|---------------------|-----------------------------|
| 1 | 72 hours before | Both satellites where they were at the first replayed screening run, the orbits drawn ahead as the look-ahead runs, then the predicted crossing flagged | `screening.detection`, `screening.history`, `tracks` |
| 2 | The encounter | The camera closes on the crossing; slow-motion closest approach with a live separation readout | `/ephemeris` (SGP4, 1 s grid), `tca_state` |
| 3 | Impact | A brief flash, then the modeled fragments leave the point along both parents' paths | `/debris` (model), `energy`, `counts` |
| 4 | The cascade | Two years in 22 seconds: streams, then rings, then a shell | `/debris` elements; `counts.in_orbit_modeled` at 1 h, 1 d, 1 w, 1 mo, 6 mo, 1 y and 2 y checkpoints; `live_catalog` |
| 5 | Why Kessler exists | One sentence, calls to action (Overview, Protect a satellite) | Kessler and Cour-Palais (1978) |

Controls: play/pause button, a scrubber, chapter buttons, keyboard (space
plays or pauses; left and right arrows change chapter; Home restarts). Deep
links: `?chapter=N` (1-5, optional `&u=0..1` within it) or `?t=<story
seconds>` open paused on that frame. With `prefers-reduced-motion`, playback
becomes a slideshow of one still frame per chapter, and the flash is replaced by a
static marker. The space layer stays dark in both themes; the glass readout,
transport and the "About this replay" notes follow the theme.

## Data honesty

- **Published facts** (time, altitude, region, relative speed, inclinations,
  debris count) live in the scenario file with their source, NASA Orbital Debris
  Quarterly News vol. 13 issue 2 (April 2009), and are shown as the scenario
  file words them ("about", "roughly", "well over").
- **Orbits are reconstructed**, and the page says so in a pill on the
  stage and in the notes: "Reconstructed from published orbital parameters". The
  archived February 2009 element sets sit in Space-Track's `gp_history` class,
  which this build could not reach. `scripts/reconstruct_scenario.py`
  generates TLE-format element sets whose SGP4 tracks pass within a few metres
  of a single encounter point at 16:56:00 UTC, 789 km altitude, longitude
  97.9° E. Its latitude (72.26° N) is solved so that the planes cross at the
  published ~11.7 km/s, and the inclinations are 86.4° and 74.0°. Eccentricities,
  zero drag terms and epochs are documented reconstruction choices, not
  archived values.
- **Screening is the live code.** `ingestion.screening.screen` gained two
  keyword-only arguments, `start` (pin the window start) and `catalog`
  (in-memory `{norad_id, line1, line2}` rows in place of the DB). Every gate,
  refinement and filter is unchanged. Pc is `screening._compute_pc`
  (`models.pc.compute_pc`) at the propagation age `record_events` uses, and risk
  levels come from `api.conjunctions.risk_level_for`.
- **The debris cloud is a model** and is labeled "illustrative model (NASA
  Standard Breakup Model, simplified)" everywhere it appears. The page never
  animates counts through values the backend did not produce.
- **"Tracked objects crossing this shell today"** is a band-overlap count from
  the live catalog (objects whose perigee-to-apogee range intersects the cloud's
  altitude band). It is labeled as a geometric count, not a risk.

## Replaying with archived TLEs

```sh
SPACETRACK_IDENTITY=... SPACETRACK_PASSWORD=... python scripts/fetch_scenario_tles.py
```

The script logs in with `ingestion.spacetrack.login`, the same session
pattern as the live ingest, and sends one `gp_history` query for both
objects covering the 7 days before the event. It keeps only element sets
with an epoch before the collision and writes
`backend/app/scenarios/iridium_cosmos_2009.archived.json` with
`"provenance": "archived"`. When that file exists the backend prefers it
automatically. Each replayed screening run then uses only the element sets
published before that run, so "first detectable" becomes the real lead time
the archive supports. The parsing lives in `backend/app/scenarios/archive.py`
and is unit-tested with a fixture (no network).

To regenerate the reconstruction: `python scripts/reconstruct_scenario.py`.

## Screening replay

Runs are replayed on an hourly cadence (`screening_window.cadence_hours`)
starting from the first whole hour at which both objects have a published
element set, and no earlier than one window before the event. The first run
that reports the pair is `first_detection` (with the reconstructed sets:
7 Feb 2009 17:00 UTC, 71.9 h ahead, miss 4.1 m, 11.70 km/s, Pc 1.1e-5,
elevated). `history` adds runs 48, 24, 12, 6, 3 and 1 h before the TCA. Pc rises
to critical inside 12 h as the assumed uncertainty shrinks. Results are cached
per process and per scenario content. The first request takes about 2 s;
concurrent requests wait on the same computation.

## Debris model (`backend/app/scenarios/breakup.py`)

- Catastrophic test: E* = 0.5 · m_p · v² / m_t > 40 J/g (here ~42.6 kJ/g).
- Count: N(>Lc) = 0.1 · M^0.75 · Lc^-1.71, with M = 560 + 900 kg (approximate
  public masses) and Lc = 10 cm, which gives **1,211** fragments. The split is
  by mass (747 Cosmos, 464 Iridium).
- Sizes follow the power-law inverse CDF, capped at 2 m. Area-to-mass uses the
  SBM spacecraft-fragment bimodal distribution for L > 11 cm. Ejection speed:
  log10 Δv ~ N(0.9χ + 2.9, 0.4), in an isotropic direction.
- Propagation: osculating Kepler elements at TCA with J2 secular rates for
  RAAN, argument of perigee and mean anomaly. Drag uses an analytic
  exponential-atmosphere decay, h(t) = h0 + H ln(1 - t/τ), with the Vallado
  table and Cd = 2.2. A fragment counts as re-entered below a 120 km perigee.
  This is not SGP4.
- Deterministic: seed 20090210, so the server and page see the same cloud.
- Altitude band: the 10th to 90th percentile of fragment mean altitude after
  the breakup, rounded outward to 10 km (640 to 1,150 km).

## API contract

All routes are read-only JSON. An unknown id returns
`404 {"detail": "Scenario 'x' not found"}`.

### `GET /scenarios`
```json
{"count": 1, "scenarios": [{"id": "iridium-cosmos-2009", "title": "...",
  "summary": "...", "provenance": "reconstructed" | "archived",
  "provenance_label": "...", "event": {...}}]}
```

### `GET /scenarios/{id}`
Scenario metadata as in the file: `id, title, summary, provenance,
provenance_label, provenance_note, replay_hint, event{time_utc, time_label,
altitude_km, altitude_label, location_label, relative_speed_kms,
relative_speed_label, significance, debris_label, source, source_short},
screening_window, catalog_fragment_prefixes, sources[{label, url}],
reconstruction{...}` (reconstructed only) or `archive{...}` (archived only).
It also carries:

- `objects[]`: `{norad_id, name, intl_designator, role: primary|secondary,
  status, description, inclination_deg_published, mass_kg, mass_note,
  elset{epoch, line1, line2}, elset_count, elements{inclination_deg,
  raan_deg, eccentricity, period_min, apogee_km, perigee_km}}`
- `screening`:
  ```json
  {"method": {"screening": "...", "pc": "...", "window_hours": 72,
              "threshold_km": 15, "coarse_step_s": 60, "cadence_hours": 1,
              "risk_thresholds": {"elevated": 1e-6, "critical": 1e-4}},
   "primary_norad": 24946, "secondary_norad": 22675,
   "detected": true, "runs_before_detection": 2,
   "first_detection": {"run_at", "elsets_available", "detected", "tca",
                       "hours_before_tca", "miss_distance_m",
                       "relative_velocity_ms", "pc", "risk_level"},
   "detection": {"run_at", "tca", "miss_distance_m", "relative_velocity_ms",
                 "pc", "risk_level", "hours_before_tca",
                 "location": {"lat_deg", "lon_deg", "alt_km", "velocity_kms"}},
   "history": [ /* first_detection-shaped runs, lead time descending */ ]}
  ```
  `tca` strings carry milliseconds (`2009-02-10T16:56:00.000Z`).
- `encounter`: `{tca, lat_deg, lon_deg, alt_km, gmst_rad}` at the refined TCA.
- `tracks`: `{start, step_s: 30, primary: [[x,y,z] km TEME], secondary: [...]}`,
  SGP4 positions for 120 minutes from `first_detection.run_at`.

### `GET /scenarios/{id}/ephemeris?window_s=1200`
The same shape as `/conjunctions/{id}/ephemeris`: `conjunction_id` (null),
`tca`, `window_s` (clamped to 60..3600), `step_s` (1.0), `start`,
`gmst_start_rad`, `earth_rot_rad_s`, `primary{norad_id, name, r}`,
`secondary{...}`, `tca_state{sep_m, rel_vel_ms, ric_m, approach_angle_deg,
rel_vel_hat}` and `sigma_m`. It adds `scenario_id`, `provenance`,
`tca_precise` and `start_precise` (millisecond ISO).

### `GET /scenarios/{id}/debris?hours=0&limit=1500`
`hours` accepts 0..87,600 (422 outside that range). `limit` accepts
1..5,000 and caps the number of fragments drawn (an even subsample).
```json
{"scenario_id", "provenance",
 "model": "illustrative model (NASA Standard Breakup Model, simplified)",
 "model_version": "sbm-simplified-1", "model_notes": ["..."],
 "epoch": "<TCA ms ISO>", "hours": 24.0, "time_utc": "...",
 "energy": {"specific_energy_j_per_g", "catastrophic_threshold_j_per_g", "catastrophic"},
 "counts": {"modeled_total_ge_10cm", "by_parent": {"<norad>": n},
            "total_mass_kg", "generated", "in_orbit_modeled",
            "reentered_modeled", "drawn"},
 "altitude_band_km": [640, 1150],
 "live_catalog": {"band_km", "catalog_objects", "objects_in_band",
                  "payloads_in_band", "collision_fragments_cataloged",
                  "collision_fragments_in_band", "fragment_name_prefixes",
                  "as_of", "label"},
 "parents": [{"norad_id", "name", "elements": {"a_km", "e", "i_rad",
              "raan_rad", "argp_rad", "m0_rad", "raan_dot", "argp_dot", "m_dot"}}],
 "fragment_fields": ["parent", "length_m", "area_to_mass", "dv_ms", "a_km", "e",
                     "i_rad", "raan_rad", "argp_rad", "m0_rad", "raan_dot",
                     "argp_dot", "m_dot", "tau_s", "scale_h_km"],
 "fragments": [[...]],          // one row per drawn fragment; tau_s null = no decay
 "positions": [[x,y,z] | null]  // TEME km at TCA + hours; null = re-entered
}
```
Rates are rad/s and times are seconds from TCA. The page advances
`fragments` with the same formulas for drawing. Counts always come from the
backend.

## Files

- `backend/app/scenarios/`: loader (`__init__.py`, prefers `*.archived.json`),
  `replay.py` (screening replay, ephemeris, tracks), `breakup.py` (model),
  `archive.py` (gp_history parsing), `iridium_cosmos_2009.json`
- `backend/app/api/scenarios.py`: routes, registered with one line in `main.py`
- `backend/tests/test_scenarios.py`: detection (miss < 1 km), shapes, 404s,
  model units, archived preference, gp_history fixture, and the fetch script
  end to end on a mock transport
- `scripts/reconstruct_scenario.py`, `scripts/fetch_scenario_tles.py`
- `frontend/replay.html`, `frontend/js/replay.js`,
  `frontend/css/pages/replay.css` (reuses `earthtex.js` read-only)
