# Kessler - engineering write-up

Kessler is an AI orbital collision-risk platform for the operators enterprise
space traffic management does not serve: university CubeSat teams and smallsat
startups. It ingests public orbital data, screens ~32k objects for close
approaches with SGP4, scores each event with a trained model that beat the
published ESA baseline, and puts an agentic AI - an operations assistant and a
scheduled watch officer - at the center of the workflow. The name is the
failure mode it exists to help prevent: the Kessler syndrome, cascading
collisions in LEO.

## The problem

Conjunction assessment for small operators today is a CelesTrak bookmark, a
Space-Track login, and a spreadsheet. Screening services exist, but the
judgment layer - is this event getting worse, what does the CDM sequence
imply, what would a burn buy - is either an enterprise contract or a human
reading numbers at 2 a.m. Kessler's bet is that the judgment layer is now
buildable: a model trained on real operator CDM sequences plus a language
model with tools, both engineered to never claim more than the data supports.

## System shape

FastAPI over plain `sqlite3` (no ORM; Postgres via a thin adapter), 29 REST
routes, and a dependency-free vanilla-JS frontend - no build step, no npm, no
CDN, fonts vendored. Six pages (dashboard, object detail, alerts, my fleet,
ops wall, model card) share 20 JS modules (12.2k lines); the backend is 9.3k
lines of application code plus 7.6k lines of tests.

```
CelesTrak GP TLEs      Space-Track cdm_public       ESA Kelvins archive
      │                        │                     (offline training)
      ▼                        ▼                            ▼
backend/app/ingestion/                          backend/app/models/
  celestrak · propagate · screening ·             pc · features · risk_model ·
  spacetrack · pipeline                           train · maneuver · anomaly
      └──────────────► SQLite / Postgres ◄──────────────┘
                              │
              backend/app/api (41 routes) + assistant/ (Claude tool loop)
                              │
        frontend/ — dashboard · object · alerts · fleet · wall · model
```

`docs/CONTRACTS.md` is the interface spec; the system grew through 15 contract
versions, each specified before it was built.

## Real data vs simulated - the exact line

Real, always:

- **TLEs**: CelesTrak GP groups (`active`, `stations`, `cubesat`, the Cosmos
  2251 / Iridium 33 / Fengyun-1C debris clouds), deduplicated per group with
  per-group failure isolation (CelesTrak renames and rate-limits groups;
  a dead group is skipped, never fatal),
  by NORAD id keeping the newest epoch. The dev database holds 32,328
  satellites and 69,925 TLE-history rows; with Space-Track credentials and
  `KESSLER_FULL_CATALOG=1` the pipeline also pulls the full public GP class,
  bringing the catalog to ~32k objects.
- **Conjunctions**: events come from our own SGP4 screening of real TLEs,
  plus Space-Track public CDMs whose encounter velocity can be derived. Each pipeline run appends one real CDM row per event
  (`source='screening'`); a month of dev operation has accumulated 22,695
  conjunctions, 55,783 CDM rows and 26,668 model predictions.
- **Space-Track CDMs**: with credentials set, each refresh pulls public
  `cdm_public` rows (`source='spacetrack'`).
- **Training data**: 162,634 real operator CDMs from ESA's Kelvins Collision
  Avoidance Challenge (public Zenodo record 4463683).

Simulated, always labeled:

- **Demo CDM backfill** (`--demo`): fresh databases have one real CDM per
  event per run, so trend charts need history. Demo rows are derived from the
  real screened geometry with time-scaled uncertainty, stored
  `source='demo'`, flagged in the footer of every console page except the wall (`DEMO CDM SEQUENCES
  PRESENT`), and never overwrite screening rows.
- **Covariance**: public TLEs carry none, so screening Pc assumes a TLE-class
  sigma (below). The simulator's uncertainty circle is labeled `1σ ASSUMED`;
  when a real Space-Track covariance is present the API reports
  `pc_source: "covariance"` and the UI tags the number `COV`.
- **Estimates that are estimates**: the reentry panel is titled `TLE-DERIVED
  ESTIMATE - NOT A REENTRY PREDICTION`; launch-window scans carry `COARSE
  120 S SCAN - SCREENING-GRADE ONLY`; maneuver results state their two-body
  approximation.

## Screening - the part that had to be right

A coarse pass propagates every LEO object (apogee <= 3,000 km) on a shared
60 s grid with `SatrecArray`, chunked to bound memory, computing
primary-vs-all distances in numpy over a 72 h window. The first working
version had a temporal aliasing bug: at ~14 km/s closure an object crosses
the 15 km threshold sphere in about two seconds, so a sampled-distance gate
misses minima that fall between samples. The fix is a velocity-aware capture
gate - the closed-form rectilinear minimum of |dr + dv·τ| over each segment,
with a +5 km margin for curvature - followed by 1 s refinement and parabolic
interpolation of d² through the discrete minimum for sub-second TCA.
Measured at the ~3k-object stage of the build, detections in one 72 h window
went from 76 to 686: the naive gate had discarded ~89% of real events.

Pairs closing under 10 m/s are excluded (docked modules and formation flyers
are co-moving, not conjuncting - without the cut they flood the board with
0 m "criticals"). Events dedupe across runs on (pair, TCA ± 30 min).

`cdm_public` publishes no relative speed, and fabricating one would be
dishonest data. Documented policy: use a supplied speed if present; else
reuse the encounter velocity of the matching screened conjunction (pair
geometry is stable across messages); else compute |v1 - v2| at TCA from
both objects' latest TLEs with SGP4; else skip the row and count it in
`skipped_no_velocity`. Where an event has operator CDMs, the board and the
model use them ahead of our assumed-sigma screening rows.

## The risk model and the ESA benchmark

**Screening Pc.** TLE states carry no covariance, so one is assumed: 300 m
1-sigma at epoch growing 400 m/day with propagation age, a standard 2-D
encounter-plane approximation (combined σ_c = √2·σ), 10 m hard-body radius.
All constants live in `config.py`; the model card's LIMITATIONS panel calls
them honest defaults, not operator-grade numbers.

**Trained model (`vigil-0.4`).** `train.py` trains on the Kelvins dataset:
162,634 CDMs grouped into 12,929 events with >=2 messages; target is the
final CDM's `risk` (log10 Pc), features are built from all prior CDMs via
the same `features.build_features` used in production (miss distance,
relative velocity, latest Pc, time to TCA, CDM count, miss-distance
delta/slope, log-Pc delta/slope, peak log-Pc, last message gap - 11 in
all). Event-level 80/20 split: 10,343 train / 2,586
held out, 76 of them high-risk (final log10 risk >= -6, the competition's
definition).

The published challenge result is that last-risk carry-forward is nearly
unbeatable on the official metric, because the metric conditions on the
final value: it scores only events whose final risk is high, so a model that
learns the real, common decay of risk toward TCA is punished exactly where
it is evaluated. Our first attempt - a plain absolute-target regressor -
reproduced that failure (high-risk MSE ~28 vs the baseline's 14.9). The
shipped model is a **hybrid residual**: at or above the high-risk threshold
it defers entirely to carry-forward; below it - where decay dominates and
carry-forward is weakest - it predicts the log10 correction to the
last-known risk, clipped to ±4 decades, and is trained only on that domain
(`--high-weight 0` excludes already-high events from training, since the
hybrid never acts there).

Held-out results (`data/models/risk_model.meta.json`): high-risk MSE
**13.77 vs 14.87** carry-forward; overall MSE **17.81 vs 22.59** (-21.1%).
Caveat, stated wherever the numbers appear: these come from our own split of
the public training file, not the official test set or scoring formula -
internally consistent, not leaderboard-comparable.

**The ship gate.** `train.py` refuses to save a model that loses to the
baseline. The gate was fixed before the hybrid was evaluated: high-risk MSE
within 2% of carry-forward AND overall MSE at least 10% better. A failing
model is not saved, any previous artifact is deleted, and the API keeps
serving the labeled heuristic (`heuristic-0.1`). Every prediction row
records its `model_version`; no risk number in the system is unattributed.
Risk levels are computed backend-side only (critical >= 1e-4, elevated >=
1e-6, NASA CARA-style); the frontend never derives them.

**Glass box.** `GET /model/card` and the MODEL page expose the metrics,
split sizes, gate conditions, xgboost gain-based feature importances, live
prediction counts, and limitations. `GET /risk/{id}/explain` returns
per-feature SHAP contributions (`pred_contribs`) when the model acted, or an
explicit `carry_forward` note when it deferred; the dashboard renders this
as a WHY THIS NUMBER bar block inside each event's expansion. A number you
cannot interrogate is a number you cannot trust.

## The AI layer

**Operations assistant.** `POST /assistant/chat` (and an SSE streaming
variant) runs a server-side agentic loop on `claude-opus-5-5` (env-
overridable), adaptive thinking, `max_tokens=16000`. The loop is manual: all
`tool_use` blocks execute per iteration, results return in one user message,
`is_error` on failures, `pause_turn` re-sent, 12-iteration cap, polite
degrade on refusal. 27 tools in two families: DATA tools that read the live
DB (status, conjunctions, objects, risk, changes, anomalies, passes,
maneuver simulation/optimization, fleet screening, risk explanation) and
UI-ACTION tools that drive the interface (navigate, globe select/filter/
speed, table filter, form filling, trajectory modal, encounter view, opening
the simulator). Streamed actions execute the moment they arrive - the model
visibly operates the page while it talks - with a step timeline showing each
tool as it runs. `create_report` renders printable briefs from real DB
tables at `/reports/{uuid}`. The system prompt is frozen (no interpolated
values) with a `cache_control` breakpoint so prompt caching hits; volatile
context travels in the user-message envelope.

**Watch officer.** With `KESSLER_WATCH_MODE=1`, every scheduled pipeline
refresh ends with `watch_pass`: one headless tool loop (DATA tools only,
enforced by a filtered tool list; iteration cap 6; effort low) that reviews
status, run-over-run changes, top conjunctions and maneuver-candidate
anomalies, then emits strict JSON via structured outputs - `summary_md`
(<=1,500 chars, watch-officer register) and `spoken` (<=450 chars, written for
TTS) - into a `watchlog` table. `GET /watchlog` feeds a dashboard WATCH LOG
panel; the assistant's SHIFT BRIEF action replays the latest entry, spoken
aloud, without spending a chat call. A failed or keyless pass logs to stderr
and never breaks the pipeline.

**Voice.** Output: `POST /assistant/tts` synthesizes via ElevenLabs
(`eleven_multilingual_v2`, fixed voice settings, 0.9 speed), with a
server-side disk cache keyed by sha256(model|voice|settings|text) so no clip
is paid for twice; a `spoken_text` pass expands notation for the ear
("3.1e-5" -> "times ten to the minus five", "m/s" -> "meters per second")
while captions keep the exact figures. The encounter simulator's NARRATE
mode gets its script from one structured-output Claude call - 8-14 timed
beats, numbers only from the provided ephemeris, cached 10 minutes - and
plays each beat against sim time. Input: a push-to-talk mic in the assistant
drawer (Web Speech API, interim results mirrored into the input,
auto-send on final). Every voice path degrades silently to text when keys or
APIs are absent; the ElevenLabs key never leaves the server.

**Honesty engineering.** The prompt's DATA HONESTY block is non-negotiable:
never state a value not fetched by a tool in the current turn (prior turns'
tool results are dropped from context, so the model must re-fetch); cite
`model_version` for every risk claim; report backend-computed risk levels,
never re-derive them; say plainly when data is missing. Tool-returned
strings - satellite names included - are declared data, never instructions.
The same posture runs through the platform: demo rows labeled and flagged,
assumed sigmas labeled `ASSUMED`, skipped CDMs counted rather than faked,
estimates titled as estimates, and a ship gate that prefers no model to a
worse one.

## v2 - from a monitor to a decision tool

v1 answered "what is close?". v2 is organized around the three decisions an
operator without an ops room actually has to make, plus a way to feel why it
matters.

- **"Should I start worrying now?" - VIGIL-E.** VIGIL-0.4 defers to the
  observed probability at high risk, so in live use it never changes a risk
  level; its value is in demoting stale nominal alarms. VIGIL-E is a second,
  decision-shaped model: for an event still below the high-risk line
  (log10 Pc < -6) it estimates the probability that the final message lands
  above it. It trains on every prefix of each Kelvins event (the moments an
  operator could have asked), with an event-level held-out split, a
  validation carve for early stopping, Platt calibration and the alert
  threshold, and honest baselines (the carry-forward rule and a logistic
  regression on latest risk). Its headline metric is **lead time** - how
  much earlier than carry-forward it flags events that do escalate -
  reported together with the share caught early. A ship gate fixed in code
  (PR-AUC margin over the logistic baseline on every seed of a re-run,
  positive median lead time) decides whether it ships; nothing is saved
  otherwise and every early-warning surface stays empty. The VIGIL Lab has
  since scored it under rules committed before any result. On ESA's full
  data, covariance included, the research build passed against its
  pre-registered baseline, a logistic regression on the latest risk (PR-AUC
  0.126 vs 0.065, 54 escalations, pooled cross-validation). A later
  independent re-check found that baseline weakened by how its
  cross-validated scores were pooled: ranked by the latest risk itself, the
  escalations score 0.098, and the research build's gain over that is not
  significant; the version limited to the features Kessler's live feed
  carries scores below it. So the app ships no VIGIL-E model, and its
  early-warning surfaces stay empty. The same research classifier does work
  as a false-alarm filter: it removes 28% of the baseline's false alarms
  while keeping 95% of its true calls, against 9% for the latest risk alone.
- **"What would I do about it?" - the avoidance plan.** For a screened
  encounter, Kessler searches (golden section, both directions) for the
  smallest along-track burn at 1 orbit, 3, 6 and 12 hours before closest
  approach that raises the miss distance above a target and brings the
  screening Pc under the elevated line, using the same two-body differential
  model as the what-if panel. The simulator replays the encounter before and
  after. It is labeled screening-grade everywhere: no covariance, no
  thruster model, not an operational maneuver plan. When the latest message
  and the latest TLEs disagree about the geometry, it refuses to recommend
  anything rather than returning a comforting "no burn needed".
- **"Is my satellite OK?" - Protect a satellite.** A guided flow that screens
  a cataloged object, a pasted TLE or a hypothetical planned orbit (labeled
  as such) against the live catalog and states the verdict in one sentence.
  It never states an all-clear when the catalog is empty, stale, failed to
  refresh, or when the element set itself is old.
- **Why it matters - Time Machine.** A replay of 10 February 2009: Kessler's
  own screening code, run hour by hour on only the element sets published
  before each run, flags Iridium 33 against Cosmos 2251 about 72 hours
  before closest approach. The shipped orbits are *reconstructed* from
  published parameters (NASA ODQN, April 2009) and built to meet at the
  published point, so the near-zero predicted miss is by construction; the
  page says so, and `scripts/fetch_scenario_tles.py` swaps in the archived
  Space-Track element sets. The debris cascade that follows is an
  illustrative, simplified NASA Standard Breakup Model propagated with J2
  and a drag term - labeled as a model, next to the real cataloged count.
- **"Will the Sun move my orbit?" - space weather.** Live NOAA SWPC feeds
  (Kp and its forecast, the R/S/G scales, GOES X-rays, the real-time solar
  wind, F10.7, the OVATION aurora) run through NRLMSIS 2.1 to give the storm's
  air density at each object's altitude. Each TLE's own drag term, scaled by
  how far the air moves from what it was while the TLE was fitted, gives an
  along-track storm drift; each close approach gets its uncertainty widened
  by it and is flagged to recheck when that matters. The stored Pc and risk
  level are never changed (a wider sigma usually dilutes Pc, and Kessler
  never presents that dilution as comfort). It shows on every page: the
  header pill, the globe's aurora and storm color mode, a Sun-to-Earth lens,
  the close-approach table, the object page, the ops wall and the AI
  (`docs/v2/spaceweather.md`).
- **Demo-day honesty.** Offline snapshots (`backend/app/snapshot.py`) let the
  app run without network while the header says "Offline snapshot" with the
  data's real age; a failed refresh surfaces as "Refresh failed", never as
  "all clear".

The interface was rebuilt around the same principle: plain language first,
the precise figure one level down (Pc in a tooltip under "1 in 11,000"),
light and dark appearance, and the globe as the home screen.

## Engineering rigor

- **Tests**: 387 backend tests across 19 modules at time of writing (v13
  adds watch-log and explanation suites), all offline - fixture TLEs, temp
  databases, never `data/kessler.db`, no network. The assistant is tested
  against a scripted fake client (tool-use sequences, SSE event order,
  iteration caps, refusal and keyless paths); screening tests pin the
  aliasing regression.
- **CI** (`.github/workflows/ci.yml`, three jobs): the full pytest suite on
  Python 3.13; `node --check` on every frontend module on Node 22; and a
  `postgres:16` service job running the adapter's end-to-end lifecycle
  against a real server.
- **Contracts**: every module was written against exact interface specs in
  `docs/CONTRACTS.md`, and the document outranks the code when they disagree.
- **Design rules enforced at review**: tokens only (no hex in JS), amber/red
  reserved exclusively for risk severity, risk never shown by color alone,
  no gradients, no skeleton flashes, hairline density targets. The review
  list exists because these are exactly the tells that make generated UIs
  look generated.
- **Deployment**: SQLite by default; `KESSLER_DATABASE_URL` switches to
  Postgres through an adapter that translates placeholders and
  `INSERT OR IGNORE` semantics so every call site runs on both. `deploy/`
  ships a Dockerfile and compose file (app + `postgres:16` + healthcheck).
- **Cost discipline**: prompt caching, TTS and narration caches, a 10-minute
  reentry cache, watch mode default-off, and demo/dev endpoints documented
  as costing real money.

## Limitations

- **Covariance is assumed, not measured.** 300 m + 400 m/day is a
  literature-informed default; real operator covariance changes Pc by orders
  of magnitude. Screening Pc is a triage number, not a maneuver-decision
  number.
- **Maneuvers are invisible until the next TLE.** SGP4 over public elements
  cannot anticipate an operator burn; the sandbox's post-burn propagation is
  two-body and says so.
- **Model transfer.** The model learned from ESA operator CDMs with real
  covariances; live features rest on the assumptions above. Treat learned
  corrections as trend guidance.
- **Metrics are split-internal.** Kelvins numbers come from our own 80/20
  event split, not the official test set.
- **Catalog coverage** without Space-Track credentials is now small:
  CelesTrak's `active` group 403s scripted clients (as of 2026-08), leaving
  only stations, CubeSats and the major debris clouds. The full ~32k catalog
  requires Space-Track credentials and `KESSLER_FULL_CATALOG=1`.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m backend.app.ingestion.pipeline --demo   # ingest + screen
.venv/bin/uvicorn backend.app.main:app                      # api + ui on :8000
.venv/bin/python -m pytest backend/tests -q                 # offline suite
```

Optional env: `ANTHROPIC_API_KEY` (assistant + watch officer),
`ELEVENLABS_API_KEY` (voice), `SPACETRACK_IDENTITY`/`SPACETRACK_PASSWORD`
(public CDMs), `KESSLER_FULL_CATALOG=1` (full ~32k Space-Track catalog;
needs the credentials), `KESSLER_REFRESH_INTERVAL_H` (scheduler),
`KESSLER_WATCH_MODE=1` (watch officer on schedule). Model training:
download the Kelvins archive (Zenodo 4463683) to `data/kelvins/` and run
`.venv/bin/python -m backend.app.models.train` -- the ship gate decides
whether the result deserves to serve.
