# v2: "Protect a satellite" (/start.html) and About (/about.html)

## New endpoint: `POST /user-objects/from-catalog`

Copies a tracked object's **latest** catalog TLE into a new user object. The
catalog itself is only read: `satellites` / `tle_history` are SELECTed, never
written.

Request body:

```json
{ "norad_id": 25544, "name": "optional, max 64", "note": "optional, max 500" }
```

Responses:

| Status | When | Body |
|---|---|---|
| 201 | New user object created | user-object item (same keys as `POST /user-objects`) plus `"created": true` |
| 200 | A user object already holds this exact TLE pair (idempotent) | that existing item plus `"created": false` |
| 404 | `norad_id` not in the catalog | `{"detail": "Object N is not in the tracked catalog"}` |
| 413 | Fleet full (`USER_OBJECTS_MAX`) | `{"detail": ...}` |
| 422 | Bad id (negative / > int64 / not an int), object has no TLE, or its latest TLE fails the strict validation used by `POST /user-objects` | `{"detail": ...}` |

`name` defaults to the catalog name. Screen the copy with the existing
`POST /user-objects/{id}/screen`. Its own catalog entry is not reported as an
event, because the screen's co-moving filter drops it (relative speed is about 0).
Tests: `backend/tests/test_user_objects_from_catalog.py`.

## /start.html (js/start.js, css/pages/start.css)

Three steps, one question per screen, with progress dots:

1. **Which satellite is yours?** Three ways in:
   - a catalog typeahead (`GET /objects?search=&limit=8`, keyboard: ↑ ↓ Enter Esc)
   - a pasted TLE, checked in the browser for leaders, length, checksum, NORAD
     match and epoch age, then strictly by the server; 422 details are shown
   - a **planned orbit** (altitude and inclination), which generates a circular
     TLE with RAAN, perigee argument and mean anomaly all 0. It is labeled
     "Hypothetical orbit" in its name, its note, a chip and the fine print.
   - `?norad=<id>` preselects a catalog object.
2. **Screening:** create the user object (from-catalog, or `POST /user-objects`,
   reusing an existing object with the same NORAD id and epoch), then
   `POST /user-objects/{id}/screen`. The copy uses the catalog size and window
   from `/status` and shows a real elapsed timer. On failure there is no verdict,
   only Try again / Back.
3. **Result:** a one-sentence verdict.
   - "N close approaches need attention" (orange or red) comes from the
     server's `risk_level`.
   - Kessler **never** says "All clear" if any of these hold: `/status` failed,
     `last_run_error`, the catalog is empty, the catalog is older than 72 h, the
     TLE epoch is more than 14 days old, or no orbit could be derived.
     Otherwise it says "All clear for the next N hours".
   - Buttons:
     - Open in simulator: `/?conj=<id>` when the same pair has a catalog event
       with TCA within 1 h. Otherwise `/?sel=<norad>` ("Show on the globe") for
       catalog objects. Hidden for pasted and hypothetical TLEs.
     - Watch in my fleet: `/fleet.html`.
     - Brief me: `POST /assistant/chat` with a crafted message, rendered as safe
       markdown. A 503 shows a "not switched on" card instead.

Documented constants used in copy: 15 km threshold (`SCREENING_THRESHOLD_KM`),
3,000 km apogee cutoff (`SCREENING_MAX_LEO_ALT_KM`), 1-in-a-million attention
level (`PC_ELEVATED`).

## /about.html (js/about.js, css/pages/about.css)

- Hero: a canvas dot globe. The land mask is rasterized from
  `data/coastlines.json`, with illustrative orbits.
- Live numbers come from `/status`, with empty, failed, stale and offline states.
- VIGIL numbers come from `/model/card`, with the caveat that they were measured
  on our own held-out split of the Kelvins training data. The heuristic fallback
  is stated plainly.
- The Kessler AI tool list is a constant that mirrors
  `backend/app/assistant/tools.py` (25). **Update it if TOOLS changes.**
- The Time Machine teaser links to `/replay.html`.
  Honest-by-design rules, audience, credits.
- The page loads `assistant.js` so the assistant is reachable from About.
