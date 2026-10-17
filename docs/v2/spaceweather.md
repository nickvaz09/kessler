# Space weather

Kessler treats space weather as one more input to collision risk, not as a
separate page. Live NOAA conditions feed the same globe, the same screening
results and the same AI as everything else.

## Where it shows up

| Surface | What it shows | Code |
|---|---|---|
| Header, every page | The storm level ("G3 storm", "Sun quiet"). Links to the Overview's space-weather view. | `js/theme.js` (from `/status`) |
| Overview globe | Live aurora (NOAA OVATION) on the Earth. The atmosphere rim warms with the storm. A **Storm** color mode bins every object by how much denser the storm makes the air at its perigee. | `js/globe.js` `setSpaceWeather` |
| Overview, **Space weather** lens | The same stage, seen from the Sun, drawn from NOAA's live readings only. Details below. | `js/sunview.js`, `js/spaceweather.js` |
| Overview sheet, *Space weather* | Details below. | `js/spaceweather.js` |
| Close-approach table | A **Storm** tag on the pairs to recheck, and a sentence in each row's drawer. | `js/dashboard.js` |
| Object page | What the storm does to *this* orbit: air density, altitude loss now vs a quiet day, and storm drift by tomorrow and in 3 days. | `js/object.js` |
| Kessler AI | `get_space_weather`: the summary, the LEO density profile, flagged approaches, and one object's effects. | `assistant/tools.py` |
| Demo deck | A slide that illustrates a G3 storm with the lens's physics: the Sun, the wind bent around the shield, the aurora, the storm-heated air. Its footer shows NOAA's live Kp when the app serves the deck. | `deck.html`, `js/decksun.js` |

The **Space weather** lens shows:
- the solar wind deflected around the magnetopause (placed by the live wind);
- the bow shock and field lines;
- energy precipitating into both poles;
- the auroral ovals;
- the upper atmosphere swelling around a LEO satellite;
- the GEO belt.

It is a tool, not a simulator: there are no scenarios and no staged
events. Everything it draws comes from the latest NOAA readings, eased in as
they update. `/?view=spaceweather` opens it.

The *Space weather* sheet section shows:
- the storm level, with what it does to LEO air;
- the flare and the solar wind;
- Kp for the last 2 days, plus NOAA's 3-day forecast;
- both poles right now (OVATION on the real Earth);
- the close approaches it flags.

## The chain

| Step | Arrives | What happens |
|---|---|---|
| Flare (X-rays) | ~8 minutes | Radio blackouts on the sunlit side (R scale). Orbits are not affected. |
| CME / fast solar wind | hours to ~3 days | A geomagnetic storm (Kp, G scale). Energy enters around both magnetic poles through Joule heating and particle precipitation. |
| Heated thermosphere | during the storm | The upper atmosphere swells, so air density at a given LEO altitude rises (NASA/NRL MSIS 2.1). |
| More drag | from then on | Orbits sink faster. Predictions that leave out the extra drag drift along-track. |

## The storm-drift model

Every TLE carries the drag its object actually felt while the TLE was
fitted: the mean-motion derivative ṅ. The prediction extrapolates that drag.
The storm's contribution is how far the air at the object's altitude moves
away from what it was during the fit:

```
R(Kp, h)      NRLMSIS 2.1 density ratio vs a quiet day (Kp 1), per Kp level and altitude
R_fit         mean R over the 36 h before the TLE epoch
extra ṅ(t)    = (R(Kp(t), h) / R_fit − 1) · ṅ
drift(T)      = a · | ∫_{t0}^{T} (T − t) · extra ṅ(t) dt |      (along-track, metres)
```

- **Span.** t0 is the TLE epoch, so this is the same span the assumed sigma
  uses.
- **Kp(t).** NOAA's 3-hourly planetary Kp: observed for the past, NOAA's
  forecast for the next 3 days, then the last value held.
- **What counts.** A TLE fitted inside a steady storm already carries it,
  so there is no extra. A storm that starts after the fit moves the object;
  so does one that ends after the fit.
- **Cap.** Drift is capped at 1,000 km. Beyond that a public TLE says little
  anyway.
- **Altitude.** h is the mean altitude, capped 60 km above perigee for
  eccentric orbits. Above 1,000 km the drift is zero.
- **Cost.** The integral is exact for piecewise-constant Kp. Prefix sums per
  altitude column make it O(log n) per object, so 13,000+ active pairs take
  about 0.3 s, once per NOAA refresh (`conjunctions.active_storm_items`).
- **Live values (G3, Kp 7, 10 Oct 2026).**
  - The ISS decays 66 m/day (36 m/day on a quiet day).
  - Storm drift is 2.3 km by tomorrow and 9.3 km in 3 days.
  - This is in line with published storm-time SGP4 errors.

## What it changes, and what it never changes

For each close approach, `attach_space_weather` adds:
- the storm drift of both objects, combined by RSS;
- the assumed sigma;
- the widened sigma, √(σ² + drift²);
- `widened`: true when the widening is at least 25 %;
- `pc_storm`: the assumed-sigma Pc at the widened sigma;
- a `flag`:
  - `watch`: already elevated or critical, and widened;
  - `may_escalate`: nominal now, but the widened uncertainty allows Pc >= 1e-6.

**The stored Pc and the risk level are never changed.** A wider sigma
usually dilutes Pc, and Kessler never presents that dilution as comfort. The
storm marks what to recheck after the next orbit update.

## Data and operations

- **Sources.** NOAA SWPC (services.swpc.noaa.gov), public and keyless:
  - planetary Kp and its forecast;
  - the R/S/G scales;
  - GOES X-rays;
  - DSCOVR/ACE real-time solar wind;
  - F10.7 (daily values, 81-day mean);
  - the OVATION aurora nowcast.
- **Fetch.** All 9 feeds in parallel, 10 s timeout. One dead feed never
  takes the others down.
- **Cache.** 5 minutes, stale-while-revalidate. `spaceweather.peek()` never
  blocks a request. A stale or cold cache starts a single background
  refresh, retried at most every 60 s after a failure. `main.py` warms it at
  startup.
- **Outage.** Data is served stale (flagged) or not at all. Every surface
  then hides or says "unavailable", and screening and risk are unaffected.
- **Switch.** `KESSLER_SPACEWEATHER=0` turns live fetching off. The test
  suite runs this way (`backend/tests/conftest.py`).
- **Endpoints:**
  - `GET /spaceweather`: conditions plus `summary`.
  - `GET /spaceweather/impact`: widened and flagged approaches.
  - `GET /spaceweather/orbit`: what the current storm does to one circular orbit.
  - `space_weather` on `/status`, `/conjunctions[/{id}]` and `/objects/{norad}`.

## Physics references

- **Air density.** NRLMSIS 2.1 via `pymsis`, orbit-averaged over latitudes
  ±60° and all longitudes. Each Kp level drives a daily Ap equal to its
  3-hourly ap (sustained conditions).
- **Magnetopause standoff.** Shue et al. (1998). The lens draws the shield as
  a Rankine half-body with that standoff, a close match to Shue's shape and
  an exact potential flow for the wind around it.
- **Lens distances.** Compressed (r^0.78). The transform is monotonic, so the
  shield is drawn inside the GEO belt exactly when the standoff is below 6.6
  Earth radii.

## Background: storms that changed orbits

These events motivate the feature. They are not part of the product.

- **February 2022.** A moderate storm raised drag on newly launched Starlink
  satellites by up to 50%. 38 of 49 re-entered (SpaceX; NASA SVS).
- **May 2024, G5 (the "Gannon" storm).**
  - About 5,000 Starlinks maneuvered.
  - The aurora was seen as far south as Florida and Mexico.
  - Orbit predictions for many satellites drifted by kilometres.
  - Sources: Parker & Linares 2024; a National Academies presentation.

`window.KSpaceWeather.simulate(seconds)` steps the lens clock, which is
useful for stills: a hidden tab gets no animation frames.
