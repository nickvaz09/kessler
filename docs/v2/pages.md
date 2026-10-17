# v2 pages: object, alerts, fleet, model, wall (+ shared helpers)

## js/format.js (all names and signatures unchanged)

Output changes (humane, sentence case):

| function | before | now |
| --- | --- | --- |
| `FMT.countdown(iso)` | `T−03:12:44` / `T+…` | `3 h 12 min`, `42 min 05 s`, `2 d 4 h`; past: `3 h 12 min ago` |
| `FMT.ageHours(iso)` | `15m`, `6.3h`, `3.2d` | `15 min`, `6.3 h`, `14 h`, `3.2 days` |
| `FMT.vel(ms)` | `10.80 km/s` | `10.8 km/s` (under 100 m/s: `42 m/s`) |
| `FMT.dist(m)` | `12.34 km` | `412 m`, `1.24 km`, `12.3 km`, `1,204 km` |
| `FMT.startUtcClock(el)` | - | now tolerates a missing element |

Unchanged: `pc`, `km`, `num`, `deg`, `utc`, `esc`.

New helpers: `FMT.when(iso, {seconds, year, day})` -> `Thu 24 Sep, 04:45 UTC`;
`FMT.rel(iso)` -> `in 3 h 12 min` / `12 min ago` / `just now`;
`FMT.pcOdds(p)` -> `1 in 11,000`; `FMT.dur(seconds)`; `FMT.span(ms)`.

## js/charts.js - drawLineChart(canvas, tipEl, opts)

Same API. Series/threshold colors may be token names (`'--k-data-1'`), which
are re-resolved on every draw; every drawn chart re-renders on
`KTheme.onChange` and resize. Labels are Inter, gridlines horizontal only,
thresholds dashed with the label at the left edge, time ticks thin out
automatically on narrow canvases. New optional opts: `area` (soft flat fill
under a single series), `emptyText`, `yFloor` (log scale: the axis stops at
the floor and lower points sit on it; the tooltip still shows the real value).

## Canvas modules

`orbit_plot.js`, `groundtrack.js`, `skyview.js` read colors through
`KTheme.css` and redraw on theme change. OrbitPlot is depth-aware (solid
Earth, far side of the orbit faint) and rotates by pointer drag or arrow keys.

## Page notes

- object.html: unknown or non-numeric ids render a search card (uses
  `/objects?search=`), never a red bar. Passes: "Use my location"
  (browser geolocation) or typed lat/lon, still stored in
  `localStorage['kessler.site']`.
- model.html: `<section id="vigil-e-slot"></section>` sits right after the
  metrics section and is left empty for another module.
- wall.html is always dark (pins `data-theme="dark"` without touching the
  saved preference).
