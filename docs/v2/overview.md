# Overview page (v2) - notes for other areas

## KGlobe additions (js/globe.js) - all optional, backward compatible

- `init({ ..., wheelZoom: 'always' | 'modifier', onWheelHint })` -- default
  `'always'` (unchanged). `'modifier'`: plain wheel scrolls the page;
  ctrl/⌘ + wheel (and trackpad pinch) zooms; `onWheelHint()` fires on a
  plain wheel so the page can show a hint. The Overview uses `'modifier'`.
- `select(norad, { fly: false })` -- the second argument is optional. By
  default a programmatic `select()` and `showEncounter()` ease the view so the
  object (or the encounter midpoint) faces the viewer. Never while
  `setAutoSpin` is non-zero (the wall keeps its own view); pass
  `showEncounter({ ..., fly: false })` to opt out. Canvas clicks never fly.
- `setViewInset({ top, right, bottom, left })` -- css px; the Earth is framed
  in the uncovered part of the canvas (an axis whose inset leaves < 240 px is
  ignored).
- `setAtmosphere(on)` -- soft limb glow, default on.
- `zoomBy(factor)` -- eased zoom (e.g. `1.4` in, `1 / 1.4` out).
- `legend()` labels are now sentence case ("Payload", "Rocket body",
  "Elevated"...).
- Hover tip markup: `<b class="globe-tip__name">`, `<span class="globe-tip__sep">`
  (" · "), `<span class="globe-tip__meta">` -- reads as one line if unstyled.

## Space palette

The globe is a dark island in both themes. Its colors come from custom
properties resolved on the **canvas element**, falling back to built-in space
defaults, and are re-read on `KTheme.onChange`:

`--k-globe-payload, -debris, -rb, -dim, -elevated, -critical, -select,
-partner, -label, -grid, -grid-strong, -coast, -disk, -link, -atmosphere`

The Overview scopes them on `.stage` (css/pages/dashboard.css). The wall page
gets the same defaults without defining anything; it may scope its own.
Theme tokens such as `--k-text-2` are no longer used for globe drawing (they
turn dark in the light theme).

## Dashboard DOM

All ids other scripts use are unchanged (`stat-*`, `globe*`, `conj-*`,
`sel-body`, `changes-*`, `watchlog-*`, `timeline-*`, `KesslerUI`). `#conj-risk`
is now a hidden native select that stays the single source of truth; the
visible control is the `#conj-risk-seg` segmented control. The "Now" button
moved out of `#globe-speed` to `#globe-now`.
