/* Kessler - orbital situational display engine (dashboard globe).
   Global: KGlobe - init / setSpeed / select / status / setConjunctions,
   plus setFilter / list (docs/CONTRACTS.md "KGlobe v2 interface additions"),
   plus showEncounter / setSimTime + status().simMs (encounter mode,
   docs/CONTRACTS.md "v4 additions"),
   plus setColorMode / getColorModes / legend / setHeatmap (color modes +
   density layer, docs/CONTRACTS.md "v6 additions"),
   plus setSurface / setAutoSpin (textured Earth + wall mode,
   docs/CONTRACTS.md "v7 additions"). All additive - existing surface
   unchanged.

   SURFACE mode (LUT renderer): when earthtex.js is loaded and its textures
   are decoded, the cached Earth layer replaces the disk fill + flat
   terminator + coastlines with KEarthTexture.drawHemisphere (the day/night
   blend is the terminator; the texture carries the coastlines); graticule +
   equator hairlines stay on top and the limb ring closes the layer. The
   cached-layer cadence is unchanged - idle re-renders happen on the same
   triggers as before. earthtex renders from a per-pixel LUT (spin is a
   free column shift; tilt changes rebuild the LUT inside earthtex), so an
   active drag re-renders the earth layer every moved frame at FULL quality -
   the half-quality drag mode and the wireframe-during-drag budget
   fallback are deleted. Toggle persists in localStorage 'kessler.surface'
   (default ON). Pages that do not load earthtex.js (or before decode) keep
   the wireframe - every KEarthTexture access is typeof-guarded.

   auto-spin (ops wall): setAutoSpin(degPerSec) yaws the view in the rAF
   loop. Satellites/orbits/labels project with the live spin every frame;
   while the textured surface is active the cached Earth layer re-renders on
   the existing GMST_EPS (0.2 deg) threshold instead of every sub-pixel spin
   step - the same quantization the engine already accepts for GMST drift -
   so wall-size per-pixel renders never land on every frame. The wireframe
   path keeps per-frame re-renders (cheap).

   color modes - DECISION (contract leaves risk-vs-type semantics open):
   risk severity dots (amber elevated / red critical, 3x3, drawn last) override
   the base color in every mode - amber/red stay risk-reserved. The DEFAULT
   mode is 'type': payload --k-text-2, debris+unknown --k-text-3, rocket body
   --k-spark - byte-identical to the older rendering. 'risk' differs from
   'type' only in that the non-risk base field is dimmed uniformly to
   --k-text-3 so the risk overrides pop. 'constellation' classifies by name
   prefix into the validated dark categorical slots; 'altitude' bins perigee
   into the validated sequential-blue ramp (steps 250->650). Constellation /
   ramp hexes are contract-fixed constants (no --k-* tokens exist for them);
   everything token-backed stays token-read. Blit color tables and draw-list
   grouping are rebuilt only on mode change / load / filter - never per frame.

   Orthographic wireframe Earth (graticule + real coastlines, back hemisphere
   culled, flat solar terminator) with the full catalog propagated client-side
   by two-body Kepler from the /fleet/positions state-vector snapshot (TEME).
   Earth-fixed features rotate with GMST inside the inertial frame; satellites
   do not.

   render pipeline (18.7k objects at 60 fps):
   - physics tick ≠ render tick: the Newton/Kepler solve runs on a ~100 ms
     wall-clock cadence into two sim-time snapshots (simA->posA, simB->posB,
     B = A + tick of SIM time); each frame lerps between them - pure
     multiply-add over Float64Arrays, no trig. The next snapshot (posC) is
     pre-solved in per-frame chunks between boundaries, so crossing simB is
     just a buffer rotation - the full-catalog solve never lands on one
     frame. Speed change / tab resume / overrun re-anchors both snapshots,
     so rendering never extrapolates.
   - the dot field is blitted as raw RGBA device pixels into one reused
     ImageData (2x2 normal, 3x3 risk blocks), putImageData'd into a dedicated
     offscreen canvas, then drawImage'd onto the main context so alpha
     composites over the Earth layer. Risk dots overwrite the neutral field.
   - the Earth layer (disk/terminator/graticule/coastlines/limb) renders into
     a second offscreen canvas, re-rendered only when view state changes,
     |ΔGMST| > 0.2°, or while dragging; otherwise it is a single drawImage.
   Selection rings, labels and orbit polylines stay vector on the main
   context. One RAF loop, zero per-frame heap allocation in the hot path.
   Contract: docs/CONTRACTS.md "Fleet situational display".

   design (Overview hero): the globe is drawn as a dark "space" island in
   both themes. Colors come from --k-globe-* custom properties resolved on the
   CANVAS element (so a page can scope a space palette onto its stage), with
   space-safe defaults when a page defines none; KTheme.onChange re-reads
   them. Additions, all optional and backward compatible:
   - atmosphere: a soft limb glow drawn into the cached earth layer
     (setAtmosphere(false) turns it off);
   - round, anti-aliased dot sprites (payload bright, debris dim, rocket
     bodies a cool tint; elevated/critical larger, critical with a halo);
   - animated selection ring + "fly to": public select()/showEncounter()
     ease the view so the object faces the viewer (skipped while auto-spin
     is on, or with select(norad, {fly: false}));
   - setViewInset({top, right, bottom, left}) (css px): frame the Earth in
     the part of the canvas not covered by floating UI;
   - sentence-case labels/legend and a two-part hover tip. */
'use strict';

const KGlobe = (() => {
  /* --- constants ---------------------------------------------------------- */
  const MU = 398600.4418;              // km^3/s^2
  const RE = 6378.137;                 // km
  const TWO_PI = Math.PI * 2;
  const DEG = Math.PI / 180;
  const J2000_MS = Date.UTC(2000, 0, 1, 12);
  const FRAME_KM = 9000;               // half-extent framed at zoom = 1
  const TILT_MAX = 88 * DEG;
  const ZOOM_MIN = 0.5, ZOOM_MAX = 8;
  const PICK_R2 = 10 * 10;             // hover/click radius^2 (css px)
  const CLICK_SLOP_PX = 4;             // < this movement => click, not drag
  const NIGHT_RGBA = 'rgba(0, 0, 0, 0.30)';  // flat terminator shade
  const TYPE_NAMES = ['PAYLOAD', 'DEBRIS', 'ROCKET BODY', 'UNKNOWN'];
  const TYPE_TOKENS = ['PAYLOAD', 'DEBRIS', 'RB', 'UNKNOWN'];  // filter/list vocabulary
  const TYPE_LABELS = ['Payload', 'Debris', 'Rocket body', 'Unknown type'];  // UI copy
  const TOKEN_CODES = { PAYLOAD: 0, DEBRIS: 1, RB: 2, UNKNOWN: 3 };
  const RISK_CODES = { nominal: 1, elevated: 2, critical: 3 };
  const GMST_EPS = 0.2 * DEG;          // earth-layer refresh threshold (~1 px @ zoom 1)
  const TICK_MS_BASE = 100;            // physics tick cadence (wall ms)
  const TICK_MS_MAX = 500;             // auto-stretch ceiling when solve is slow
  const SOLVE_CHUNKS = 5;              // frames the next-snapshot solve is spread over
  const ENC_SEGS = 24;                 // encounter connector subdivisions (disk culling)

  /* --- color modes / heat (docs/CONTRACTS.md "v6 additions") ---------------- */
  const COLOR_MODES = ['risk', 'type', 'constellation', 'altitude', 'storm'];
  const CONST_LABELS = ['Starlink', 'OneWeb', 'Iridium', 'Globalstar', 'Planet', 'Other'];
  /* validated dark categorical set, contract-exact slot order (OTHER = --k-text-3) */
  const CONST_HEX = ['#3987e5', '#d95926', '#199e70', '#9085e9', '#d55181'];
  const ALT_LABELS = ['Below 450 km', '450–600 km', '600–800 km', '800–1,200 km', 'Above 1,200 km'];
  /* validated sequential-blue ramp, steps 250/350/450/550/650: light -> dark =
     low -> high perigee (dense LEO shells get the brightest steps on the dark
     surface) */
  const ALT_HEX = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'];
  /* space weather (docs/v2/spaceweather.md): 'storm' bins each object's
     perigee air by how much denser the current storm makes it (NRLMSIS ratio
     vs a quiet day). A violet ramp - never the amber/red of risk, which still
     draws on top in every mode. Bin 0 = above the drag (1,000 km+). */
  const STORM_LABELS = ['Above the drag', 'Air up to 1.25× denser', '1.25–1.75×', '1.75–2.5×', 'Over 2.5×'];
  const STORM_HEX = ['#3c4452', '#6f6bd8', '#9d79ea', '#cf86ec', '#ff9ed8'];
  const STORM_MAX_KM = 1000;
  const MAX_GROUPS = 6;                // constellation mode has the most base groups
  const HEAT_CELL = 16;                // screen-bin size, DEVICE px (contract)
  const HEAT_LEVELS = 14;              // alpha quantization: level * 0.025, cap 0.35
  /* heat wash color table: sequential-blue anchor (step 400 #3987e5) at
     absolute per-dot alpha 0.025/dot capped at 0.35 - fixed mapping, so cell
     brightness is stable across frames (no per-frame max renormalization
     flicker). Strings built once at module init: zero frame-loop allocation. */
  const HEAT_STYLES = [null];
  for (let l = 1; l <= HEAT_LEVELS; l++) {
    HEAT_STYLES.push(`rgba(57,135,229,${(l * 0.025).toFixed(3)})`);
  }

  /* --- dom / style -------------------------------------------------------- */
  let canvas = null, ctx = null, tipEl = null, onSelectCb = null;
  const C = {};                        // resolved --k-* tokens (css strings)
  let labelFont = '10px monospace';
  /* dot-field RGBA per class, parsed from tokens once at init */
  const colPayload = new Uint8ClampedArray(4);
  const colDebris = new Uint8ClampedArray(4);
  const colRb = new Uint8ClampedArray(4);
  const colElev = new Uint8ClampedArray(4);
  const colCrit = new Uint8ClampedArray(4);
  const colDim = new Uint8ClampedArray(4);   // 'risk' mode uniform base field
  /* space palette defaults - used when the page defines no --k-globe-* token.
     The globe is a dark island in both themes, so these never follow the
     page theme (theme tokens like --k-text-2 turn dark in light mode). */
  const SPACE = {
    payload: 'rgba(238, 243, 252, 0.92)',
    debris: 'rgba(150, 160, 176, 0.62)',
    rb: 'rgba(137, 180, 245, 0.88)',
    dim: 'rgba(196, 206, 222, 0.50)',
    elevated: '#ff9f0a',
    critical: '#ff453a',
    select: '#2997ff',
    partner: '#f5f5f7',
    label: 'rgba(245, 245, 247, 0.92)',
    grid: 'rgba(255, 255, 255, 0.07)',
    gridStrong: 'rgba(255, 255, 255, 0.14)',
    coast: 'rgba(200, 212, 230, 0.55)',
    disk: '#0b0e14',
    atmo: 'rgb(92, 160, 255)',
    link: 'rgba(245, 245, 247, 0.55)',
  };
  let atmosphereOn = true;             // limb glow in the cached earth layer
  let wheelMode = 'always';            // init({wheelZoom}) - 'always' | 'modifier'
  let onWheelHint = null;              // init({onWheelHint}) - plain wheel in 'modifier' mode
  let zoomTarget = NaN;                // zoomBy() animation target (NaN = idle)
  let themeHooked = false;             // KTheme.onChange registered (once per page)
  /* view inset (css px) - the Earth is framed in the uncovered region */
  const inset = { top: 0, right: 0, bottom: 0, left: 0 };
  /* selection animation + fly-to (performance.now() ms) */
  let selAnimStart = -1e9;
  let fly = null;                      // {s0, ds, t0, dt, start, dur} or null

  /* --- view --------------------------------------------------------------- */
  let spin = 0.6;                      // rad, drag-controlled yaw
  let tilt = 24 * DEG;                 // rad, clamp ±88°; >0 shows the north pole
  let zoom = 1;

  /* --- surface + wall mode (docs/CONTRACTS.md "v7 additions") --------------- */
  let surfaceOn = true;                // SURFACE|WIREFRAME toggle, default ON
  try {
    if (typeof localStorage !== 'undefined'
        && localStorage.getItem('kessler.surface') === '0') surfaceOn = false;
  } catch (e) { /* storage disabled - keep the default */ }
  let autoSpinRad = 0;                 // rad/s yaw about the vertical axis (0 = off)

  /* --- sim clock ---------------------------------------------------------- */
  let speed = 60;                      // sim-time multiplier
  let epochIso = null;                 // snapshot epoch from /fleet/positions
  let epochMs = Date.now();            // sim base (wall clock until fleet loads)
  let simOffsetMs = 0;                 // accumulated wall-dt * speed

  /* --- catalog (parallel arrays, sized once at load) ----------------------- */
  let count = 0;
  let names = [];
  let noradArr = null;                 // Int32Array
  let typeCode = null, riskCode = null;               // Uint8Array
  let fleetRisk = null;                // Uint8Array - snapshot risk_code baseline
  let elA = null, elE = null, elB = null, elN = null, elM0 = null; // Float64Array
  let basisP = null, basisQ = null;    // Float64Array 3N, perifocal -> ECI
  let incDeg = null;                   // Float32Array
  let posX = null, posY = null, posZ = null;          // Float64Array, per frame (lerped)
  let scrX = null, scrY = null;        // Float32Array, cached screen coords
  let vis = null;                      // Uint8Array, 0 = occluded by disk
  const idxByNorad = new Map();

  /* --- physics snapshots (tick ≠ frame) ------------------------------------ */
  let posAX = null, posAY = null, posAZ = null;       // Float64Array, sim time simA
  let posBX = null, posBY = null, posBZ = null;       // Float64Array, sim time simB
  let posCX = null, posCY = null, posCZ = null;       // Float64Array, sim time simC (pre-solve)
  let simA = 0, simB = 0;              // sim ms offsets of the two live snapshots
  let simC = 0;                        // sim ms offset of the snapshot being pre-solved
  let cProgress = 0;                   // objects of the C snapshot solved so far
  let tickWallMs = TICK_MS_BASE;       // auto-stretched if the solve is slow
  let needAnchor = true;               // re-anchor both snapshots next frame

  /* draw-order index lists (rebuilt on load, setConjunctions, setFilter and
     setColorMode - never per frame). the base (non-risk) field is grouped
     per color mode into up to MAX_GROUPS lists; risk lists stay separate and
     are always blitted last, in every mode. */
  let baseLists = null;                // Array(MAX_GROUPS) of Int32Array(n)
  const baseCounts = new Int32Array(MAX_GROUPS);
  const baseCols = [];                 // Uint8ClampedArray(4) per group - mode-change only
  for (let g = 0; g < MAX_GROUPS; g++) baseCols.push(new Uint8ClampedArray(4));
  let listElev = null, listCrit = null;
  let nElev = 0, nCrit = 0;

  /* --- color mode + heat state --------------------------------------------- */
  let colorMode = 'type';              // DEFAULT - byte-identical to the earlier version (see header)
  let baseOrder = [0, 1, 2];           // group blit order for the current mode
  const ORDER_TYPE = [0, 1, 2];        // payload, debris(+unknown), rocket body
  const ORDER_RISK = [0];              // one uniform dimmed group
  const ORDER_CONST = [5, 0, 1, 2, 3, 4];  // OTHER first so branded dots sit on top
  const ORDER_ALT = [0, 1, 2, 3, 4];
  let constClass = null;               // Uint8Array - name-prefix slot (0-5), fixed at load
  let altBin = null;                   // Uint8Array - perigee bin (0-4), fixed at load
  const ORDER_STORM = [0, 1, 2, 3, 4];
  let perigeeKm = null;                // Float32Array - perigee altitude, fixed at load
  let stormBin = null;                 // Uint8Array - storm-drag bin (0-4), on setSpaceWeather
  /* space weather layer: live aurora (earth-fixed xyz + strength) and the
     density-ratio profile; swWarm (0..1) warms the atmosphere rim */
  let swAur = null, swAurN = 0, swAlts = null, swRatio = null, swWarm = 0;
  let auroraSprite = null;
  let appPaused = false;               // setPaused: a full-stage view covers the globe
  let heatOn = false;
  let heatBins = null;                 // Uint16Array bw*bh - realloc on RESIZE only
  let heatBW = 0, heatBH = 0;

  /* --- type/risk filter (default: everything visible) ----------------------- */
  const typeMask = Uint8Array.of(1, 1, 1, 1);  // indexed by typeCode
  let riskOnly = false;
  let filtOk = null;                   // Uint8Array, 1 = passes the filter
  let visibleCount = 0;                // post-filter rendered count (incl. sel/partner)

  /* --- earth line geometry (flattened xyz + line-start offsets) ------------ */
  let coastXYZ = null, coastStarts = null;
  let gratXYZ = null, gratStarts = null;
  let eqXYZ = null, eqStarts = null;

  /* --- selection / hover --------------------------------------------------- */
  let selIdx = -1, partnerIdx = -1, hoverIdx = -1;
  let pendingSelect;                   // norad queued before fleet load
  let pendingEncounter;                // showEncounter opts queued before fleet load
  let conjItems = [], conjSet = false;
  const selOrbit = new Float64Array(128 * 3);
  const partnerOrbit = new Float64Array(128 * 3);

  /* --- encounter overlay (persists until deselect/new select/destroy) ---- */
  let encActive = false;
  let encSecIdx = -1;                  // pinned partner index while the overlay is live
  let encLabel = '';                   // 'Closest approach hh:mm UTC'
  const encP = new Float64Array(3);    // primary TCA position (ECI km, fixed)
  const encS = new Float64Array(3);    // secondary TCA position
  const encM = new Float64Array(3);    // 3-D midpoint (label anchor)
  let encSepKm = NaN;                  // |encP - encS| on the globe's orbits
  let encPriNorad = null, encSecNorad = null;
  /* per-frame label boxes (x0, y0, x1, y1) so names, rings and the
     encounter caption never draw on top of each other */
  const LBL_MAX = 12;
  const lblBox = new Float64Array(LBL_MAX * 4);
  let lblN = 0;
  let pendingSimMs;                    // setSimTime() queued before the fleet loads

  /* --- per-frame view state + scratch (no allocation in the loop) ---------- */
  let fCw = 0, fCh = 0, fCx = 0, fCy = 0, fScale = 1, fRs = 1, fDpr = 1;
  let fCosT = 1, fSinT = 0;            // tilt
  let fCosS = 1, fSinS = 0;            // spin (inertial / ECI features)
  let fCosG = 1, fSinG = 0;            // spin + GMST (earth-fixed features)
  let pjX = 0, pjY = 0, pjD = 0;       // projection output slot
  const p3 = new Float64Array(3);      // kepler output slot
  const sunV = new Float64Array(3);

  /* --- offscreen layers ----------------------------------------------------- */
  let earthCanvas = null, earthCtx = null;   // cached earth (device px)
  let dotCanvas = null, dotCtx = null;       // dot-field compositor (device px)
  let dotImage = null, dotData = null;       // reused ImageData + its bytes
  let devW = 0, devH = 0;                    // device-pixel canvas size
  let earthDirty = true;
  let eSpin = NaN, eTilt = NaN, eZoom = NaN, eW = 0, eH = 0, eDpr = 0, eGmst = NaN;

  /* --- loop state ---------------------------------------------------------- */
  let initGen = 0;                     // bumped by init/destroy to cancel stale loads
  let rafId = 0, running = false, lastNow = 0;
  let dragging = false, dragMoved = 0, lastPX = 0, lastPY = 0;
  let downX = 0, downY = 0;

  /* ==========================================================================
     Kepler pipeline (validated against Vallado ex. 2-5; see build self-check)
     ======================================================================== */

  /* State vector (km, km/s) -> elements + perifocal basis, or null when the
     orbit is discarded (e >= 0.98, hyperbolic, or a <= 6478 km). */
  function elemsFromRV(rx, ry, rz, vx, vy, vz) {
    const rmag = Math.hypot(rx, ry, rz);
    const v2 = vx * vx + vy * vy + vz * vz;
    const a = 1 / (2 / rmag - v2 / MU);
    const hx = ry * vz - rz * vy;
    const hy = rz * vx - rx * vz;
    const hz = rx * vy - ry * vx;
    const hmag = Math.hypot(hx, hy, hz);
    if (!Number.isFinite(a) || a <= 6478 || hmag === 0) return null;
    const ex = (vy * hz - vz * hy) / MU - rx / rmag;
    const ey = (vz * hx - vx * hz) / MU - ry / rmag;
    const ez = (vx * hy - vy * hx) / MU - rz / rmag;
    const e = Math.hypot(ex, ey, ez);
    if (e >= 0.98) return null;
    if (a * (1 - e) <= RE) return null;  // perigee inside the Earth: decayed
    const wx = hx / hmag, wy = hy / hmag, wz = hz / hmag;
    let px, py, pz, E0;
    if (e > 1e-8) {
      px = ex / e; py = ey / e; pz = ez / e;
      /* e·sinE = r·v/sqrt(mu a);  e·cosE = 1 - |r|/a */
      E0 = Math.atan2((rx * vx + ry * vy + rz * vz) / Math.sqrt(MU * a), 1 - rmag / a);
    } else {
      /* circular: perigee direction undefined - anchor P at r, E0 = 0 */
      px = rx / rmag; py = ry / rmag; pz = rz / rmag;
      E0 = 0;
    }
    const qx = wy * pz - wz * py;
    const qy = wz * px - wx * pz;
    const qz = wx * py - wy * px;
    return {
      a, e, b: a * Math.sqrt(1 - e * e), n: Math.sqrt(MU / (a * a * a)),
      M0: E0 - e * Math.sin(E0),
      px, py, pz, qx, qy, qz,
      incDeg: Math.acos(Math.max(-1, Math.min(1, wz))) / DEG,
    };
  }

  /* advance object i by tSec since snapshot epoch -> ECI km into out[3] */
  function propagateOne(i, tSec, out) {
    const e = elE[i];
    let M = (elM0[i] + elN[i] * tSec) % TWO_PI;
    if (M > Math.PI) M -= TWO_PI;
    else if (M < -Math.PI) M += TWO_PI;
    let E = e > 0.8 ? (M < 0 ? -Math.PI : Math.PI) : M;
    for (let k = 0; k < 6; k++) {      // contract: Newton-solve Kepler (<=6 iters)
      const f = E - e * Math.sin(E) - M;
      E -= f / (1 - e * Math.cos(E));
      if (f < 1e-8 && f > -1e-8) break;
    }
    const xp = elA[i] * (Math.cos(E) - e);
    const yp = elB[i] * Math.sin(E);
    const j = 3 * i;
    out[0] = basisP[j] * xp + basisQ[j] * yp;
    out[1] = basisP[j + 1] * xp + basisQ[j + 1] * yp;
    out[2] = basisP[j + 2] * xp + basisQ[j + 2] * yp;
  }

  /* Fixed 128-point orbit ellipse (inertial - time-independent for two-body). */
  function buildOrbit(i, buf) {
    const a = elA[i], e = elE[i], b = elB[i], j0 = 3 * i;
    for (let k = 0; k < 128; k++) {
      const E = (k / 128) * TWO_PI;
      const xp = a * (Math.cos(E) - e);
      const yp = b * Math.sin(E);
      const o = 3 * k;
      buf[o] = basisP[j0] * xp + basisQ[j0] * yp;
      buf[o + 1] = basisP[j0 + 1] * xp + basisQ[j0 + 1] * yp;
      buf[o + 2] = basisP[j0 + 2] * xp + basisQ[j0 + 2] * yp;
    }
  }

  /* --- per-object classifiers (run once at load, never in the loop) --------- */

  /* Name-prefix constellation slot: STARLINK / ONEWEB / IRIDIUM / GLOBALSTAR /
     PLANET group (FLOCK | SKYSAT | PLANET) / else OTHER (5). */
  function constellationOf(name) {
    const u = name.toUpperCase();
    if (u.startsWith('STARLINK')) return 0;
    if (u.startsWith('ONEWEB')) return 1;
    if (u.startsWith('IRIDIUM')) return 2;
    if (u.startsWith('GLOBALSTAR')) return 3;
    if (u.startsWith('FLOCK') || u.startsWith('SKYSAT') || u.startsWith('PLANET')) return 4;
    return 5;
  }

  /* Perigee-altitude bin: [<450, 450-600, 600-800, 800-1200, >1200] km. */
  function altBinOf(perigeeKm) {
    return perigeeKm < 450 ? 0
      : perigeeKm < 600 ? 1
        : perigeeKm < 800 ? 2
          : perigeeKm <= 1200 ? 3 : 4;
  }

  /* --- astro helpers ------------------------------------------------------- */

  function gmstRad(ms) {
    const d = (ms - J2000_MS) / 86400000;
    return ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
  }

  /* Low-precision solar direction in ECI (unit vector) - drives the terminator. */
  function sunDirEci(ms, out) {
    const d = (ms - J2000_MS) / 86400000;
    const g = (357.529 + 0.98560028 * d) * DEG;
    const L = (280.459 + 0.98564736 * d) * DEG
      + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
    const eps = (23.439 - 0.00000036 * d) * DEG;
    const sl = Math.sin(L);
    out[0] = Math.cos(L);
    out[1] = Math.cos(eps) * sl;
    out[2] = Math.sin(eps) * sl;
  }

  /* ==========================================================================
     Projection.  p1 = Rz(spin)·p, p2 = Rx(tilt)·p1;
     screen = (cx + s·p2.x, cy - s·p2.z); depth toward viewer = -p2.y.
     North pole projects above center for tilt > 0.
     ======================================================================== */

  function projEci(x, y, z) {
    const x1 = x * fCosS - y * fSinS;
    const y1 = x * fSinS + y * fCosS;
    const y2 = y1 * fCosT - z * fSinT;
    const z2 = y1 * fSinT + z * fCosT;
    pjX = fCx + fScale * x1;
    pjY = fCy - fScale * z2;
    pjD = -y2;
  }

  function projEarth(x, y, z) {
    const x1 = x * fCosG - y * fSinG;
    const y1 = x * fSinG + y * fCosG;
    const y2 = y1 * fCosT - z * fSinT;
    const z2 = y1 * fSinT + z * fCosT;
    pjX = fCx + fScale * x1;
    pjY = fCy - fScale * z2;
    pjD = -y2;
  }

  /* ==========================================================================
     Earth geometry
     ======================================================================== */

  function llToXyz(lonDeg, latDeg, arr, o) {
    const la = latDeg * DEG, lo = lonDeg * DEG;
    const cl = Math.cos(la);
    arr[o] = RE * cl * Math.cos(lo);
    arr[o + 1] = RE * cl * Math.sin(lo);
    arr[o + 2] = RE * Math.sin(la);
  }

  /* Graticule every 30°; the equator is a separate (one step brighter) set. */
  function buildGraticule() {
    const latRings = [-60, -30, 30, 60];
    const nRing = 121, nMer = 57, nMerLines = 12;
    const total = latRings.length * nRing + nMerLines * nMer;
    gratXYZ = new Float64Array(total * 3);
    gratStarts = new Int32Array(latRings.length + nMerLines + 1);
    let p = 0, line = 0;
    for (const lat of latRings) {
      gratStarts[line++] = p;
      for (let k = 0; k < nRing; k++) llToXyz(-180 + k * 3, lat, gratXYZ, 3 * p++);
    }
    for (let m = 0; m < nMerLines; m++) {
      gratStarts[line++] = p;
      const lon = m * 30;
      for (let k = 0; k < nMer; k++) llToXyz(lon, -84 + k * 3, gratXYZ, 3 * p++);
    }
    gratStarts[line] = p;
    eqXYZ = new Float64Array(nRing * 3);
    for (let k = 0; k < nRing; k++) llToXyz(-180 + k * 3, 0, eqXYZ, 3 * k);
    eqStarts = Int32Array.from([0, nRing]);
  }

  async function loadCoastlines() {
    const gen = initGen;               // bail if destroy()/re-init raced the fetch
    const res = await fetch('data/coastlines.json');
    if (!res.ok) throw new Error(`coastlines HTTP ${res.status}`);
    const data = await res.json();
    if (gen !== initGen) return;
    const lines = data.lines || [];
    let total = 0;
    for (const line of lines) total += line.length;
    const xyz = new Float64Array(total * 3);
    const starts = new Int32Array(lines.length + 1);
    let p = 0;
    for (let l = 0; l < lines.length; l++) {
      starts[l] = p;
      const line = lines[l];
      for (let k = 0; k < line.length; k++) llToXyz(line[k][0], line[k][1], xyz, 3 * p++);
    }
    starts[lines.length] = p;
    coastXYZ = xyz;
    coastStarts = starts;
    earthDirty = true;                 // cached layer must pick the coastlines up
  }

  /* ==========================================================================
     Fleet snapshot
     ======================================================================== */

  async function loadFleet() {
    const gen = initGen;               // bail if destroy()/re-init raced the fetch
    const res = await fetch(new URL('/fleet/positions', window.location.origin));
    if (!res.ok) throw new Error(`fleet HTTP ${res.status}`);
    const data = await res.json();
    if (gen !== initGen) return;
    const objs = data.objects || [];
    const nMax = objs.length;
    names = new Array(nMax);
    noradArr = new Int32Array(nMax);
    typeCode = new Uint8Array(nMax);
    riskCode = new Uint8Array(nMax);
    fleetRisk = new Uint8Array(nMax);
    elA = new Float64Array(nMax); elE = new Float64Array(nMax);
    elB = new Float64Array(nMax); elN = new Float64Array(nMax);
    elM0 = new Float64Array(nMax);
    basisP = new Float64Array(nMax * 3); basisQ = new Float64Array(nMax * 3);
    incDeg = new Float32Array(nMax);
    constClass = new Uint8Array(nMax);
    altBin = new Uint8Array(nMax);
    perigeeKm = new Float32Array(nMax);
    stormBin = new Uint8Array(nMax);
    idxByNorad.clear();
    let n = 0;
    for (const o of objs) {
      const el = elemsFromRV(o[3], o[4], o[5], o[6], o[7], o[8]);
      if (!el) continue;                       // e >= 0.98 or a <= 6478: discard
      noradArr[n] = o[0];
      names[n] = String(o[1]);
      const t = o[2] | 0;
      typeCode[n] = (t >= 0 && t <= 3) ? t : 3;
      riskCode[n] = o[9] | 0;
      fleetRisk[n] = o[9] | 0;
      elA[n] = el.a; elE[n] = el.e; elB[n] = el.b; elN[n] = el.n; elM0[n] = el.M0;
      const j = 3 * n;
      basisP[j] = el.px; basisP[j + 1] = el.py; basisP[j + 2] = el.pz;
      basisQ[j] = el.qx; basisQ[j + 1] = el.qy; basisQ[j + 2] = el.qz;
      incDeg[n] = el.incDeg;
      constClass[n] = constellationOf(names[n]);           // fixed at load
      altBin[n] = altBinOf(el.a * (1 - el.e) - RE);        // perigee bin
      perigeeKm[n] = el.a * (1 - el.e) - RE;
      idxByNorad.set(o[0], n);
      n++;
    }
    count = n;
    rebinStorm();
    posX = new Float64Array(n); posY = new Float64Array(n); posZ = new Float64Array(n);
    posAX = new Float64Array(n); posAY = new Float64Array(n); posAZ = new Float64Array(n);
    posBX = new Float64Array(n); posBY = new Float64Array(n); posBZ = new Float64Array(n);
    posCX = new Float64Array(n); posCY = new Float64Array(n); posCZ = new Float64Array(n);
    scrX = new Float32Array(n); scrY = new Float32Array(n);
    vis = new Uint8Array(n);
    filtOk = new Uint8Array(n);
    baseLists = new Array(MAX_GROUPS);
    for (let g = 0; g < MAX_GROUPS; g++) baseLists[g] = new Int32Array(n);
    listElev = new Int32Array(n); listCrit = new Int32Array(n);
    epochIso = data.epoch;
    epochMs = Date.parse(data.epoch);
    simOffsetMs = pendingSimMs !== undefined ? pendingSimMs - epochMs : 0;
    pendingSimMs = undefined;          // pre-load setSimTime applies at the snapshot
    needAnchor = true;                 // first frame anchors both snapshots
    if (conjSet) applyRiskFromItems();
    applyFilter();                     // filter set before load applies now
    rebuildDrawLists();
    if (pendingSelect !== undefined) {
      const sel = pendingSelect;
      pendingSelect = undefined;
      applySelect(sel);
    }
    if (pendingEncounter !== undefined) {
      const enc = pendingEncounter;
      pendingEncounter = undefined;
      showEncounter(enc);
    }
  }

  /* --- conjunction-driven risk styling ------------------------------------- */

  function applyRiskFromItems() {
    /* The fleet snapshot's risk_code covers every active conjunction; the
       polled items are only the top page. Re-apply the snapshot baseline and
       let items raise levels on top of it - never zero the whole field. */
    riskCode.set(fleetRisk.subarray(0, count));
    for (const item of conjItems) {
      if (!item || (item.status && item.status !== 'active')) continue;
      const lvl = RISK_CODES[item.risk_level] || 0;
      if (!lvl) continue;
      const pi = item.primary ? idxByNorad.get(item.primary.norad_id) : undefined;
      const si = item.secondary ? idxByNorad.get(item.secondary.norad_id) : undefined;
      if (pi !== undefined && riskCode[pi] < lvl) riskCode[pi] = lvl;
      if (si !== undefined && riskCode[si] < lvl) riskCode[si] = lvl;
    }
  }

  /* Risk objects (elevated/critical) always land in the risk lists - blitted
     last, on top, in every color mode. The remaining base field is grouped by
     the active mode's classifier. 'type' grouping (payload / debris+unknown /
     rocket body, in that blit order) is byte-identical to the older lists. */
  function rebuildDrawLists() {
    nElev = 0; nCrit = 0;
    baseCounts.fill(0);
    const mode = colorMode;
    for (let i = 0; i < count; i++) {
      if (!filtOk[i]) continue;        // hidden by filter - not batched
      const r = riskCode[i];
      if (r >= 3) { listCrit[nCrit++] = i; continue; }
      if (r === 2) { listElev[nElev++] = i; continue; }
      let g;
      if (mode === 'type') {
        const t = typeCode[i];
        g = t === 0 ? 0 : (t === 2 ? 2 : 1);       // unknown shares the debris list
      } else if (mode === 'risk') g = 0;           // uniform dimmed field
      else if (mode === 'constellation') g = constClass[i];
      else if (mode === 'storm') g = stormBin[i];
      else g = altBin[i];
      baseLists[g][baseCounts[g]++] = i;
    }
  }

  /* blit color table + group blit order for the active mode. Runs on mode
     change and at init (readColors) only - never in the frame loop. Colors
     land in the persistent baseCols byte arrays (no reallocation). */
  function rebuildColorTables() {
    if (colorMode === 'risk') {
      baseOrder = ORDER_RISK;
      baseCols[0].set(colDim);                     // uniform dim: risk pops
    } else if (colorMode === 'constellation') {
      baseOrder = ORDER_CONST;
      for (let s = 0; s < 5; s++) parseColorInto(CONST_HEX[s], baseCols[s]);
      baseCols[5].set(colDebris);                  // OTHER
    } else if (colorMode === 'altitude') {
      baseOrder = ORDER_ALT;
      for (let b = 0; b < 5; b++) parseColorInto(ALT_HEX[b], baseCols[b]);
    } else if (colorMode === 'storm') {
      baseOrder = ORDER_STORM;
      for (let b = 0; b < 5; b++) parseColorInto(STORM_HEX[b], baseCols[b]);
    } else {                                       // 'type' - older colors
      baseOrder = ORDER_TYPE;
      baseCols[0].set(colPayload);
      baseCols[1].set(colDebris);
      baseCols[2].set(colRb);
    }
  }

  /* --- filter --------------------------------------------------------------- */

  /* Recompute per-object filter pass + the post-filter visible count. Runs on
     load, setFilter and setConjunctions (risk codes feed riskOnly) - never in
     the RAF loop. */
  function applyFilter() {
    for (let i = 0; i < count; i++) {
      filtOk[i] = (typeMask[typeCode[i]] && (!riskOnly || riskCode[i] > 0)) ? 1 : 0;
    }
    recountVisible();
  }

  /* Rendered-object count for status().visible: everything passing the filter,
     plus the selection/partner when the filter would hide them. */
  function recountVisible() {
    let v = 0;
    for (let i = 0; i < count; i++) v += filtOk[i];
    if (selIdx >= 0 && !filtOk[selIdx]) v++;
    if (partnerIdx >= 0 && partnerIdx !== selIdx && !filtOk[partnerIdx]) v++;
    visibleCount = v;
  }

  /* Selection and its partner stay visible/pickable regardless of the filter. */
  function passesFilter(i) {
    return filtOk[i] === 1 || i === selIdx || i === partnerIdx;
  }

  /* ==========================================================================
     Selection
     ======================================================================== */

  /* drop the encounter overlay (markers/connector/label + pinned partner). */
  function clearEncounter() {
    encActive = false;
    encSecIdx = -1;
    encLabel = '';
    encSepKm = NaN;
    encPriNorad = encSecNorad = null;
  }

  function applySelect(nor) {
    clearEncounter();                   // any select/deselect drops the TCA overlay
    if (!count) {                       // fleet not loaded yet - queue it
      pendingSelect = nor;
      return;
    }
    let i = -1;
    if (nor !== null && nor !== undefined) {
      const f = idxByNorad.get(Number(nor));
      i = f === undefined ? -1 : f;
    }
    if (i !== selIdx) selAnimStart = performance.now();
    selIdx = i;
    if (i >= 0) buildOrbit(i, selOrbit);
    updatePartner();
    recountVisible();                  // sel/partner count as visible when filtered out
    fireSelect();
  }

  /* Highest-risk active conjunction featuring the selection (items arrive
     pre-sorted by effective risk from the API) supplies the partner orbit. */
  function updatePartner() {
    if (encActive) {                    // encounter mode pins the secondary as partner
      partnerIdx = encSecIdx;           // (partnerOrbit already built by showEncounter)
      return;
    }
    partnerIdx = -1;
    if (selIdx < 0) return;
    const nor = noradArr[selIdx];
    for (const item of conjItems) {
      if (!item || (item.status && item.status !== 'active')) continue;
      const p = item.primary && item.primary.norad_id;
      const s = item.secondary && item.secondary.norad_id;
      const other = p === nor ? s : (s === nor ? p : null);
      if (other === null || other === undefined) continue;
      const i = idxByNorad.get(other);
      if (i !== undefined) {
        partnerIdx = i;
        buildOrbit(i, partnerOrbit);
        return;
      }
    }
  }

  function fireSelect() {
    if (!onSelectCb) return;
    if (selIdx < 0) {
      onSelectCb(null);
      return;
    }
    const i = selIdx;
    propagateOne(i, simOffsetMs / 1000, p3);
    const r = Math.hypot(p3[0], p3[1], p3[2]);
    onSelectCb({
      norad: noradArr[i],
      name: names[i],
      type: TYPE_NAMES[typeCode[i]],
      alt_km: r - RE,
      speed_kms: Math.sqrt(MU * (2 / r - 1 / elA[i])),
      elements: {
        inclination_deg: incDeg[i],
        period_min: TWO_PI / elN[i] / 60,
        apogee_km: elA[i] * (1 + elE[i]) - RE,
        perigee_km: elA[i] * (1 - elE[i]) - RE,
      },
    });
  }

  /* ==========================================================================
     Physics tick (decoupled from the render tick)
     ======================================================================== */

  /* full-catalog Kepler solve at sim offset simMsOff -> the given buffers */
  function solveSnapshot(simMsOff, ox, oy, oz) {
    const t = simMsOff / 1000;
    for (let i = 0; i < count; i++) {
      propagateOne(i, t, p3);
      ox[i] = p3[0]; oy[i] = p3[1]; oz[i] = p3[2];
    }
  }

  /* Auto-stretch the tick when a single solve is slow (keeps ticks rare
     instead of janking every one; lerp spans grow correspondingly). */
  function stretchCheck(solveMs) {
    if (solveMs > 50 && tickWallMs < TICK_MS_MAX) {
      tickWallMs = Math.min(TICK_MS_MAX, tickWallMs * 2);
    }
  }

  /* Begin pre-solving the next snapshot (C) one tick past B; it is advanced
     in per-frame chunks by advanceNextSolve so the full-catalog solve never
     lands on a single frame. */
  function startNextSolve() {
    simC = simB + tickWallMs * speed;
    cProgress = 0;
  }

  /* Solve one chunk (~1/SOLVE_CHUNKS of the catalog) of the C snapshot; cheap
     no-op once complete. A 100 ms tick spans ~6 frames at 60 Hz, so the solve
     finishes with a frame to spare. */
  function advanceNextSolve() {
    if (cProgress >= count) return;
    const t = simC / 1000;
    let end = cProgress + Math.ceil(count / SOLVE_CHUNKS);
    if (end > count) end = count;
    for (let i = cProgress; i < end; i++) {
      propagateOne(i, t, p3);
      posCX[i] = p3[0]; posCY[i] = p3[1]; posCZ[i] = p3[2];
    }
    cProgress = end;
  }

  /* Re-solve both snapshots at the current sim time (speed change, tab
     resume, overrun). Rendering after this always lerps with alpha in [0,1). */
  function anchorSnapshots() {
    const t0 = performance.now();
    simA = simOffsetMs;
    simB = simA + tickWallMs * speed;
    solveSnapshot(simA, posAX, posAY, posAZ);
    solveSnapshot(simB, posBX, posBY, posBZ);
    startNextSolve();
    stretchCheck((performance.now() - t0) / 2);
  }

  /* Runs every frame. Between boundaries it advances the amortized pre-solve
     of the next snapshot; crossing simB just rotates buffers (A <- B <- C) -
     the full-catalog solve never runs synchronously on a boundary frame
     unless the pre-solve was outrun (long frame gap), which is also when
     stretchCheck may grow the tick. */
  function physicsTick() {
    if (!count) return;
    if (needAnchor) {
      needAnchor = false;
      anchorSnapshots();
      return;
    }
    const span = simB - simA;
    const alpha = span > 0 ? (simOffsetMs - simA) / span : 2;
    if (alpha < 0 || alpha >= 2) {     // overrun/rewind: never extrapolate
      anchorSnapshots();
      return;
    }
    if (alpha >= 1) {                  // boundary: rotate A <- B <- C
      if (cProgress < count) {         // pre-solve outrun - finish the rest now
        const t0 = performance.now();
        while (cProgress < count) advanceNextSolve();
        stretchCheck(performance.now() - t0);
      }
      let t = posAX; posAX = posBX; posBX = posCX; posCX = t;
      t = posAY; posAY = posBY; posBY = posCY; posCY = t;
      t = posAZ; posAZ = posBZ; posBZ = posCZ; posCZ = t;
      simA = simB;
      simB = simC;
      startNextSolve();
      return;
    }
    advanceNextSolve();
  }

  /* ==========================================================================
     Rendering
     ======================================================================== */

  function ensureSize() {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (!cw || !ch) return false;
    const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
    if (canvas.width !== w || canvas.height !== h || fDpr !== dpr) {
      canvas.width = w;
      canvas.height = h;
      earthCanvas.width = w;
      earthCanvas.height = h;
      dotCanvas.width = w;
      dotCanvas.height = h;
      dotImage = dotCtx.createImageData(w, h);   // reused until the next resize
      dotData = dotImage.data;
      devW = w;
      devH = h;
      /* persistent heat-bin buffer - canvas-size dependent, realloc HERE
         (resize) only, reused (fill(0)) every heat frame. */
      heatBW = Math.ceil(w / HEAT_CELL);
      heatBH = Math.ceil(h / HEAT_CELL);
      const nb = heatBW * heatBH;
      if (!heatBins || heatBins.length !== nb) heatBins = new Uint16Array(nb);
      earthDirty = true;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fDpr = dpr;
    fCw = cw; fCh = ch;
    return true;
  }

  /* Stroke a set of surface polylines, breaking wherever a vertex crosses to
     the back hemisphere (depth <= 0). Draws into the given context. */
  function strokeLineSet(tc, xyz, starts, color, earthFixed) {
    if (!xyz) return;
    tc.strokeStyle = color;
    tc.lineWidth = 1;
    tc.lineJoin = 'round';
    tc.beginPath();
    const nLines = starts.length - 1;
    for (let l = 0; l < nLines; l++) {
      let pen = false;
      const end = starts[l + 1];
      for (let k = starts[l]; k < end; k++) {
        const j = 3 * k;
        if (earthFixed) projEarth(xyz[j], xyz[j + 1], xyz[j + 2]);
        else projEci(xyz[j], xyz[j + 1], xyz[j + 2]);
        if (pjD > 0) {
          if (pen) tc.lineTo(pjX, pjY);
          else tc.moveTo(pjX, pjY);
          pen = true;
        } else {
          pen = false;
        }
      }
    }
    tc.stroke();
  }

  /* Night-side overlay: limb semicircle on the anti-sun side closed by the
     projected terminator ellipse. Flat rgba, clipped to the disk. */
  function drawTerminator(tc) {
    sunDirEci(epochMs + simOffsetMs, sunV);
    const x1 = sunV[0] * fCosS - sunV[1] * fSinS;
    const y1 = sunV[0] * fSinS + sunV[1] * fCosS;
    const y2 = y1 * fCosT - sunV[2] * fSinT;
    const z2 = y1 * fSinT + sunV[2] * fCosT;
    const dSun = -y2;                  // sun depth toward viewer
    const px = x1, py = -z2;           // projected sun direction, canvas coords
    const len = Math.hypot(px, py);
    tc.save();
    tc.beginPath();
    tc.arc(fCx, fCy, fRs, 0, TWO_PI);
    tc.clip();
    tc.fillStyle = NIGHT_RGBA;
    if (len < 1e-6) {
      if (dSun < 0) tc.fillRect(fCx - fRs, fCy - fRs, fRs * 2, fRs * 2);
    } else {
      tc.translate(fCx, fCy);
      tc.rotate(Math.atan2(py, px));   // sun now along +x
      tc.beginPath();
      tc.arc(0, 0, fRs, Math.PI / 2, Math.PI * 1.5, false);           // anti-sun limb
      tc.ellipse(0, 0, fRs * Math.abs(dSun), fRs, 0,
        Math.PI * 1.5, Math.PI / 2, dSun > 0);                        // terminator
      tc.closePath();
      tc.fill();
    }
    tc.restore();
  }

  /* textured SURFACE is available (module loaded, textures decoded,
     toggle on). Typeof-guarded - pages without earthtex.js stay wireframe. */
  function surfaceReady() {
    return surfaceOn && typeof KEarthTexture !== 'undefined' && KEarthTexture.ready();
  }

  /* Cached Earth layer into the offscreen earth canvas. Re-run only when the
     view moved, the canvas resized, or GMST drifted > 0.2° - cadence
     identical in both modes.
     WIREFRAME: disk -> terminator -> graticule -> coastlines -> limb.
     SURFACE: textured hemisphere (its day/night blend is the terminator
     and the texture carries the coastlines) -> graticule -> limb. Always
     full quality - earthtex's LUT renderer makes drag re-renders a cheap
     per-frame resample, so there is no reduced-quality drag mode and no
     wireframe drag fallback. */
  function renderEarth(g) {
    const ec = earthCtx;
    ec.setTransform(fDpr, 0, 0, fDpr, 0, 0);
    ec.clearRect(0, 0, fCw, fCh);
    if (atmosphereOn) drawAtmosphere(ec, false);   // outer halo, under the disk
    let textured = false;
    if (surfaceReady()) {
      sunDirEci(epochMs + simOffsetMs, sunV);        // same sun the terminator uses
      textured = KEarthTexture.drawHemisphere(ec, {
        cx: fCx, cy: fCy, radiusPx: fRs, spin, tilt, gmstRad: g - spin,
        sunDirTEME: sunV,
      });
    }
    if (!textured) {
      ec.fillStyle = C.disk;
      ec.beginPath();
      ec.arc(fCx, fCy, fRs, 0, TWO_PI);
      ec.fill();
      drawTerminator(ec);
      /* the outline map carries a graticule; the imagery does not need one */
      strokeLineSet(ec, gratXYZ, gratStarts, C.grid, true);
      strokeLineSet(ec, eqXYZ, eqStarts, C.gridStrong, true);
      strokeLineSet(ec, coastXYZ, coastStarts, C.coast, true);
    }
    drawAurora(ec, g);                             // live aurora, earth-fixed
    if (atmosphereOn) drawAtmosphere(ec, true);    // limb haze, over the disk
    else {
      ec.strokeStyle = C.gridStrong;
      ec.lineWidth = 1;
      ec.beginPath();
      ec.arc(fCx, fCy, fRs, 0, TWO_PI);
      ec.stroke();
    }
    eSpin = spin; eTilt = tilt; eZoom = zoom;
    eW = fCw; eH = fCh; eDpr = fDpr; eGmst = g;
    earthDirty = false;
  }

  /* Soft atmospheric rim: an outer halo fading into space (under the disk)
     and a thin limb haze on the disk edge (over it). Gradients are built per
     earth-layer render only - never per frame. */
  function drawAtmosphere(ec, inner) {
    const r = fRs;
    if (!(r > 4)) return;
    const base = swWarm > 0 ? warmRgb(C.atmoRgb, swWarm) : C.atmoRgb;
    if (!inner) {
      const gr = ec.createRadialGradient(fCx, fCy, r * 0.985, fCx, fCy, r * 1.16);
      gr.addColorStop(0, `rgba(${base}, 0.40)`);
      gr.addColorStop(0.16, `rgba(${base}, 0.17)`);
      gr.addColorStop(0.5, `rgba(${base}, 0.05)`);
      gr.addColorStop(1, `rgba(${base}, 0)`);
      ec.fillStyle = gr;
      ec.beginPath();
      ec.arc(fCx, fCy, r * 1.16, 0, TWO_PI);
      ec.fill();
      return;
    }
    const gr = ec.createRadialGradient(fCx, fCy, r * 0.80, fCx, fCy, r);
    gr.addColorStop(0, `rgba(${base}, 0)`);
    gr.addColorStop(0.72, `rgba(${base}, 0.04)`);
    gr.addColorStop(0.94, `rgba(${base}, 0.20)`);
    gr.addColorStop(1, `rgba(${base}, 0.50)`);
    ec.fillStyle = gr;
    ec.beginPath();
    ec.arc(fCx, fCy, r, 0, TWO_PI);
    ec.fill();
  }

  /* Lerp the whole catalog between the two physics snapshots and project;
     cache screen coords for picking (dots are drawn from these exact values).
     Pure multiply-add - no trig, no allocation. */
  function interpSats() {
    const span = simB - simA;
    const w = span > 0 ? (simOffsetMs - simA) / span : 0;
    const ax = posAX, ay = posAY, az = posAZ;
    const bx = posBX, by = posBY, bz = posBZ;
    const cs = fCosS, ss = fSinS, ct = fCosT, st = fSinT;
    const cx = fCx, cy = fCy, sc = fScale, rs2 = fRs * fRs;
    for (let i = 0; i < count; i++) {
      const x = ax[i] + w * (bx[i] - ax[i]);
      const y = ay[i] + w * (by[i] - ay[i]);
      const z = az[i] + w * (bz[i] - az[i]);
      posX[i] = x; posY[i] = y; posZ[i] = z;
      const x1 = x * cs - y * ss;
      const y1 = x * ss + y * cs;
      const y2 = y1 * ct - z * st;
      const z2 = y1 * st + z * ct;
      const sx = cx + sc * x1;
      const sy = cy - sc * z2;
      scrX[i] = sx; scrY[i] = sy;
      if (sx !== sx) {                 // NaN guard (matches old finiteness check)
        vis[i] = 0;
        continue;
      }
      const dx = sx - cx, dy = sy - cy;
      vis[i] = (y2 > 0 && dx * dx + dy * dy < rs2) ? 0 : 1;   // occluded by disk
    }
  }

  /* Round dot sprites: per-size coverage masks (0..1), built lazily once per
     size - a disc of the given DEVICE-px diameter, anti-aliased on its edge.
     Blocks <= 2 px stay solid (a 2x2 dot is already as round as it gets). */
  const MASKS = [];
  function maskFor(size) {
    let m = MASKS[size];
    if (m) return m;
    m = new Float32Array(size * size);
    const c = (size - 1) / 2, r = size / 2;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (size <= 2) { m[y * size + x] = 1; continue; }
        let cov = 0;                     // 4x4 supersampled coverage
        for (let sy = 0; sy < 4; sy++) {
          for (let sx = 0; sx < 4; sx++) {
            const dx = x - c + (sx - 1.5) / 4, dy = y - c + (sy - 1.5) / 4;
            if (dx * dx + dy * dy <= r * r) cov++;
          }
        }
        m[y * size + x] = cov / 16;
      }
    }
    MASKS[size] = m;
    return m;
  }

  /* Blit one draw list into the reused ImageData as sizexsize device-pixel
     sprites with explicit R,G,B,A byte writes (no 32-bit endianness games).
     Later batches overwrite earlier ones - risk colors land on top; a
     sprite's soft corners never erase a stronger pixel under them. */
  function blitBatch(list, n, col, size) {
    if (!n) return;
    const W = devW, H = devH, dpr = fDpr, data = dotData;
    const r = col[0], g = col[1], b = col[2], a = col[3];
    const mask = maskFor(size);
    const rowSkip = (W - size) * 4;
    const off = size >> 1;             // center the block on the projected point
    for (let k = 0; k < n; k++) {
      const i = list[k];
      if (!vis[i]) continue;
      const x0 = Math.round(scrX[i] * dpr) - off;
      const y0 = Math.round(scrY[i] * dpr) - off;
      if (x0 < 0 || y0 < 0 || x0 + size > W || y0 + size > H) continue;
      let p = (y0 * W + x0) << 2;
      let q = 0;
      for (let row = 0; row < size; row++) {
        for (let c = 0; c < size; c++) {
          const cov = mask[q++];
          if (cov >= 1) {
            data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = a;
          } else if (cov > 0) {
            const aa = a * cov;
            if (aa > data[p + 3]) {
              data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = aa;
            }
          }
          p += 4;
        }
        p += rowSkip;
      }
    }
  }

  /* Critical objects get a soft halo ring on the vector context (there are
     only ever a handful, so arcs are cheap). */
  function drawCritHalos() {
    if (!nCrit) return;
    ctx.strokeStyle = C.critical;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k < nCrit; k++) {
      const i = listCrit[k];
      if (!vis[i]) continue;
      ctx.moveTo(scrX[i] + 6, scrY[i]);
      ctx.arc(scrX[i], scrY[i], 6, 0, TWO_PI);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /* Selected / partner orbit polyline; segments behind the disk are culled the
     same way satellites are (depth < 0 inside the limb). */
  function drawOrbit(buf, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const rs2 = fRs * fRs;
    let pen = false;
    for (let k = 0; k <= 128; k++) {
      const j = 3 * (k & 127);
      projEci(buf[j], buf[j + 1], buf[j + 2]);
      const dx = pjX - fCx, dy = pjY - fCy;
      if (pjD < 0 && dx * dx + dy * dy < rs2) {
        pen = false;
      } else {
        if (pen) ctx.lineTo(pjX, pjY);
        else ctx.moveTo(pjX, pjY);
        pen = true;
      }
    }
    ctx.stroke();
  }

  /* Selection marker: a filled core plus a ring that settles in from a
     wider radius when the selection changes (~320 ms ease-out), then a slow,
     quiet breathing halo. */
  function drawRing(i, color, now) {
    if (!vis[i]) return;
    const x = scrX[i], y = scrY[i];
    const t = Math.min(1, Math.max(0, (now - selAnimStart) / 320));
    const e = 1 - Math.pow(1 - t, 3);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, TWO_PI);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.35 + 0.65 * e;
    ctx.beginPath();
    ctx.arc(x, y, 7 + 14 * (1 - e), 0, TWO_PI);
    ctx.stroke();
    if (t >= 1) {
      const b = 0.5 + 0.5 * Math.sin(now / 520);
      ctx.globalAlpha = 0.10 + 0.12 * b;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(x, y, 11 + 2 * b, 0, TWO_PI);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /* --- label layout: a tiny per-frame obstacle list ------------------------ */
  function lblHit(x0, y0, x1, y1) {
    for (let k = 0; k < lblN; k++) {
      const o = k * 4;
      if (x0 < lblBox[o + 2] && x1 > lblBox[o] && y0 < lblBox[o + 3] && y1 > lblBox[o + 1]) return true;
    }
    return false;
  }
  function lblAdd(x0, y0, x1, y1) {
    if (lblN >= LBL_MAX) return;
    const o = lblN * 4;
    lblBox[o] = x0; lblBox[o + 1] = y0; lblBox[o + 2] = x1; lblBox[o + 3] = y1;
    lblN++;
  }
  /* First candidate offset (dx, dy from the anchor; text is bottom-aligned
     at y, left- or right-aligned by the sign of dx, centered when dx = 0)
     whose box stays on the canvas and clear of every registered box.
     Writes lpX/lpY/lpAlign/lpBox; returns the candidate index (-1: none fit,
     the first candidate is used). */
  let lpX = 0, lpY = 0, lpAlign = 'left';
  const lpBox = new Float64Array(4);
  const LBL_H = 15;
  function placeLabel(ax, ay, w, cands) {
    let pick = -1;
    for (let c = 0; c < cands.length; c += 2) {
      const dx = cands[c], dy = cands[c + 1];
      const x0 = dx > 0 ? ax + dx : dx < 0 ? ax + dx - w : ax - w / 2;
      const y1 = ay + dy + 2, y0 = y1 - LBL_H;
      if (x0 < 2 || x0 + w > fCw - 2 || y0 < 2 || y1 > fCh - 2) continue;
      if (lblHit(x0 - 2, y0, x0 + w + 2, y1)) continue;
      pick = c;
      break;
    }
    const c = pick < 0 ? 0 : pick;
    const dx = cands[c], dy = cands[c + 1];
    lpAlign = dx > 0 ? 'left' : dx < 0 ? 'right' : 'center';
    lpX = ax + dx;
    lpY = ay + dy;
    const x0 = dx > 0 ? lpX : dx < 0 ? lpX - w : lpX - w / 2;
    lpBox[0] = x0; lpBox[1] = lpY + 2 - LBL_H; lpBox[2] = x0 + w; lpBox[3] = lpY + 2;
    lblAdd(lpBox[0] - 2, lpBox[1], lpBox[2] + 2, lpBox[3]);
    return pick;
  }
  const NAME_CANDS = [11, -9, 11, 21, -11, -9, -11, 21, 14, -26, 14, 38, -14, -26, -14, 38];
  const ENC_CANDS = [8, -8, 8, 22, -8, -8, -8, 22, 0, -30, 0, 44, 30, -34, 30, 48, -30, -34, -30, 48,
    0, -56, 0, 70, 60, -60, -60, -60, 60, 74, -60, 74];

  /* Direct name label off the dot, placed clear of the other labels and
     rings; suppressed while the hover tip already names the same object. */
  function drawLabel(i) {
    if (!vis[i] || i === hoverIdx) return;
    ctx.font = labelFont;
    ctx.fillStyle = C.label;
    ctx.textBaseline = 'bottom';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
    ctx.shadowBlur = 6;
    const w = ctx.measureText(names[i]).width;
    placeLabel(scrX[i], scrY[i], w, NAME_CANDS);
    ctx.textAlign = lpAlign;
    ctx.fillText(names[i], lpX, lpY);
    ctx.shadowBlur = 0;
    ctx.shadowColor = 'transparent';
    ctx.textAlign = 'left';
  }

  /* Occlusion test for the last projEci output: behind the disk and inside
     the limb - the exact rule satellites use. */
  function projOccluded() {
    const dx = pjX - fCx, dy = pjY - fCy;
    return pjD < 0 && dx * dx + dy * dy < fRs * fRs;
  }

  function drawEncMarker(x, y, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, TWO_PI);        // 4px ring marker
    ctx.stroke();
  }

  /* encounter overlay: 4px ring markers at both TCA positions (primary
     --k-data-1, secondary --k-data-2), a 1px --k-text-2 connector, and the
     'TCA hh:mm:ssZ' mono label at the 3-D midpoint. The TCA positions are
     fixed ECI points (two-body elements are valid at any t), computed once in
     showEncounter; everything culls behind the disk like satellites do. */
  function drawEncounter() {
    /* connector - subdivided in 3-D so the straight chord clips at the limb */
    ctx.strokeStyle = C.link;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    let pen = false;
    for (let k = 0; k <= ENC_SEGS; k++) {
      const w = k / ENC_SEGS;
      projEci(encP[0] + w * (encS[0] - encP[0]),
        encP[1] + w * (encS[1] - encP[1]),
        encP[2] + w * (encS[2] - encP[2]));
      if (projOccluded()) {
        pen = false;
      } else {
        if (pen) ctx.lineTo(pjX, pjY);
        else ctx.moveTo(pjX, pjY);
        pen = true;
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
    projEci(encP[0], encP[1], encP[2]);
    if (!projOccluded()) drawEncMarker(pjX, pjY, C.select);
    projEci(encS[0], encS[1], encS[2]);
    if (!projOccluded()) drawEncMarker(pjX, pjY, C.partner);
    projEci(encM[0], encM[1], encM[2]);
    if (!projOccluded()) {
      const ax = pjX, ay = pjY;
      ctx.font = labelFont;
      const w = ctx.measureText(encLabel).width;
      const pick = placeLabel(ax, ay, w, ENC_CANDS);
      /* displaced past the nearest slots: a thin leader ties it to the point */
      if (pick >= 8) {
        const lx = Math.max(lpBox[0], Math.min(ax, lpBox[2]));
        const ly = ay < lpBox[1] ? lpBox[1] : lpBox[3];
        ctx.strokeStyle = C.link;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(lx, ly);
        ctx.stroke();
      }
      ctx.fillStyle = C.label;
      ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
      ctx.shadowBlur = 6;
      ctx.textBaseline = 'bottom';
      ctx.textAlign = lpAlign;
      ctx.fillText(encLabel, lpX, lpY);
      ctx.shadowBlur = 0;
      ctx.shadowColor = 'transparent';
      ctx.textAlign = 'left';
    }
  }

  /* heat layer: 16-device-px screen bins over the dots the blitter will
     draw this frame (filter-visible && not disk-occluded), washed as flat
     alpha cells in the sequential blue UNDER the dot field. Reuses the
     persistent Uint16 bin buffer + the module-init style strings - zero
     per-frame allocation. Alpha = 0.025 per dot, capped at 0.35 (level 14):
     an absolute mapping, so brightness is comparable across frames/zoom. */
  function drawHeat() {
    const bins = heatBins, bw = heatBW, bh = heatBH;
    bins.fill(0);
    const dpr = fDpr, W = devW, H = devH;
    for (let i = 0; i < count; i++) {
      if (!vis[i] || !filtOk[i]) continue;
      const sx = scrX[i] * dpr, sy = scrY[i] * dpr;
      if (!(sx >= 0 && sx < W && sy >= 0 && sy < H)) continue;   // NaN-safe
      const b = ((sy / HEAT_CELL) | 0) * bw + ((sx / HEAT_CELL) | 0);
      if (bins[b] < 65535) bins[b]++;
    }
    const cell = HEAT_CELL / dpr;      // ctx is CSS-px scaled
    for (let by = 0; by < bh; by++) {
      const y = by * cell, row = by * bw;
      for (let bx = 0; bx < bw; bx++) {
        const v = bins[row + bx];
        if (!v) continue;
        ctx.fillStyle = HEAT_STYLES[v < HEAT_LEVELS ? v : HEAT_LEVELS];
        ctx.fillRect(bx * cell, y, cell, cell);
      }
    }
  }

  function escHtml(v) {
    return String(v).replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  let tipIdx = -1, tipAltKm = NaN;     // tip content cache: rebuild on change only

  function refreshTip() {
    if (!tipEl) return;
    if (hoverIdx >= 0 && (hoverIdx >= count || !vis[hoverIdx] || !passesFilter(hoverIdx))) hoverIdx = -1;
    if (hoverIdx < 0) {
      tipEl.style.display = 'none';
      tipIdx = -1;
      return;
    }
    const i = hoverIdx;
    const alt = Math.round(Math.hypot(posX[i], posY[i], posZ[i]) - RE);
    if (i !== tipIdx || alt !== tipAltKm) {
      tipIdx = i;
      tipAltKm = alt;
      tipEl.innerHTML =
        `<b class="globe-tip__name">${escHtml(names[i])}</b>` +
        '<span class="globe-tip__sep"> · </span>' +
        `<span class="globe-tip__meta">${TYPE_LABELS[typeCode[i]]} · NORAD ${noradArr[i]}` +
        ` · ${escHtml(FMT.km(alt, 0))} up</span>`;
    }
    tipEl.style.display = 'block';
    const bx = canvas.offsetLeft, by = canvas.offsetTop;
    let lx = scrX[i] + 12, ly = scrY[i] + 12;
    if (lx + tipEl.offsetWidth > fCw - 4) lx = scrX[i] - 12 - tipEl.offsetWidth;
    if (ly + tipEl.offsetHeight > fCh - 4) ly = scrY[i] - 12 - tipEl.offsetHeight;
    tipEl.style.left = `${Math.round(bx + lx)}px`;
    tipEl.style.top = `${Math.round(by + ly)}px`;
  }

  function draw() {
    if (!ensureSize()) return;
    const now = performance.now();
    const simMs = epochMs + simOffsetMs;
    fCosT = Math.cos(tilt); fSinT = Math.sin(tilt);
    fCosS = Math.cos(spin); fSinS = Math.sin(spin);
    const g = spin + gmstRad(simMs);
    fCosG = Math.cos(g); fSinG = Math.sin(g);
    /* frame the Earth in the region not covered by floating UI (setViewInset);
       an inset that would leave too little room is ignored */
    let il = inset.left, ir = inset.right, it = inset.top, ib = inset.bottom;
    if (fCw - il - ir < 240) { il = 0; ir = 0; }
    if (fCh - it - ib < 240) { it = 0; ib = 0; }
    const vw = fCw - il - ir, vh = fCh - it - ib;
    fScale = (Math.min(vw, vh) / 2 / FRAME_KM) * zoom;
    fRs = RE * fScale;
    fCx = il + vw / 2; fCy = it + vh / 2;

    /* earth: cached layer, re-rendered only when its inputs moved (drags
       invalidate through spin/tilt like any other view change - it renders
       them at full quality). while auto-spinning over the textured
       surface, sub-threshold spin steps fold into the GMST_EPS check
       (g = spin + gmst) instead of invalidating every frame - the same 0.2°
       quantization GMST drift already gets. */
    const spinMoved = (autoSpinRad !== 0 && !dragging && surfaceReady())
      ? false
      : spin !== eSpin;
    if (earthDirty
        || spinMoved || tilt !== eTilt || zoom !== eZoom
        || fCw !== eW || fCh !== eH || fDpr !== eDpr
        || Math.abs(g - eGmst) > GMST_EPS) {
      renderEarth(g);
    }
    ctx.clearRect(0, 0, fCw, fCh);
    ctx.drawImage(earthCanvas, 0, 0, fCw, fCh);

    /* satellites: lerp + project, then pixel-blit the dot field */
    interpSats();
    if (heatOn && heatBins) drawHeat();     // density wash UNDER the dots
    if (selIdx >= 0 || partnerIdx >= 0) {
      /* orbits fade in with the selection ring */
      const fade = Math.min(1, Math.max(0, (now - selAnimStart) / 400));
      if (partnerIdx >= 0) {
        ctx.globalAlpha = 0.45 * fade;
        drawOrbit(partnerOrbit, C.partner);
      }
      if (selIdx >= 0) {
        ctx.globalAlpha = 0.9 * fade;
        drawOrbit(selOrbit, C.select);
      }
      ctx.globalAlpha = 1;
    }
    if (dotData && baseLists) { /* baseLists lands with the fleet snapshot -
         guard the pre-load frames (dotData exists as soon as the canvas
         is sized, several seconds earlier) */
      dotData.fill(0);
      /* block sizes are DEVICE pixels - scale by dpr so the contract's
         2x2 / 3x3 CSS-px dots hold on retina displays */
      const szDot = Math.max(2, Math.round(2 * fDpr));
      const szElev = Math.max(3, Math.round(3.5 * fDpr));
      const szCrit = Math.max(4, Math.round(4.5 * fDpr));
      for (let k = 0; k < baseOrder.length; k++) {   // mode's base groups
        const g = baseOrder[k];
        blitBatch(baseLists[g], baseCounts[g], baseCols[g], szDot);
      }
      blitBatch(listElev, nElev, colElev, szElev);  // risk last, on top - every mode
      blitBatch(listCrit, nCrit, colCrit, szCrit);
      dotCtx.putImageData(dotImage, 0, 0);      // putImageData never composites - 
      ctx.drawImage(dotCanvas, 0, 0, fCw, fCh); // drawImage does the alpha blend
      drawCritHalos();
    }
    /* labels: rings and encounter markers are obstacles first, then the
       selection's name, the partner's, and the encounter caption place
       themselves around each other */
    lblN = 0;
    if (partnerIdx >= 0 && vis[partnerIdx]) {
      lblAdd(scrX[partnerIdx] - 9, scrY[partnerIdx] - 9, scrX[partnerIdx] + 9, scrY[partnerIdx] + 9);
    }
    if (selIdx >= 0 && vis[selIdx]) {
      lblAdd(scrX[selIdx] - 9, scrY[selIdx] - 9, scrX[selIdx] + 9, scrY[selIdx] + 9);
    }
    if (encActive) {
      projEci(encP[0], encP[1], encP[2]);
      if (!projOccluded()) lblAdd(pjX - 6, pjY - 6, pjX + 6, pjY + 6);
      projEci(encS[0], encS[1], encS[2]);
      if (!projOccluded()) lblAdd(pjX - 6, pjY - 6, pjX + 6, pjY + 6);
    }
    if (partnerIdx >= 0) drawRing(partnerIdx, C.partner, now);
    if (selIdx >= 0) {
      drawRing(selIdx, C.select, now);
      drawLabel(selIdx);
    }
    if (partnerIdx >= 0) drawLabel(partnerIdx);
    if (encActive) drawEncounter();
    refreshTip();
  }

  /* ==========================================================================
     Loop + interaction
     ======================================================================== */

  function frame(now) {
    rafId = requestAnimationFrame(frame);
    const dt = now - lastNow;
    lastNow = now;
    simOffsetMs += dt * speed;
    if (autoSpinRad !== 0 && dt > 0) { // wall mode: yaw about the vertical axis
      spin += autoSpinRad * (dt / 1000);
      if (spin > TWO_PI || spin < -TWO_PI) spin %= TWO_PI;  // float hygiene, days-long walls
    }
    if (zoomTarget === zoomTarget && dt > 0) {   // eased zoomBy() (NaN = idle)
      const k = Math.min(1, dt / 110);
      zoom += (zoomTarget - zoom) * k;
      if (Math.abs(zoomTarget - zoom) < 0.002) { zoom = zoomTarget; zoomTarget = NaN; }
    }
    if (fly) {                         // eased "fly to" (public select / encounter)
      const t = Math.min(1, (now - fly.start) / fly.dur);
      const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      spin = fly.s0 + fly.ds * e;
      tilt = fly.t0 + fly.dt * e;
      if (t >= 1) {
        fly = null;
        spin %= TWO_PI;
      }
    }
    physicsTick();
    draw();
  }

  function startLoop() {
    if (running) return;
    running = true;
    lastNow = performance.now();
    needAnchor = true;                 // tab-resume: re-anchor, never extrapolate
    rafId = requestAnimationFrame(frame);
  }

  function stopLoop() {
    if (!running) return;
    running = false;
    cancelAnimationFrame(rafId);
  }

  function pickAt(mx, my) {
    let best = -1, bestD = PICK_R2;
    for (let i = 0; i < count; i++) {
      if (!vis[i] || !passesFilter(i)) continue;
      const dx = scrX[i] - mx, dy = scrY[i] - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  /* handler references kept so destroy() can unbind every listener */
  const handlers = {
    pointerdown(e) {
      if (e.button !== 0) return;
      canvas.setPointerCapture(e.pointerId);
      fly = null;                      // the user takes the view back
      dragging = true;
      dragMoved = 0;
      downX = lastPX = e.clientX;
      downY = lastPY = e.clientY;
      canvas.style.cursor = 'grabbing';
    },
    pointermove(e) {
      if (dragging) {
        const dx = e.clientX - lastPX, dy = e.clientY - lastPY;
        lastPX = e.clientX;
        lastPY = e.clientY;
        dragMoved = Math.max(dragMoved, Math.hypot(e.clientX - downX, e.clientY - downY));
        spin += dx / fRs;                       // surface-locked drag
        tilt = Math.max(-TILT_MAX, Math.min(TILT_MAX, tilt + dy / fRs));
        return;
      }
      hoverIdx = pickAt(e.offsetX, e.offsetY);
      canvas.style.cursor = hoverIdx >= 0 ? 'pointer' : 'grab';
    },
    pointerup(e) {
      if (!dragging) return;
      dragging = false;
      canvas.style.cursor = 'grab';
      /* no release repaint - drag frames already rendered full quality */
      if (dragMoved < CLICK_SLOP_PX) {          // click, not drag
        const i = pickAt(e.offsetX, e.offsetY);
        applySelect(i >= 0 ? noradArr[i] : null);
      }
    },
    pointercancel() {
      dragging = false;
      canvas.style.cursor = 'grab';
    },
    pointerleave() {
      if (!dragging) hoverIdx = -1;
    },
    wheel(e) {
      /* wheelZoom 'modifier' (page-embedded hero): plain wheel scrolls the
         page; ctrl/⌘ + wheel - and trackpad pinch, which arrives as a
         ctrl-wheel - zooms. Default 'always' keeps the original behavior. */
      if (wheelMode === 'modifier' && !(e.ctrlKey || e.metaKey)) {
        if (onWheelHint) onWheelHint();
        return;
      }
      e.preventDefault();
      zoomTarget = NaN;
      zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * Math.exp(-e.deltaY * 0.0012)));
    },
    visibilitychange() {
      if (document.hidden) stopLoop();
      else if (!appPaused) startLoop();
    },
  };

  function bindEvents() {
    canvas.addEventListener('pointerdown', handlers.pointerdown);
    canvas.addEventListener('pointermove', handlers.pointermove);
    canvas.addEventListener('pointerup', handlers.pointerup);
    canvas.addEventListener('pointercancel', handlers.pointercancel);
    canvas.addEventListener('pointerleave', handlers.pointerleave);
    canvas.addEventListener('wheel', handlers.wheel, { passive: false });
    document.addEventListener('visibilitychange', handlers.visibilitychange);
  }

  function unbindEvents() {
    canvas.removeEventListener('pointerdown', handlers.pointerdown);
    canvas.removeEventListener('pointermove', handlers.pointermove);
    canvas.removeEventListener('pointerup', handlers.pointerup);
    canvas.removeEventListener('pointercancel', handlers.pointercancel);
    canvas.removeEventListener('pointerleave', handlers.pointerleave);
    canvas.removeEventListener('wheel', handlers.wheel, { passive: false });
    document.removeEventListener('visibilitychange', handlers.visibilitychange);
  }

  /* --- tokens -------------------------------------------------------------- */

  /* css color string -> RGBA bytes (hex #rgb/#rgba/#rrggbb/#rrggbbaa or
     rgb()/rgba()); explicit byte channels for the pixel blitter. */
  function parseColorInto(str, out) {
    const s = String(str).trim();
    out[0] = 255; out[1] = 255; out[2] = 255; out[3] = 255;
    if (s[0] === '#') {
      const h = s.slice(1);
      if (h.length === 3 || h.length === 4) {
        out[0] = parseInt(h[0] + h[0], 16);
        out[1] = parseInt(h[1] + h[1], 16);
        out[2] = parseInt(h[2] + h[2], 16);
        if (h.length === 4) out[3] = parseInt(h[3] + h[3], 16);
      } else if (h.length === 6 || h.length === 8) {
        out[0] = parseInt(h.slice(0, 2), 16);
        out[1] = parseInt(h.slice(2, 4), 16);
        out[2] = parseInt(h.slice(4, 6), 16);
        if (h.length === 8) out[3] = parseInt(h.slice(6, 8), 16);
      }
    } else {
      const m = s.match(/[\d.]+/g);
      if (m && m.length >= 3) {
        out[0] = +m[0]; out[1] = +m[1]; out[2] = +m[2];
        out[3] = m.length >= 4 ? Math.round(+m[3] <= 1 ? +m[3] * 255 : +m[3]) : 255;
      }
    }
  }

  /* Space palette: --k-globe-* tokens resolved on the canvas element (a page
     scopes them onto its stage), else the SPACE defaults. Theme tokens are
     deliberately not used for the space itself - the globe stays dark in
     both themes. Re-run on KTheme.onChange. */
  function readColors() {
    const cs = getComputedStyle(canvas || document.documentElement);
    const token = (name, fallback) => (cs.getPropertyValue(name) || '').trim() || fallback;
    C.payload = token('--k-globe-payload', SPACE.payload);
    C.debris = token('--k-globe-debris', SPACE.debris);
    C.rb = token('--k-globe-rb', SPACE.rb);
    C.dim = token('--k-globe-dim', SPACE.dim);
    C.elevated = token('--k-globe-elevated', SPACE.elevated);
    C.critical = token('--k-globe-critical', SPACE.critical);
    C.select = token('--k-globe-select', SPACE.select);
    C.partner = token('--k-globe-partner', SPACE.partner);
    C.label = token('--k-globe-label', SPACE.label);
    C.grid = token('--k-globe-grid', SPACE.grid);
    C.gridStrong = token('--k-globe-grid-strong', SPACE.gridStrong);
    C.coast = token('--k-globe-coast', SPACE.coast);
    C.disk = token('--k-globe-disk', SPACE.disk);
    C.link = token('--k-globe-link', SPACE.link);
    const atmo = new Uint8ClampedArray(4);
    parseColorInto(token('--k-globe-atmosphere', SPACE.atmo), atmo);
    C.atmoRgb = `${atmo[0]}, ${atmo[1]}, ${atmo[2]}`;
    labelFont = `500 12px ${token('--k-sans', 'Inter, system-ui, sans-serif')}`;
    parseColorInto(C.payload, colPayload);
    parseColorInto(C.debris, colDebris);
    parseColorInto(C.rb, colRb);
    parseColorInto(C.dim, colDim);
    parseColorInto(C.elevated, colElev);
    parseColorInto(C.critical, colCrit);
    rebuildColorTables();              // base-group table for the active mode
    earthDirty = true;
  }

  /* ==========================================================================
     Public interface
     ======================================================================== */

  function init(opts) {
    if (ctx) return;                    // already initialized (destroy() first to re-init)
    initGen++;
    canvas = opts.canvas;
    tipEl = opts.tipEl || null;
    onSelectCb = opts.onSelect || null;
    wheelMode = opts.wheelZoom === 'modifier' ? 'modifier' : 'always';
    onWheelHint = typeof opts.onWheelHint === 'function' ? opts.onWheelHint : null;
    if (!themeHooked && typeof KTheme !== 'undefined' && typeof KTheme.onChange === 'function') {
      themeHooked = true;
      KTheme.onChange(() => { if (ctx) readColors(); });
    }
    ctx = canvas.getContext('2d');
    earthCanvas = document.createElement('canvas');
    earthCtx = earthCanvas.getContext('2d');
    dotCanvas = document.createElement('canvas');
    dotCtx = dotCanvas.getContext('2d');
    canvas.style.cursor = 'grab';
    readColors();
    buildGraticule();
    bindEvents();
    /* kick the one-time texture decode; the first ready() flip swaps the
       cached layer to SURFACE. Guarded - pages without earthtex.js stay
       wireframe, and a failed decode just logs (wireframe persists). */
    if (typeof KEarthTexture !== 'undefined') {
      KEarthTexture.onReady(() => { earthDirty = true; });
      KEarthTexture.load().catch((err) => console.error('KGlobe: earth texture load failed —', err));
    }
    loadFleet().catch((err) => console.error('KGlobe: fleet load failed —', err));
    loadCoastlines().catch((err) => console.error('KGlobe: coastline load failed —', err));
    startLoop();
  }

  /* Tear the engine down: stop the RAF loop, unbind every listener (canvas +
     document), release the catalog arrays, and allow a later re-init with a
     fresh canvas. Call when the hosting panel DOM is removed. */
  function destroy() {
    if (!ctx) return;                   // never initialized (or already destroyed)
    initGen++;                          // cancels any in-flight fleet/coastline load
    stopLoop();
    unbindEvents();
    if (tipEl) tipEl.style.display = 'none';
    canvas = null; ctx = null; tipEl = null; onSelectCb = null;
    earthCanvas = null; earthCtx = null;
    dotCanvas = null; dotCtx = null;
    dotImage = null; dotData = null;
    devW = 0; devH = 0; earthDirty = true;
    count = 0; visibleCount = 0;
    names = [];
    noradArr = null; typeCode = null; riskCode = null; fleetRisk = null;
    elA = null; elE = null; elB = null; elN = null; elM0 = null;
    basisP = null; basisQ = null; incDeg = null;
    posX = null; posY = null; posZ = null;
    posAX = null; posAY = null; posAZ = null;
    posBX = null; posBY = null; posBZ = null;
    posCX = null; posCY = null; posCZ = null;
    scrX = null; scrY = null; vis = null; filtOk = null;
    baseLists = null; baseCounts.fill(0);
    listElev = null; listCrit = null;
    constClass = null; altBin = null;
    heatBins = null; heatBW = 0; heatBH = 0;
    /* colorMode / heatOn / surfaceOn / autoSpinRad are settings, not data -
       they persist across destroy/re-init exactly like the filter does. */
    coastXYZ = null; coastStarts = null;
    idxByNorad.clear();
    selIdx = -1; partnerIdx = -1; hoverIdx = -1;
    pendingSelect = undefined;
    pendingEncounter = undefined;
    conjItems = []; conjSet = false;
    epochIso = null; needAnchor = true; cProgress = 0;
    clearEncounter();
    pendingSimMs = undefined;
  }

  function setSpeed(mult) {
    const m = Number(mult);
    if (Number.isFinite(m) && m > 0 && m !== speed) {
      speed = m;
      needAnchor = true;               // re-anchor snapshots at the new rate
    }
  }

  /* re-anchor the sim clock at the given wall-clock ms epoch. Both physics
     snapshots and the chunked C pre-solve are restarted coherently on the next
     tick (anchorSnapshots -> startNextSolve resets cProgress), so a jump can
     never leave stale pre-solved positions in the lerp pipeline. Safe before
     the fleet loads (applied when the snapshot arrives); non-finite ignored. */
  function setSimTime(msEpoch) {
    const ms = Number(msEpoch);
    if (!Number.isFinite(ms)) return;
    if (!count) {
      pendingSimMs = ms;
      return;
    }
    simOffsetMs = ms - epochMs;
    needAnchor = true;                 // next tick re-solves A/B and restarts C
  }

  /* an optional second argument {fly: false} keeps the view
     still; by default a programmatic selection eases the view so the object
     faces the viewer (never while auto-spin runs - the wall owns its view). */
  function select(noradOrNull, opts) {
    applySelect(noradOrNull);
    const o = opts || {};
    if (o.fly !== false && selIdx >= 0) {
      propagateOne(selIdx, simOffsetMs / 1000, p3);
      flyToward(p3[0], p3[1], p3[2]);
    }
  }

  /* Ease spin/tilt (~900 ms) so the ECI point (x, y, z) faces the viewer:
     spin puts its azimuth at the front meridian, tilt levels its latitude
     (clamped to ±55° so the view never flips over a pole). */
  function flyToward(x, y, z) {
    if (autoSpinRad !== 0 || dragging) return;
    const rxy = Math.hypot(x, y);
    if (!(rxy > 0 || z !== 0)) return;
    const s1 = -Math.PI / 2 - Math.atan2(y, x);
    let ds = (s1 - spin) % TWO_PI;
    if (ds > Math.PI) ds -= TWO_PI;
    else if (ds < -Math.PI) ds += TWO_PI;
    const t1 = Math.max(-55 * DEG, Math.min(55 * DEG, Math.atan2(z, rxy)));
    fly = {
      s0: spin, ds, t0: tilt, dt: t1 - tilt,
      start: performance.now(),
      dur: 650 + 400 * Math.min(1, Math.abs(ds) / Math.PI),
    };
  }

  /* focus a conjunction - select the primary, pin the secondary as the
     styled partner, and arm the TCA overlay (markers/connector/label, drawn
     each frame until deselect/new select/destroy). Returns false - never
     throws - for unknown norads or an unparsable tcaIso. Before the fleet
     snapshot loads the call is queued (like select) and applied on load.
     watch: jump the sim clock to TCA - 120 s at x60. */
  function showEncounter(opts) {
    const o = opts || {};
    if (!count) { pendingEncounter = o; return true; }
    const pi = idxByNorad.get(Number(o.primaryNorad));
    const si = idxByNorad.get(Number(o.secondaryNorad));
    const tcaMs = Date.parse(o.tcaIso);
    if (pi === undefined || si === undefined || !Number.isFinite(tcaMs)) return false;
    applySelect(noradArr[pi]);         // clears any prior overlay, fires onSelect
    partnerIdx = si;                   // encounter partner overrides conjunction lookup
    buildOrbit(si, partnerOrbit);
    const t = (tcaMs - epochMs) / 1000;
    propagateOne(pi, t, p3);
    encP[0] = p3[0]; encP[1] = p3[1]; encP[2] = p3[2];
    propagateOne(si, t, p3);
    encS[0] = p3[0]; encS[1] = p3[1]; encS[2] = p3[2];
    encM[0] = (encP[0] + encS[0]) / 2;
    encM[1] = (encP[1] + encS[1]) / 2;
    encM[2] = (encP[2] + encS[2]) / 2;
    encLabel = `Closest approach ${new Date(tcaMs).toISOString().slice(11, 16)} UTC`;
    encSepKm = Math.hypot(encP[0] - encS[0], encP[1] - encS[1], encP[2] - encS[2]);
    encPriNorad = noradArr[pi];
    encSecNorad = noradArr[si];
    encSecIdx = si;
    encActive = true;
    recountVisible();                  // pinned partner counts as visible when filtered
    if (o.fly !== false) flyToward(encM[0], encM[1], encM[2]);
    if (o.watch) {
      setSimTime(tcaMs - 120000);
      setSpeed(60);
    }
    return true;
  }

  function status() {
    return {
      epoch: epochIso,
      simIso: epochIso
        ? `${new Date(epochMs + simOffsetMs).toISOString().slice(0, 19)}Z`
        : null,
      simMs: epochIso ? epochMs + simOffsetMs : null,
      count,
      visible: visibleCount,
      speed,
      /* live encounter overlay: the pair and how far apart the globe's
         (simplified, newest-TLE) orbits put the two markers at TCA */
      encounter: encActive
        ? { primaryNorad: encPriNorad, secondaryNorad: encSecNorad, sepKm: encSepKm }
        : null,
    };
  }

  function setConjunctions(items) {
    conjItems = Array.isArray(items) ? items : [];
    conjSet = true;
    if (count) {
      applyRiskFromItems();
      updatePartner();
      applyFilter();                   // riskOnly depends on refreshed risk codes
      rebuildDrawLists();
    }
  }

  /* {types: ['PAYLOAD'|'DEBRIS'|'RB'|'UNKNOWN', ...], riskOnly: bool}.
     Omitted keys keep their current value; unknown tokens are ignored. Safe to
     call before load - the stored filter applies when the snapshot arrives. */
  function setFilter(opts) {
    const o = opts || {};
    if (Array.isArray(o.types)) {
      typeMask[0] = typeMask[1] = typeMask[2] = typeMask[3] = 0;
      for (const t of o.types) {
        const c = TOKEN_CODES[String(t).toUpperCase()];
        if (c !== undefined) typeMask[c] = 1;
      }
    }
    if (o.riskOnly !== undefined) riskOnly = !!o.riskOnly;
    if (filtOk) {
      applyFilter();
      rebuildDrawLists();
    }
  }

  /* switch the base-field color mode ('risk'|'type'|'constellation'|
     'altitude'). Unknown modes are ignored (same tolerance as setFilter).
     Rebuilds the blit color table + draw-list grouping HERE - the frame loop
     only reads them. Safe before the fleet loads (lists regroup at load).
     See the header block for the risk-vs-type decision; the engine default
     ('type') renders byte-identically to the earlier version. */
  function setColorMode(mode) {
    const m = String(mode).toLowerCase();
    if (COLOR_MODES.indexOf(m) < 0 || m === colorMode) return;
    colorMode = m;
    rebuildColorTables();
    if (count) rebuildDrawLists();
  }

  function getColorModes() {
    return COLOR_MODES.slice();
  }

  /* legend descriptor for the dashboard - {mode, entries:[{label,color}]}.
     Risk entries close every legend: amber/red overrides apply in all modes.
     Token-backed colors come from the resolved --k-* set (fallbacks before
     init); constellation/ramp colors are the contract-fixed hexes. */
  function legend() {
    const t2 = SPACE.payload;
    const t3 = SPACE.debris;
    const entries = [];
    if (colorMode === 'risk') {
      entries.push({ label: 'No elevated risk', color: C.dim || t3 });
    } else if (colorMode === 'constellation') {
      for (let s = 0; s < 5; s++) entries.push({ label: CONST_LABELS[s], color: CONST_HEX[s] });
      entries.push({ label: 'Other', color: C.debris || t3 });
    } else if (colorMode === 'altitude') {
      for (let b = 0; b < 5; b++) entries.push({ label: ALT_LABELS[b], color: ALT_HEX[b] });
    } else if (colorMode === 'storm') {
      for (let b = 0; b < 5; b++) entries.push({ label: STORM_LABELS[b], color: STORM_HEX[b] });
    } else {
      entries.push({ label: 'Payload', color: C.payload || t2 },
        { label: 'Debris', color: C.debris || t3 },
        { label: 'Rocket body', color: C.rb || t3 });
    }
    entries.push({ label: 'Elevated', color: C.elevated || '#e0a63c' },
      { label: 'Critical', color: C.critical || '#e05252' });
    return { mode: colorMode, entries };
  }

  /* toggle the additive screen-density layer (see drawHeat). */
  function setHeatmap(on) {
    heatOn = !!on;
  }

  /* SURFACE (textured hemisphere) vs WIREFRAME. Takes effect on the next
     cached-layer render; persisted in localStorage 'kessler.surface'
     (default ON). A missing/undecoded earthtex module leaves the wireframe
     regardless - the preference still persists. */
  function setSurface(on) {
    const v = !!on;
    if (v === surfaceOn) return;
    surfaceOn = v;
    earthDirty = true;
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('kessler.surface', v ? '1' : '0');
      }
    } catch (e) { /* storage disabled - session-only toggle */ }
  }

  /* wall mode: continuous view yaw in deg/s (0 stops; non-finite ignored).
     Additive - drag/zoom/selection behave exactly as before while spinning. */
  function setAutoSpin(degPerSec) {
    const d = Number(degPerSec);
    if (!Number.isFinite(d)) return;
    autoSpinRad = d * DEG;
    if (d === 0) earthDirty = true;    // commit any sub-threshold spin residual
  }

  /* full catalog for search/typeahead - [] before the fleet loads. */
  function list() {
    const out = new Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = {
        norad: noradArr[i],
        name: names[i],
        type: TYPE_TOKENS[typeCode[i]],
        risk: riskCode[i],
      };
    }
    return out;
  }

  /* frame the Earth inside the canvas region not covered by
     floating UI - {top, right, bottom, left} in css px (omitted keys keep
     their value). Too-large insets are ignored per axis at draw time. */
  function setViewInset(o) {
    const v = o || {};
    for (const k of ['top', 'right', 'bottom', 'left']) {
      if (v[k] === undefined) continue;
      const n = Number(v[k]);
      if (Number.isFinite(n) && n >= 0) inset[k] = n;
    }
  }

  /* atmospheric limb glow on/off (default on). */
  function setAtmosphere(on) {
    const v = !!on;
    if (v === atmosphereOn) return;
    atmosphereOn = v;
    earthDirty = true;
  }

  /* eased zoom by a factor (e.g. 1.4 in, 1/1.4 out). */
  function zoomBy(factor) {
    const f = Number(factor);
    if (!Number.isFinite(f) || f <= 0) return;
    const from = zoomTarget === zoomTarget ? zoomTarget : zoom;
    zoomTarget = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, from * f));
  }

  /* --- space weather layer --------------------------------------------------- */

  /* 'r, g, b' token mixed toward a heated orange by f (0..1): the rim of
     the atmosphere warms as the storm swells it */
  function warmRgb(rgb, f) {
    const p = String(rgb).split(',').map((v) => parseFloat(v));
    if (p.length < 3 || p.some((v) => !Number.isFinite(v))) return rgb;
    const w = [255, 150, 108], k = Math.min(1, Math.max(0, f)) * 0.6;
    return p.slice(0, 3).map((v, i) => Math.round(v + (w[i] - v) * k)).join(', ');
  }

  function ratioAt(km) {
    if (!swAlts) return 1;
    if (km <= swAlts[0]) return swRatio[0];
    for (let k = 1; k < swAlts.length; k++) {
      if (km <= swAlts[k]) {
        const t = (km - swAlts[k - 1]) / (swAlts[k] - swAlts[k - 1]);
        return swRatio[k - 1] + (swRatio[k] - swRatio[k - 1]) * t;
      }
    }
    return swRatio[swRatio.length - 1];
  }

  function rebinStorm() {
    if (!stormBin || !perigeeKm) return;
    for (let i = 0; i < count; i++) {
      const pk = perigeeKm[i];
      if (!swAlts || !(pk < STORM_MAX_KM)) { stormBin[i] = 0; continue; }
      const r = ratioAt(pk);
      stormBin[i] = r < 1.25 ? 1 : r < 1.75 ? 2 : r < 2.5 ? 3 : 4;
    }
  }

  function makeAuroraSprite() {
    const c = document.createElement('canvas');
    c.width = c.height = 32;
    const g = c.getContext('2d');
    const gr = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    gr.addColorStop(0, 'rgba(70, 255, 140, 0.9)');
    gr.addColorStop(0.45, 'rgba(50, 235, 125, 0.38)');
    gr.addColorStop(1, 'rgba(30, 210, 110, 0)');
    g.fillStyle = gr;
    g.fillRect(0, 0, 32, 32);
    return c;
  }

  /* Live aurora (NOAA OVATION nowcast) in the cached earth layer: soft
     additive sprites at 110 km, full strength on the night side and a faint
     trace in daylight (the energy lands there too; the sky is just bright). */
  function drawAurora(ec, g) {
    if (!swAurN || !(fRs > 6)) return;
    if (!auroraSprite) auroraSprite = makeAuroraSprite();
    sunDirEci(epochMs + simOffsetMs, sunV);
    const gm = g - spin;                             // GMST at sim time
    const cg = Math.cos(gm), sg = Math.sin(gm);
    const sx = cg * sunV[0] + sg * sunV[1], sy = -sg * sunV[0] + cg * sunV[1], sz = sunV[2];
    /* sprites overlap ~16 deep on the 1° x 2° nowcast grid: keep each faint
       so the oval builds up to a green glow, never a white band */
    const size = Math.max(4, fRs * 0.07), half = size / 2;
    const rA = RE + 110, rr2 = fRs * fRs;
    ec.save();
    ec.globalCompositeOperation = 'lighter';
    for (let k = 0; k < swAurN; k++) {
      const j = 4 * k;
      const x = swAur[j], y = swAur[j + 1], z = swAur[j + 2];
      projEarth(x, y, z);
      if (pjD <= 0) {                                // behind: only past the limb
        const dx = pjX - fCx, dy = pjY - fCy;
        if (dx * dx + dy * dy < rr2) continue;
      }
      const cz = (x * sx + y * sy + z * sz) / rA;    // sine of the sun's elevation
      const night = cz < -0.1 ? 1 : cz > 0.1 ? 0.14 : 0.14 + 0.86 * (0.1 - cz) / 0.2;
      const a = swAur[j + 3] * night;
      if (a < 0.02) continue;
      ec.globalAlpha = Math.min(1, a * 0.11);
      ec.drawImage(auroraSprite, pjX - half, pjY - half, size, size);
    }
    ec.restore();
  }

  /* Feed the layer from /spaceweather: {aurora: [[lon, lat, pct], ...],
     alts: [km...], ratio: [...]} (either part may be absent). Re-bins the
     'storm' color mode and invalidates the cached earth layer. */
  function setSpaceWeather(o) {
    const q = o || {};
    const ok = Array.isArray(q.alts) && Array.isArray(q.ratio) &&
      q.alts.length > 1 && q.alts.length === q.ratio.length &&
      q.ratio.every((v) => Number.isFinite(v));
    swAlts = ok ? q.alts.map(Number) : null;
    swRatio = ok ? q.ratio.map(Number) : null;
    const pts = Array.isArray(q.aurora) ? q.aurora : [];
    swAur = new Float32Array(pts.length * 4);
    swAurN = 0;
    const rA = RE + 110;
    for (const pt of pts) {
      const lon = Number(pt[0]) * DEG, lat = Number(pt[1]) * DEG, pct = Number(pt[2]);
      if (!(pct > 0) || !Number.isFinite(lon) || !Number.isFinite(lat)) continue;
      const cl = Math.cos(lat), j = 4 * swAurN++;
      swAur[j] = rA * cl * Math.cos(lon);
      swAur[j + 1] = rA * cl * Math.sin(lon);
      swAur[j + 2] = rA * Math.sin(lat);
      swAur[j + 3] = Math.min(1, pct / 60);
    }
    swWarm = swAlts ? Math.max(0, Math.min(1, (ratioAt(550) - 1.2) / 1.8)) : 0;
    rebinStorm();
    if (colorMode === 'storm' && count) rebuildDrawLists();
    earthDirty = true;
  }

  /* objects per storm-drag bin (index = STORM_LABELS) for the dashboard */
  function stormCounts() {
    const out = [0, 0, 0, 0, 0];
    if (!stormBin) return out;
    for (let i = 0; i < count; i++) out[stormBin[i]]++;
    return out;
  }

  /* A full-stage view covering the globe: stop the loop (and the physics)
     until it is uncovered; visibility changes respect it. */
  function setPaused(on) {
    appPaused = !!on;
    if (appPaused) stopLoop();
    else if (!document.hidden) startLoop();
  }

  return { init, destroy, setSpeed, setSimTime, select, showEncounter, status,
    setConjunctions, setFilter, list,
    setColorMode, getColorModes, legend, setHeatmap,
    setSurface, setAutoSpin,
    setViewInset, setAtmosphere, zoomBy, setSpaceWeather, setPaused, stormCounts };
})();
