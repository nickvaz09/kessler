/* =============================================================================
   KSim - Encounter Simulator controller (docs/CONTRACTS.md).

   Owns the full-viewport overlay DOM (z=700, above the assistant trajectory
   modal at z=600), the transport state and the rAF loop. All canvas rendering
   is delegated to the stateless renderers in js/simviews.js:

     SimViews.drawRelative(canvas, sim)
     SimViews.drawContext(canvas, sim)
     SimViews.drawRange(canvas, sim)

   The `sim` object handed to every draw call (one stable reference, mutated
   in place each frame):
     ephem    - GET /conjunctions/{id}/ephemeris payload
     tRel     - seconds relative to TCA, in [-window_s, +window_s]
     pr, sr   - interpolated primary/secondary position [x,y,z] km TEME
     relKm    - sr - pr (km TEME);  sepKm - |relKm| (km)
     speed    - 0.1 | 1 | 10 | 60 | 300;  playing - bool
     zoom     - {mode:'AUTO'|'MANUAL', pxPerKm}
     trailRel - ring buffer (<=120 entries, 1 s spacing, oldest->newest) of
                {t: secondsVsTca, rel:[x,y,z] km}; rebuilt on every seek
     basis    - encounter-plane basis {e1,e2,n}: orthonormal 3-vectors,
                n = rel_vel_hat (TEME), e1/e2 span the plane ⊥ n. Falls back
                to the RIC R̂/Ĉ projection when rel_vel_hat is null.
     rangeCache - per-sample separations (km, grid order, length N),
                precomputed once per load for SimViews.drawRange.
     maneuver - POST /conjunctions/{id}/maneuver response after a successful
                APPLY in the WHAT-IF MANEUVER section, else null. The
                post-burn ghost position in this primary-centred relative
                frame is rel(new_tca) - primary_rel_offset_km: the burn moves
                the PRIMARY by +offset, so the secondary-relative-to-primary
                vector shifts by MINUS that offset
                (rel' = sec - (pri + off) = rel - off).
     header   - {primaryName, secondaryName, riskLevel} for the burned-in
                header of SimViews.drawComposite (export), set once at load.

   EXPORT WEBM: a scripted pass tRel -120 s -> +60 s (clamped to ±window)
   at x15 is driven by the SAME rAF loop on the SAME sim object - the live
   view keeps rendering and mirrors the recording - composited each frame
   onto an offscreen 1280x720 canvas via SimViews.drawComposite and captured
   with canvas.captureStream(30) + MediaRecorder (vp9, vp8 fallback). The
   transport is disabled while recording ('RECORDING <n>S' meta); ESC cancels
   and restores the EXACT prior transport state (tRel/playing/speed/zoom),
   which is also restored after a completed export downloads
   kessler-encounter-<id>.webm. Unsupported browsers get a disabled button
   with a title note.

   NARRATE/VOICE: NARRATE fetches POST /assistant/narrate
   {conjunction_id} -> {beats:[{t_rel_s, text}], model} once per open (cached
   for the session), then schedules beats against sim tRel - the caption is a
   CONTROLLER-owned DOM line overlaying the bottom of the relative view
   (never canvas), swapped on the next beat, re-synced on any seek. VOICE
   plays beats as ElevenLabs audio (below); voice is cancelled on close, on
   pause, on backward seek and on toggle-off. Loading/error states render
   inline in the caption line; a 404 (endpoint absent - pre-restart server)
   hides the narration controls for the session.

   CLARITY (first-time legibility, no dumbing down):
   - FLYBY PATH: sim.pathCache {key, pts} - the secondary's full relative
     trajectory across the window decimated to <= 400 rel 3-vectors, drawn
     by drawRelative as a faint polyline beneath everything. The CONTROLLER
     owns the cache: rebuilt only when the rounded pxPerKm key changes (the
     basis is fixed per load, so zoom is the only material invalidator).
   - PHASE + PLAIN LANGUAGE: phase from tRel (APPROACH / CLOSEST APPROACH
     within ±15 s / SEPARATION) as a .label in the relative view's top-left;
     under the SEPARATION hero a plain line 'WILL PASS <dist> AWAY · IN
     <mm:ss>' before TCA -> 'PASSED <dist> AWAY · <mm:ss> AGO' after. At the
     TCA crossing during playback ONE ring pulse (sim.pulse, normalized
     0..1 over 1.2 s) fires at the ghost - once per pass, re-armed on seek.
   - VOICE (ElevenLabs): each beat is synthesized server-side via POST
     /assistant/tts {text} -> audio/mpeg (the API key never reaches the
     browser; the server disk-caches every clip so a beat is billed once).
     Clips are cached here as object URLs (revoked on close) and prefetched
     when VOICE turns on so playback lands on the beat. One clip plays at a
     time after a 250 ms breath; pause, backward seek, toggle-off and close
     all stop it. A 503 (no ELEVENLABS_API_KEY on the server) disables the
     toggle for the session with a title note.

   Positions between the 1 s ephemeris samples are LINEARLY interpolated:
   over one second the relative motion near an encounter is locally
   rectilinear, so interpolation is exact at sample points and sub-meter
   accurate at midpoints.

   REDESIGN + ONE-CLICK AVOIDANCE:
   - The dominant view is a dark "space stage" (.ksim-space pins the dark
     palette in both themes; simviews resolves tokens against the canvas)
     with glass overlays: phase pill, view switch, separation hero, zoom.
     Chrome (header, side column, transport) follows the page theme.
   - PLAN AVOIDANCE: POST /conjunctions/{id}/avoidance-plan -> a recommended
     burn card ("Fire 0.04 m/s prograde at 06:53 UTC, 6 h before closest
     approach -> miss 2.19 km, probability 8.1e-7"). "Apply to simulation"
     loads the burn into the what-if controls (exact dv + burn instant),
     POSTs /maneuver through the existing path and replays the pass so the
     before/after ghosts land. KSim.open({conjunctionId, applyPlan:true})
     does the whole thing on open (assistant open_simulator apply_plan);
     KSim.planAvoidance() runs it on the open simulator.
   - AUTO zoom scales by the IN-PLANE (encounter-plane) separation - the
     plane view projects out the relative-velocity axis, so the 3-D range
     (thousands of km early in the window) left the secondary pinned on the
     primary until the last seconds.

   ESC coordination: while open, a CAPTURING document keydown listener closes
   KSim and stops propagation, so assistant.js's bubble-phase handler (which
   would otherwise close the trajectory modal or the chat drawer) never sees
   the event. Full teardown on close: rAF cancelled, every listener removed
   (tracked add/remove pairs), DOM removed, arrays dropped.
   ========================================================================== */
'use strict';

const KSim = (() => {
  const RE_KM = 6378.137;
  const SPEEDS = [0.1, 1, 10, 60, 300];
  const DEFAULT_SPEED = 60;
  /* Wheel/AUTO zoom clamp. The contract's "0.01 px/m .. 500 px/km" literal
     cannot reach the HBR rule (visible when pxPerKm > 2000) nor fit
     window-edge separations (~17,500 km at x14.6 km/s over 1200 s), so it is
     read as 0.01 px/km .. 500 px/m - spanning both extremes. */
  const PXKM_MIN = 0.01;      /* 0.01 px per km */
  const PXKM_MAX = 500000;    /* 500 px per m   */
  const ZOOM_EASE = 0.15;     /* pxPerKm eases toward target each frame */
  const TRAIL_MAX = 120;
  const WINDOW_S = 1200;      /* requested half-window, server may clamp */
  const MNV_BURNS = [12, 6, 3, 1];  /* burn-time choices, hours before TCA */
  const MNV_BURN_DEFAULT = 6;
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const PLAN_PATH = (id) => `/conjunctions/${id}/avoidance-plan`;
  const REPLAY_FROM_S = 90;   /* after applying a burn, replay from TCA-90 s */
  const REPLAY_SPEED = 10;
  const MNV_DV_MAX = 5;       /* |Δv| clamp, m/s (server 422s beyond ±5) */
  /* export: scripted pass -120 -> +60 s at x15 onto a 1280x720 canvas */
  const EXP_T0 = -120;
  const EXP_T1 = 60;
  const EXP_SPEED = 15;
  const EXP_W = 1280;
  const EXP_H = 720;
  const EXP_FPS = 30;
  const EXP_MIMES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const NARRATE_PATH = '/assistant/narrate';
  const TTS_PATH = '/assistant/tts';
  const VOICE_GAP_MS = 250;  /* breath inserted before each spoken beat */
  const TTS_GONE_TITLE = 'Voice isn’t set up on this server yet';
  /* one user-facing sentence for a missing Anthropic key, shared with the
     assistant (js/assistant.js) - operator detail only in tooltips */
  const AI_NOKEY_TEXT = 'Kessler AI isn’t set up on this server yet.';
  const AI_NEEDS_TITLE = 'Needs Kessler AI to be set up';
  const AI_NOKEY_KEY = 'kessler.ai.nokey';   /* sessionStorage flag, see assistant.js */
  const MNV_DV_DEFAULT = 0.05; /* manual what-if starting burn, m/s (QA A10) */
  const PATH_MAX = 400;      /* flyby-path decimation cap (points) */
  const PULSE_S = 1.2;       /* closest-approach ring pulse duration, s */
  const PHASE_TCA_S = 15;    /* |tRel| <= 15 s reads CLOSEST APPROACH */

  /* --- module state --------------------------------------------------------- */
  let overlay = null;        /* root .ksim element; null = closed */
  let refs = null;           /* direct element refs (no querySelector) */
  let listeners = null;      /* [[target, type, fn, opts]] for teardown */
  let focusables = null;     /* tab-trap cycle, DOM order */
  let rafId = null;
  let lastTs = null;
  let token = 0;             /* bumped on open/close - cancels stale fetches */
  let prevFocus = null;
  let prevBodyOverflow = '';
  let openConjId = null;
  let ready = false;
  let viewsOk = false;       /* typeof SimViews guard, checked per open */

  let conj = null;           /* /conjunctions/{id} payload */
  let ephem = null;          /* /conjunctions/{id}/ephemeris payload */
  let W = 0;                 /* window_s (half-window, seconds) */
  let N = 0;                 /* usable sample count (min of both arrays) */
  let tcaMs = 0;
  let sim = null;
  let zoomTarget = 1;
  let lastTrailSec = 0;
  let dragging = false;
  let mnv = null;            /* WHAT-IF MANEUVER ui state:
                                {dv, burnH, busy, openUi, ghostRel, ghostKm} */
  let plan = null;           /* avoidance plan: {status:'idle'|'loading'|
                                'ready'|'error', data, err, pick, promise} */
  let applyPlanOnLoad = false;
  let exp = null;            /* EXPORT WEBM state while recording:
                                {t, t0, t1, id, canvas, recorder, prior,
                                 cancelled} - null when idle */
  let exportOk = false;      /* MediaRecorder + captureStream + drawComposite */
  let exportWhy = '';        /* title note for the disabled button */
  let nar = null;            /* NARRATE state: {on, voice, status:'idle'|
                                'loading'|'ready'|'error', beats, idx, gone} */
  let missM = 0;             /* miss distance at TCA (m) for the plain line */
  let pulseT = null;         /* closest-approach pulse elapsed s, null idle */
  let pulseArmed = true;     /* one pulse per pass; re-armed on seek */
  let voiceTimer = null;     /* pending 250 ms inter-beat gap timer */
  let lastAriaS = null;      /* last aria-valuenow written (whole s) */
  let audioEl = null;        /* live beat clip (HTMLAudio), null idle */
  let audioUrls = new Map(); /* beat text -> object URL, revoked on close */
  let audioReqs = new Map(); /* beat text -> in-flight fetch, deduped */
  let voiceGone = false;     /* server 503'd (no TTS key) - sticky per session */
  let spokenDone = new Set(); /* beat texts whose clip played to the end -
                                 resume-after-pause won't re-read them */
  let aiGone = false;        /* narrate 503'd (no Anthropic key) - sticky */
  let pendingOpen = null;    /* open() called before the page finished loading */

  /* --- tiny DOM + listener helpers ------------------------------------------ */

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function addL(target, type, fn, opts) {
    target.addEventListener(type, fn, opts);
    listeners.push([target, type, fn, opts]);
  }

  function clearChildren(e) {
    while (e.firstChild) e.removeChild(e.firstChild);
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* js/api.js exposes no ephemeris/maneuver wrappers - mirror its request()
     semantics (error shape {detail}) for these two extra calls. err.detail
     carries the server detail verbatim (422 maneuver rejections show it raw). */
  async function request(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* keep */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      err.detail = detail;
      throw err;
    }
    return res.json();
  }

  function getJson(path) { return request(path); }

  function postJson(path, body) {
    return request(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  function isoZ(ms) {
    return new Date(ms).toISOString().replace(/\.\d+Z$/, 'Z');
  }

  /* --- vector helpers -------------------------------------------------------- */

  const vDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const vCross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const vMag = (a) => Math.hypot(a[0], a[1], a[2]);
  function vUnit(a) {
    const m = vMag(a);
    return m > 0 ? [a[0] / m, a[1] / m, a[2] / m] : [0, 0, 0];
  }

  /* h minus its component along unit n, normalized; null when h ∥ n */
  function gsPerp(h, n) {
    const d = vDot(h, n);
    const p = [h[0] - d * n[0], h[1] - d * n[1], h[2] - d * n[2]];
    const m = vMag(p);
    return m > 1e-6 ? [p[0] / m, p[1] / m, p[2] / m] : null;
  }

  /* Encounter-plane basis. n = rel_vel_hat; e1 = RIC R̂ Gram-Schmidt'd
     against n (fixed-axis fallbacks when parallel); e2 = n x e1.
     rel_vel_hat null -> RIC R̂/Ĉ from primary position + finite-diff velocity. */
  function computeBasis() {
    const ts = ephem.tca_state || {};
    const iTca = clamp(W, 0, N - 1);
    const rTca = ephem.primary.r[iTca] || [1, 0, 0];
    const hat = Array.isArray(ts.rel_vel_hat) ? vUnit(ts.rel_vel_hat) : null;
    if (!hat || vMag(hat) === 0) {
      const rHat = vUnit(rTca);
      const i1 = clamp(iTca + 1, 0, N - 1);
      const i0 = clamp(iTca - 1, 0, N - 1);
      const v = [
        ephem.primary.r[i1][0] - ephem.primary.r[i0][0],
        ephem.primary.r[i1][1] - ephem.primary.r[i0][1],
        ephem.primary.r[i1][2] - ephem.primary.r[i0][2],
      ];
      let cHat = vUnit(vCross(rTca, v));
      if (vMag(cHat) === 0) cHat = vUnit(vCross(rTca, [0, 0, 1]));
      return { e1: rHat, e2: cHat, n: vUnit(vCross(rHat, cHat)) };
    }
    const e1 = gsPerp(vUnit(rTca), hat) || gsPerp([0, 0, 1], hat) || gsPerp([1, 0, 0], hat);
    return { e1, e2: vCross(hat, e1), n: hat };
  }

  /* --- interpolation on the 1 s grid ----------------------------------------- */

  function sampleInto(arr, tRel, out) {
    const f = clamp(tRel + W, 0, N - 1);
    const i0 = clamp(Math.floor(f), 0, Math.max(N - 2, 0));
    const fr = f - i0;
    const a = arr[i0];
    const b = arr[Math.min(i0 + 1, N - 1)];
    out[0] = a[0] + (b[0] - a[0]) * fr;
    out[1] = a[1] + (b[1] - a[1]) * fr;
    out[2] = a[2] + (b[2] - a[2]) * fr;
    return out;
  }

  function interpTo(tRel) {
    sampleInto(ephem.primary.r, tRel, sim.pr);
    sampleInto(ephem.secondary.r, tRel, sim.sr);
    sim.relKm[0] = sim.sr[0] - sim.pr[0];
    sim.relKm[1] = sim.sr[1] - sim.pr[1];
    sim.relKm[2] = sim.sr[2] - sim.pr[2];
    sim.sepKm = vMag(sim.relKm);
  }

  const _ta = [0, 0, 0], _tb = [0, 0, 0];
  function relAt(t) {
    sampleInto(ephem.primary.r, t, _ta);
    sampleInto(ephem.secondary.r, t, _tb);
    return [_tb[0] - _ta[0], _tb[1] - _ta[1], _tb[2] - _ta[2]];
  }

  /* --- flyby path cache --------------------------------------------------------
     The full relative trajectory across the window, decimated to <= PATH_MAX
     grid samples, for drawRelative's faint flyby polyline. The basis is fixed
     per load, so the only material invalidator is zoom: keyed by the ROUNDED
     pxPerKm so the per-frame zoom ease never causes per-frame rebuilds. */

  function relAtIdx(j) {
    const a = ephem.primary.r[j];
    const b = ephem.secondary.r[j];
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  }

  function ensurePathCache() {
    const key = Math.round(sim.zoom.pxPerKm);
    if (sim.pathCache && sim.pathCache.key === key) return;
    const step = Math.max(1, Math.ceil(N / PATH_MAX));
    const pts = [];
    for (let j = 0; j < N; j += step) pts.push(relAtIdx(j));
    if (step > 1 && (N - 1) % step !== 0) pts.push(relAtIdx(N - 1));
    sim.pathCache = { key, pts };
  }

  /* --- trail ring buffer (120 entries, 1 s spacing) --------------------------- */

  function rebuildTrail() {
    sim.trailRel.length = 0;
    const end = Math.floor(sim.tRel);
    for (let s = Math.max(-W, end - (TRAIL_MAX - 1)); s <= end; s++) {
      sim.trailRel.push({ t: s, rel: relAt(s) });
    }
    lastTrailSec = end;
  }

  function extendTrail() {
    const cur = Math.floor(sim.tRel);
    while (lastTrailSec < cur) {
      lastTrailSec += 1;
      sim.trailRel.push({ t: lastTrailSec, rel: relAt(lastTrailSec) });
      if (sim.trailRel.length > TRAIL_MAX) sim.trailRel.shift();
    }
  }

  /* --- transport -------------------------------------------------------------- */

  function setTRel(t, seek) {
    const prev = sim.tRel;
    sim.tRel = clamp(t, -W, W);
    if (seek) {
      rebuildTrail();
      resyncBeats(prev);   /* re-sync captions on any seek (narration) */
      pulseArmed = true;   /* any seek re-arms the TCA pulse ... */
      pulseT = null;       /* ... and cancels one in flight */
      sim.pulse = null;
    }
    interpTo(sim.tRel);
  }

  function setPlaying(p) {
    if (p && sim.tRel >= W) setTRel(-W, true); /* play at end restarts */
    sim.playing = p;
    if (!p) cancelSpeech();     /* pause silences the narration voice */
    else speakCurrent(false);   /* resume re-reads a beat cut off mid-clip */
    if (refs && refs.playBtn) renderPlayBtn();
  }

  const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="currentColor"/><rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="currentColor"/></svg>';
  const ICON_CLOSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/></svg>';

  function renderPlayBtn() {
    const p = !!(sim && sim.playing);
    refs.playBtn.innerHTML = p ? ICON_PAUSE : ICON_PLAY;
    refs.playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
    refs.playBtn.title = p ? 'Pause (space)' : 'Play (space)';
  }

  function setSpeed(s) {
    sim.speed = s;
    if (refs && refs.speedBtns) {
      refs.speedBtns.forEach((b, i) => {
        const on = SPEEDS[i] === s;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-pressed', String(on));
      });
    }
  }

  function cycleSpeed(dir) {
    const i = clamp(SPEEDS.indexOf(sim.speed) + dir, 0, SPEEDS.length - 1);
    setSpeed(SPEEDS[i]);
  }

  function setZoomMode(mode) {
    sim.zoom.mode = mode;
    if (refs && refs.autoBtn) {
      refs.autoBtn.classList.toggle('is-active', mode === 'AUTO');
      refs.autoBtn.setAttribute('aria-pressed', String(mode === 'AUTO'));
    }
  }

  /* main-panel mode: encounter plane or the space-camera chase view.
     Zoom controls only mean something on the plane - hidden on camera. */
  function setView(v) {
    if (!sim || !refs) return;
    sim.view = v;
    refs.viewRelBtn.classList.toggle('is-active', v === 'rel');
    refs.viewCamBtn.classList.toggle('is-active', v === 'cam');
    refs.viewRelBtn.setAttribute('aria-pressed', String(v === 'rel'));
    refs.viewCamBtn.setAttribute('aria-pressed', String(v === 'cam'));
    refs.zoomBox.hidden = v !== 'rel';
    refs.axisCap.textContent = v === 'cam'
      ? 'Chase camera above the primary, Earth-centered frame'
      : 'Looking along the approach: the plane perpendicular to relative velocity';
    collectFocusables();
  }

  /* In-plane (encounter-plane) radius of a relative vector, km. The plane
     view projects out the relative-velocity axis, so this - not |rel| - is
     how far from the primary a point is DRAWN. */
  function inPlaneKm(rel) {
    const b = sim.basis;
    if (!b || !rel) return 0;
    return Math.hypot(vDot(rel, b.e1), vDot(rel, b.e2));
  }

  /* AUTO keeps the drawn secondary ~ 40% of the half-view (smoothed via
     ZOOM_EASE), scaled by the IN-PLANE separation (the 3-D range
     is dominated by the along-v_rel component the view projects away).
     The TCA miss point and any post-burn ghost are folded in so both stay
     on screen. */
  function autoZoomTarget() {
    const w = (refs && refs.relCanvas && refs.relCanvas.clientWidth) || 600;
    const h = (refs && refs.relCanvas && refs.relCanvas.clientHeight) || 400;
    const half = Math.min(w, h) / 2;
    const ghostKm = sim.maneuver && mnv && mnv.ghostRel ? inPlaneKm(mnv.ghostRel) : 0;
    const r = Math.max(inPlaneKm(sim.relKm), sim.tcaInPlaneKm || 0, ghostKm);
    return clamp(0.4 * half / Math.max(r, 1e-6), PXKM_MIN, PXKM_MAX);
  }

  function onWheel(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    if (exp) return; /* transport (incl. zoom) disabled while recording */
    if (sim.view !== 'rel') return; /* zoom is an encounter-plane control */
    if (sim.zoom.mode !== 'MANUAL') setZoomMode('MANUAL');
    zoomTarget = clamp(zoomTarget * Math.exp(-ev.deltaY * 0.0015), PXKM_MIN, PXKM_MAX);
  }

  /* --- rAF loop ---------------------------------------------------------------- */

  function frame(ts) {
    rafId = requestAnimationFrame(frame);
    const dt = lastTs == null ? 0 : Math.min((ts - lastTs) / 1000, 0.25);
    lastTs = ts;
    const tPrev = sim.tRel;   /* TCA-crossing detection for the pulse */
    if (exp) {
      /* scripted export pass - drives the shared sim, so the live view
         mirrors the recording; transport is disabled meanwhile */
      if (dt > 0) {
        exp.t = Math.min(exp.t + dt * EXP_SPEED, exp.t1);
        sim.tRel = exp.t;
        interpTo(sim.tRel);
        extendTrail();
        refs.expMeta.textContent =
          `Recording ${Math.floor((exp.t - exp.t0) / EXP_SPEED)} s`;
      }
    } else if (sim.playing && dt > 0) {
      const t = sim.tRel + dt * sim.speed;
      if (t >= W) {
        setTRel(W, false);
        setPlaying(false);          /* clamp at +window -> pause */
      } else {
        sim.tRel = t;
        interpTo(sim.tRel);
      }
      extendTrail();
    }
    /* closest-approach pulse: fires once when playback (or the export
       pass) carries tRel across TCA - a seek is not a pass and never fires;
       it only re-arms. Wall-clock 1.2 s life, normalized for the renderer. */
    if (dt > 0) {
      if (pulseArmed && tPrev < 0 && sim.tRel >= 0) {
        pulseT = 0;
        pulseArmed = false;
      } else if (pulseT != null) {
        pulseT += dt;
        if (pulseT >= PULSE_S) pulseT = null;
      }
    }
    sim.pulse = pulseT == null ? null : pulseT / PULSE_S;
    tickBeats();
    if (sim.zoom.mode === 'AUTO') zoomTarget = autoZoomTarget();
    sim.zoom.pxPerKm += (zoomTarget - sim.zoom.pxPerKm) * ZOOM_EASE;
    ensurePathCache();
    updateTimeline();
    updateReadouts();
    if (viewsOk) {
      if (sim.view === 'cam' && typeof SimViews.drawCamera === 'function') {
        SimViews.drawCamera(refs.relCanvas, sim);
      } else {
        SimViews.drawRelative(refs.relCanvas, sim);
      }
      SimViews.drawContext(refs.ctxCanvas, sim);
      SimViews.drawRange(refs.rangeCanvas, sim);
      if (exp) SimViews.drawComposite(exp.canvas, sim);
    }
    if (exp && exp.t >= exp.t1) endExport(true);
  }

  /* --- readouts ----------------------------------------------------------------- */

  const pad2 = (n) => String(n).padStart(2, '0');

  function fmtVsTca(t) {
    const a = Math.abs(t);
    const h = Math.floor(a / 3600), m = Math.floor((a % 3600) / 60), s = Math.floor(a % 60);
    return `${t >= 0 ? '+' : '−'}${h ? `${pad2(h)}:` : ''}${pad2(m)}:${pad2(s)}`;
  }

  function mmss(s) {
    return `${pad2(Math.floor(s / 60))}:${pad2(Math.floor(s % 60))}`;
  }

  function fmtZoom(p) {
    /* as a scale: how much distance 100 px spans */
    const km = 100 / p;
    return `100 px = ${FMT.dist(km * 1000)}`;
  }

  /* geocentric lat / GMST-derived lon / spherical alt from a TEME position */
  function latLonAlt(r, tRel) {
    const rm = vMag(r);
    const gmst = (ephem.gmst_start_rad || 0) + (ephem.earth_rot_rad_s || 0) * (tRel + W);
    let lon = (Math.atan2(r[1], r[0]) - gmst) * 180 / Math.PI;
    lon = ((lon % 360) + 540) % 360 - 180;
    const lat = Math.asin(clamp(r[2] / rm, -1, 1)) * 180 / Math.PI;
    return { lat, lon, alt: rm - RE_KM };
  }

  /* current relative speed from the grid segment slope (1 s step -> km/s) */
  function gridRelVelMs() {
    const i0 = clamp(Math.floor(sim.tRel + W), 0, Math.max(N - 2, 0));
    const i1 = Math.min(i0 + 1, N - 1);
    const pa = ephem.primary.r, sa = ephem.secondary.r;
    return vMag([
      (sa[i1][0] - sa[i0][0]) - (pa[i1][0] - pa[i0][0]),
      (sa[i1][1] - sa[i0][1]) - (pa[i1][1] - pa[i0][1]),
      (sa[i1][2] - sa[i0][2]) - (pa[i1][2] - pa[i0][2]),
    ]) * 1000;
  }

  function updateTimeline() {
    const pct = ((sim.tRel + W) / (2 * W)) * 100;
    refs.tlFill.style.width = `${pct}%`;
    refs.tlCursor.style.left = `${pct}%`;
    const nowS = Math.round(sim.tRel);
    if (refs.tl && nowS !== lastAriaS) {
      lastAriaS = nowS;
      refs.tl.setAttribute('aria-valuenow', String(nowS));
      refs.tl.setAttribute('aria-valuetext',
        `${Math.abs(nowS)} seconds ${nowS < 0 ? 'before' : 'after'} closest approach`);
    }
  }

  /* encounter phase from tRel - the one word a first-time user needs */
  function phaseOf(t) {
    if (t < -PHASE_TCA_S) return 'Approaching';
    if (t <= PHASE_TCA_S) return 'Closest approach';
    return 'Moving apart';
  }

  /* After a burn is applied the hero follows the POST-burn pass: the same
     constant-offset drawing aid the stage uses for the after ghost
     (rel - primary_rel_offset_km), with the server's new_miss_m as the
     closest-approach figure. */
  function burnOffset() {
    const m = sim && sim.maneuver;
    const off = m && m.primary_rel_offset_km;
    return Array.isArray(off) && off.length >= 3 && off.every(Number.isFinite) ? off : null;
  }

  function updateReadouts() {
    const off = burnOffset();
    const after = !!off;
    const sepKm = after
      ? vMag([sim.relKm[0] - off[0], sim.relKm[1] - off[1], sim.relKm[2] - off[2]])
      : sim.sepKm;
    const passM = after && Number.isFinite(Number(sim.maneuver.new_miss_m))
      ? Number(sim.maneuver.new_miss_m) : missM;
    const heroTxt = after ? 'Separation after the burn' : 'Separation';
    if (refs.heroLbl.textContent !== heroTxt) refs.heroLbl.textContent = heroTxt;
    refs.sepVal.textContent = FMT.dist(sepKm * 1000);
    const ph = phaseOf(sim.tRel);
    if (refs.phaseLbl.textContent !== ph) {
      refs.phaseLbl.textContent = ph;
      refs.phaseLbl.classList.toggle('is-tca', ph === 'Closest approach');
    }
    refs.plain.textContent = (sim.tRel < 0
      ? `Will pass ${FMT.dist(passM)} apart in ${mmss(-sim.tRel)}`
      : `Passed ${FMT.dist(passM)} apart ${mmss(sim.tRel)} ago`)
      + (after ? ` · ${FMT.dist(missM)} without it` : '');
    refs.relVel.textContent = FMT.vel(gridRelVelMs());
    refs.tcaRow.textContent = fmtVsTca(sim.tRel);
    refs.tVal.textContent = fmtVsTca(sim.tRel);
    refs.zoomInfo.textContent = fmtZoom(sim.zoom.pxPerKm);
    for (const o of refs.objRows) {
      const s = latLonAlt(o.which === 'p' ? sim.pr : sim.sr, sim.tRel);
      o.alt.textContent = FMT.km(s.alt, 1);
      o.lat.textContent = `${s.lat.toFixed(2)}°`;
      o.lon.textContent = `${s.lon.toFixed(2)}°`;
    }
  }

  /* --- keyboard (capturing - ESC precedence over the trajectory modal) ---------- */

  function onKeyCapture(ev) {
    if (!overlay) return;
    /* keystrokes typed into the AI drawer (or any editable) never
       drive the sim - the assistant floats above this overlay and owns
       its own input. */
    const t = ev.target;
    if (t instanceof Element) {
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
          t.tagName === 'SELECT' || t.isContentEditable) return;
      if (!overlay.contains(t) &&
          t.closest('.kai-panel, .kai-modal, .kai-fab')) return;
    }
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      if (exp) { endExport(false); return; }  /* ESC cancels the export first */
      close();
      return;
    }
    if (ev.key === 'Tab') { trapTab(ev); return; }
    if (!ready) return;
    if (exp) { ev.preventDefault(); ev.stopPropagation(); return; } /* transport disabled */
    /* Space on a focused button activates the button (CLOSE, APPLY, a speed)
       - only bare focus gets the video-player play/pause toggle */
    if (ev.key === ' ' && ev.target instanceof HTMLButtonElement) return;
    switch (ev.key) {
      case ' ': setPlaying(!sim.playing); break;
      case 'ArrowLeft': setTRel(sim.tRel - (ev.shiftKey ? 10 : 1), true); break;
      case 'ArrowRight': setTRel(sim.tRel + (ev.shiftKey ? 10 : 1), true); break;
      case 'Home': setTRel(-W, true); break;
      case 'End': setTRel(W, true); break;
      case 't': case 'T': setTRel(Math.max(-W, -120), true); break;
      case '-': case '_': cycleSpeed(-1); break;
      case '+': case '=': cycleSpeed(1); break;
      default: return;
    }
    ev.preventDefault();
    ev.stopPropagation();
  }

  function trapTab(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    /* skip hidden/disabled stops - focus() on them no-ops and Tab would
       otherwise freeze there (e.g. transport buttons during export) */
    const list = (focusables || []).filter((b) => b && !b.hidden && !b.disabled);
    if (!list.length) return;
    const i = list.indexOf(ev.target);
    const next = ev.shiftKey
      ? (i <= 0 ? list.length - 1 : i - 1)
      : (i === -1 || i === list.length - 1 ? 0 : i + 1);
    if (list[next].focus) list[next].focus();
  }

  /* --- timeline scrubber --------------------------------------------------------- */

  function timelineSeek(clientX) {
    const rect = refs.timeline.getBoundingClientRect();
    const frac = clamp((clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
    setTRel(-W + frac * 2 * W, true);
  }

  function buildTimeline() {
    const tl = el('div', 'ksim-timeline');
    refs.tl = tl;
    tl.setAttribute('role', 'slider');
    tl.tabIndex = 0;   /* role=slider must be focusable; <-/-> seek globally */
    tl.setAttribute('aria-label', 'Encounter timeline');
    tl.setAttribute('aria-valuemin', String(-W));
    tl.setAttribute('aria-valuemax', String(W));
    tl.setAttribute('aria-valuenow', String(Math.round(-W)));
    tl.appendChild(el('div', 'ksim-timeline__track'));
    refs.tlFill = el('div', 'ksim-timeline__fill');
    tl.appendChild(refs.tlFill);
    /* CDM message-time markers (only those inside ±window) */
    for (const c of (conj.cdms || [])) {
      const dt = (new Date(c.message_time).getTime() - tcaMs) / 1000;
      if (!(dt >= -W && dt <= W)) continue;
      const m = el('div', 'ksim-timeline__cdm');
      m.style.left = `${((dt + W) / (2 * W)) * 100}%`;
      m.title = `Conjunction message ${FMT.utc(c.message_time)}: ${FMT.dist(c.miss_distance_m)}`;
      tl.appendChild(m);
    }
    const tca = el('div', 'ksim-timeline__tca');
    tca.appendChild(el('span', 'ksim-timeline__tcalbl', 'Closest approach'));
    tl.appendChild(tca);
    refs.tlCursor = el('div', 'ksim-timeline__cursor');
    tl.appendChild(refs.tlCursor);
    tl.appendChild(el('span', 'ksim-timeline__lbl ksim-timeline__lbl--l', `−${mmss(W)}`));
    tl.appendChild(el('span', 'ksim-timeline__lbl ksim-timeline__lbl--r', `+${mmss(W)}`));
    refs.timeline = tl;
    addL(tl, 'pointerdown', (ev) => {
      if (exp) return; /* transport disabled while recording */
      dragging = true;
      if (tl.setPointerCapture && ev.pointerId != null) tl.setPointerCapture(ev.pointerId);
      timelineSeek(ev.clientX);
      ev.preventDefault();
    });
    addL(tl, 'pointermove', (ev) => { if (dragging && !exp) timelineSeek(ev.clientX); });
    addL(tl, 'pointerup', () => { dragging = false; });
    addL(tl, 'pointercancel', () => { dragging = false; });
    return tl;
  }

  /* --- DOM build ------------------------------------------------------------------ */

  function nodata(msg) { return el('div', 'chart-nodata', msg); }

  function mkCanvas(cls) { return el('canvas', cls); }

  function btn(cls, text, label) {
    const b = el('button', cls, text);
    b.type = 'button';
    if (label) b.setAttribute('aria-label', label);
    return b;
  }

  function buildOverlay(id) {
    overlay = el('div', 'ksim');
    overlay.tabIndex = -1;
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Encounter simulator');
    refs = {};

    const hd = el('header', 'ksim__hd');
    refs.closeBtn = btn('ksim__close', null, 'Close simulator');
    refs.closeBtn.innerHTML = ICON_CLOSE;
    refs.closeBtn.title = 'Close (Esc)';
    hd.appendChild(refs.closeBtn);
    const title = el('div', 'ksim__title');
    const row = el('div', 'ksim__titlerow');
    refs.pair = el('h2', 'ksim__pair', `Encounter ${id}`);
    row.appendChild(refs.pair);
    refs.chip = el('span', 'chip', '');
    refs.chip.hidden = true;
    row.appendChild(refs.chip);
    title.appendChild(row);
    refs.hdMeta = el('div', 'ksim__hdmeta', 'Loading the encounter…');
    title.appendChild(refs.hdMeta);
    hd.appendChild(title);
    /* narration - hidden until the layout is ready; hidden for the whole
       session when /assistant/narrate 404s */
    const tools = el('div', 'ksim__hdtools');
    refs.narBtn = btn('btn btn--xs ksim__narbtn', 'Narrate');
    refs.narBtn.setAttribute('aria-pressed', 'false');
    refs.narBtn.hidden = true;
    tools.appendChild(refs.narBtn);
    refs.voiceBtn = btn('btn btn--xs ksim__voicebtn', 'Voice');
    refs.voiceBtn.setAttribute('aria-pressed', 'false');
    refs.voiceBtn.hidden = true;
    tools.appendChild(refs.voiceBtn);
    hd.appendChild(tools);
    overlay.appendChild(hd);
    addL(refs.narBtn, 'click', toggleNarrate);
    addL(refs.voiceBtn, 'click', toggleVoice);

    refs.stage = el('div', 'ksim__stage');
    overlay.appendChild(refs.stage);

    document.body.appendChild(overlay);
    addL(refs.closeBtn, 'click', close);
    addL(document, 'keydown', onKeyCapture, true);
    focusables = [refs.closeBtn];
  }

  function setStage(kind, msg) {
    ready = false;
    refs.stage.className = 'ksim__stage';
    clearChildren(refs.stage);
    if (kind === 'loading') {
      const n = el('div', 'ksim-note');
      n.appendChild(el('div', 'ksim-spinner'));
      n.appendChild(el('div', '', 'Propagating both orbits around closest approach…'));
      refs.stage.appendChild(n);
      focusables = [refs.closeBtn];
    } else if (kind === 'error') {
      const box = el('div', 'ksim-errbox');
      box.appendChild(el('div', 'ksim-errbox__title', 'The encounter could not be loaded'));
      box.appendChild(el('div', 'ksim-errbox__msg', msg || 'Unknown error'));
      refs.retryBtn = btn('btn btn--primary', 'Try again');
      box.appendChild(refs.retryBtn);
      refs.stage.appendChild(box);
      addL(refs.retryBtn, 'click', () => load(token));
      focusables = [refs.retryBtn, refs.closeBtn];
    } else if (kind === 'degrade') {
      const box = el('div', 'ksim-errbox');
      box.appendChild(el('div', 'ksim-errbox__title', 'No ephemeris for this event'));
      box.appendChild(el('div', 'ksim-errbox__msg',
        'Propagation failed for one of the two objects, so there is nothing honest to replay.'));
      refs.stage.appendChild(box);
      focusables = [refs.closeBtn];
    }
  }

  function panelHd(title, meta) {
    const hd = el('div', 'panel__hd');
    hd.appendChild(el('h3', 'panel__title', title));
    if (meta) hd.appendChild(el('span', 'panel__meta', meta));
    return hd;
  }

  function kvGrid() {
    const g = el('div', 'ksim-kv');
    const row = (lbl, title) => {
      const l = el('div', 'ksim-kv__label', lbl);
      if (title) l.title = title;
      g.appendChild(l);
      const v = el('div', 'ksim-kv__value', '—');
      g.appendChild(v);
      return v;
    };
    return { g, row };
  }

  function buildReadouts() {
    const bd = el('div', 'ksim-readouts');
    const { g, row } = kvGrid();
    const ts = ephem.tca_state || {};
    refs.relVel = row('Relative speed now', 'Instantaneous relative speed at the replay position');
    row('Relative speed at closest approach').textContent = FMT.vel(ts.rel_vel_ms);
    refs.tcaRow = row('Time from closest approach', 'Time relative to TCA (time of closest approach)');
    row('Miss at closest approach', 'Simulated from the latest TLEs').textContent = FMT.dist(ts.sep_m);
    /* the sim propagates TLEs, so its geometry can differ from the screened
       CDM miss - show both so the numbers never look contradictory next to
       the dashboard table */
    if (conj.miss_distance_m != null) {
      row('Miss in latest message', 'Screened miss distance from the latest conjunction data message (CDM)')
        .textContent = FMT.dist(conj.miss_distance_m);
    }
    /* RIC miss components (QA C2): one row each, formatted with units */
    const ric = ts.ric_m || {};
    const ricFmt = (v) => (v == null || !Number.isFinite(Number(v)) ? '—'
      : `${Number(v) < 0 ? '−' : ''}${FMT.dist(Math.abs(Number(v)))}`);
    row('Radial offset', 'Miss component along the primary’s radius (up/down), RIC frame')
      .textContent = ricFmt(ric.radial);
    row('In-track offset', 'Miss component along the primary’s direction of motion, RIC frame')
      .textContent = ricFmt(ric.in_track);
    row('Cross-track offset', 'Miss component across the primary’s orbit plane, RIC frame')
      .textContent = ricFmt(ric.cross_track);
    const eff = conj.predicted_final_risk != null ? conj.predicted_final_risk : conj.pc_latest;
    const pcCell = row('Probability of collision', 'Predicted final Pc, else the latest Pc');
    pcCell.textContent = `${FMT.pc(eff)} `;
    const mChip = el('span', 'chip chip--type', conj.model_version || '—');
    mChip.title = 'Model version';
    pcCell.appendChild(mChip);
    bd.appendChild(g);

    refs.objRows = [
      { which: 'p', o: conj.primary },
      { which: 's', o: conj.secondary },
    ].map((spec) => {
      const blk = el('div', 'ksim-obj');
      const nm = el('div', 'ksim-obj__name');
      nm.appendChild(el('span', `ksim-obj__dot ksim-obj__dot--${spec.which}`));
      nm.appendChild(el('span', '', spec.o.name));
      nm.appendChild(el('span', 'code ksim-obj__id', String(spec.o.norad_id)));
      blk.appendChild(nm);
      const k = kvGrid();
      const out = { which: spec.which, alt: k.row('Altitude'), lat: k.row('Latitude'), lon: k.row('Longitude') };
      blk.appendChild(k.g);
      bd.appendChild(blk);
      return out;
    });
    return bd;
  }

  /* --- WHAT-IF MANEUVER + AVOIDANCE PLAN ---------------------------------------
     The side column's first card. "Plan avoidance" asks the server for the
     smallest safe burn; "Apply to simulation" loads it into the what-if
     controls below (exact dv + burn instant) and runs the existing APPLY
     path (POST /maneuver -> sim.maneuver -> post-burn ghost). The manual
     controls stay available under "Adjust manually". Built only for future
     events (tca >= now). */

  function fmtDv(v) {
    const a = Math.abs(v);
    const digits = a === 0 ? 1 : a < 0.1 ? 3 : a < 1 ? 2 : 1;
    return a.toFixed(digits);
  }

  function hhmm(ms) { return new Date(ms).toISOString().slice(11, 16); }

  function fmtLead(sec) {
    const s = Math.round(sec);
    const h = Math.floor(s / 3600);
    const m = Math.round((s % 3600) / 60);
    if (h && m) return `${h} h ${m} min`;
    if (h) return `${h} h`;
    return `${m} min`;
  }

  function renderDv() {
    const v = mnv.dv;
    if (document.activeElement !== refs.mnvDv) refs.mnvDv.value = fmtDv(v);
    refs.mnvDir.textContent = v > 0 ? 'prograde' : v < 0 ? 'retrograde' : 'no burn';
    refs.mnvDv.setAttribute('aria-valuetext',
      `${fmtDv(v)} meters per second ${refs.mnvDir.textContent}`);
    syncApplyEnabled();
  }

  /* a 0 m/s "burn" is not a what-if: Simulate stays off with a hint (A10) */
  function syncApplyEnabled() {
    if (!refs || !refs.mnvApply || !mnv) return;
    const zero = !(Math.abs(mnv.dv) >= 0.0005);
    refs.mnvApply.disabled = mnv.busy || zero;
    if (refs.mnvHint) refs.mnvHint.hidden = !zero;
  }

  /* stepper granularity follows magnitude: plans read in cm/s, big what-ifs
     in half-m/s steps */
  /* magnitude stepper (direction lives in its own segmented control);
     granularity follows magnitude: plans read in cm/s, big what-ifs in
     half-m/s steps */
  function stepDv(dir) {
    mnv.touched = true;
    const mag = Math.abs(mnv.dv);
    const probe = dir > 0 ? mag : Math.max(mag - 1e-9, 0);
    const step = probe < 0.1 ? 0.01 : probe < 1 ? 0.05 : 0.5;
    const next = clamp(Math.round((mag + dir * step) / step) * step, 0, MNV_DV_MAX);
    mnv.dv = mnv.dir * (next < 1e-9 ? 0 : next);
    renderDv();
  }

  function onDvInput() {
    mnv.touched = true;
    const raw = Math.abs(parseFloat(refs.mnvDv.value));
    if (!Number.isFinite(raw)) { mnv.dv = 0; syncApplyEnabled(); return; }
    mnv.dv = mnv.dir * clamp(raw, 0, MNV_DV_MAX);
    syncApplyEnabled();
  }

  function setDirection(sgn) {
    mnv.dir = sgn;
    mnv.dv = sgn * Math.abs(mnv.dv);
    renderDv();
    syncDirButtons();
  }

  function syncDirButtons() {
    const s = mnv.dir;
    refs.mnvPro.classList.toggle('is-active', s > 0);
    refs.mnvRetro.classList.toggle('is-active', s < 0);
    refs.mnvPro.setAttribute('aria-pressed', String(s > 0));
    refs.mnvRetro.setAttribute('aria-pressed', String(s < 0));
  }

  /* burn instant: a preset (hours before TCA) or the plan's exact instant */
  function setBurn(h, ms) {
    mnv.burnH = h;
    mnv.burnMs = Number.isFinite(ms) ? ms : tcaMs - h * 3600000;
    refs.mnvBurnBtns.forEach((b, i) => {
      const on = h != null && MNV_BURNS[i] === h;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', String(on));
    });
    const lead = (tcaMs - mnv.burnMs) / 1000;
    refs.mnvBurnNote.textContent =
      `Burn at ${hhmm(mnv.burnMs)} UTC, ${fmtLead(lead)} before closest approach`;
  }

  function toggleManeuver(force) {
    mnv.openUi = typeof force === 'boolean' ? force : !mnv.openUi;
    refs.mnvBody.hidden = !mnv.openUi;
    refs.mnvHd.setAttribute('aria-expanded', String(mnv.openUi));
    refs.mnvHd.classList.toggle('is-open', mnv.openUi);
    collectFocusables();
  }

  /* Post-burn ghost in the relative frame. The burn moves the PRIMARY by
     primary_rel_offset_km at the new TCA; the view is primary-centred, so
     the secondary shifts by MINUS that offset:
       rel' = sec - (pri + off) = rel(new_tca) - off.
     |rel'| then matches the server's new_miss_m (asserted in the harness). */
  function computeManeuverGhost(resp) {
    let dt = 0;
    if (resp.new_tca) {
      const d = (new Date(resp.new_tca).getTime() - tcaMs) / 1000;
      if (Number.isFinite(d)) dt = clamp(d, -W, W);
    }
    const rel = relAt(dt);
    const raw = resp.primary_rel_offset_km;
    const off = Array.isArray(raw) && raw.length >= 3 && raw.every(Number.isFinite)
      ? raw : [0, 0, 0];
    mnv.ghostRel = [rel[0] - off[0], rel[1] - off[1], rel[2] - off[2]];
    mnv.ghostKm = vMag(mnv.ghostRel);
  }

  async function applyManeuver() {
    if (!mnv || mnv.busy) return null;
    mnv.busy = true;
    const myToken = token;
    refs.mnvApply.disabled = true;
    refs.mnvApply.textContent = 'Simulating…';
    refs.mnvErr.hidden = true;
    refs.mnvErr.textContent = '';
    const body = {
      dv_along_ms: Math.round(mnv.dv * 1e4) / 1e4,
      burn_iso: isoZ(mnv.burnMs),
    };
    let resp = null;
    let err = null;
    try {
      resp = await postJson(`/conjunctions/${openConjId}/maneuver`, body);
    } catch (e) { err = e; }
    if (myToken !== token || !overlay || !refs || !refs.mnvApply) return null; /* closed */
    mnv.busy = false;
    refs.mnvApply.textContent = 'Simulate burn';
    syncApplyEnabled();
    if (err) {
      /* 422 detail verbatim; anything else labelled. Inputs and any
         previously applied maneuver are preserved. */
      refs.mnvErr.textContent = err.status === 422 && err.detail
        ? `The server rejected this burn: ${err.detail}`
        : `The burn could not be simulated (${err.detail || err.message || 'unknown error'}).`;
      refs.mnvErr.hidden = false;
      return null;
    }
    sim.maneuver = resp;
    computeManeuverGhost(resp);
    const d = Number(resp.new_miss_m) - beforeMiss(resp.old && resp.old.miss_m);
    const dTxt = Number.isFinite(d)
      ? ` (${d >= 0 ? '+' : '−'}${FMT.dist(Math.abs(d))})` : '';
    refs.mnvMiss.textContent = `${FMT.dist(resp.new_miss_m)}${dTxt}`;
    refs.mnvPc.textContent = FMT.pc(resp.new_pc);
    refs.mnvRes.hidden = false;
    renderBurnBadge();
    return resp;
  }

  function clearManeuver() {
    if (!mnv || mnv.busy) return;
    mnv.dv = MNV_DV_DEFAULT;
    mnv.dir = 1;
    mnv.touched = false;
    mnv.ghostRel = null;
    mnv.ghostKm = 0;
    sim.maneuver = null;
    renderDv();
    syncDirButtons();
    setBurn(MNV_BURN_DEFAULT);
    refs.mnvErr.hidden = true;
    refs.mnvErr.textContent = '';
    refs.mnvRes.hidden = true;
    refs.mnvMiss.textContent = '—';
    refs.mnvPc.textContent = '—';
    if (refs.planApply) refs.planApply.textContent = 'Apply to simulation';
    renderBurnBadge();
  }

  /* stage overlay: before -> after once a burn is applied */
  /* One number for the unburned miss everywhere (QA C9): the plan and the
     maneuver sandbox report their own grid minimum rounded to 0.1 m, which
     can straddle a rounding edge against tca_state.sep_m (206.5 vs 206.48 ->
     "207 m" vs "206 m"). Within 1 m it is the same pass - show sep_m. */
  function beforeMiss(v) {
    const n = Number(v);
    return Number.isFinite(n) && Number.isFinite(missM) && Math.abs(n - missM) < 1 ? missM : n;
  }

  function renderBurnBadge() {
    if (!refs.burnBadge) return;
    const m = sim && sim.maneuver;
    refs.burnBadge.hidden = !m;
    if (!m) return;
    clearChildren(refs.burnBadge);
    const old = m.old || {};
    refs.burnBadge.appendChild(el('span', 'ksim-badge__k', 'Before'));
    refs.burnBadge.appendChild(el('span', 'ksim-badge__v', FMT.dist(beforeMiss(old.miss_m))));
    const arrow = el('span', 'ksim-badge__arrow', '→');
    arrow.setAttribute('aria-hidden', 'true');
    refs.burnBadge.appendChild(arrow);
    refs.burnBadge.appendChild(el('span', 'ksim-badge__k', 'After burn'));
    refs.burnBadge.appendChild(el('span', 'ksim-badge__v ksim-badge__v--after', FMT.dist(m.new_miss_m)));
  }

  /* ---- avoidance plan ---- */

  function planSentence(o) {
    const dir = o.direction === 'retrograde' ? 'retrograde' : 'prograde';
    return `Fire ${fmtDv(o.dv_ms)} m/s ${dir} at ${hhmm(Date.parse(o.burn_iso))} UTC`;
  }

  function renderPlan() {
    const out = refs.planOut;
    clearChildren(out);
    if (!plan || plan.status === 'idle') { out.hidden = true; return; }
    out.hidden = false;
    if (plan.status === 'loading') {
      out.appendChild(el('div', 'ksim-plan__loading',
        'Searching burns 1 orbit, 3, 6 and 12 hours out…'));
      return;
    }
    if (plan.status === 'error') {
      const e = plan.err || {};
      const msg = e.status === 409
        ? `There is no time left to plan a burn: ${e.detail || 'closest approach is too close'}.`
        : `The plan could not be computed (${e.detail || e.message || 'unknown error'}).`;
      out.appendChild(el('div', 'ksim-plan__note', msg));
      return;
    }
    const d = plan.data;
    if (d.status !== 'burn_recommended') {
      const box = el('div', 'ksim-plan__note');
      if (d.status === 'no_burn_needed') {
        box.appendChild(el('strong', '', 'No burn needed.'));
        box.appendChild(document.createTextNode(
          ` The latest orbits already pass ${FMT.dist(beforeMiss(d.baseline.miss_m))} apart, with probability ${FMT.pc(d.baseline.pc)}.`));
      } else {
        box.appendChild(el('strong', '', d.status === 'geometry_mismatch'
          ? 'Tracking data disagree.' : 'No burn within limits.'));
        box.appendChild(document.createTextNode(` ${d.note || ''}`));
      }
      out.appendChild(box);
      return;
    }
    const opts = d.options.filter((o) => o.reachable);
    const o = opts.find((x) => x.lead_s === plan.pick) || d.recommended;
    const card = el('div', 'ksim-plan__card');
    card.appendChild(el('div', 'ksim-plan__eyebrow',
      o.lead_s === d.recommended.lead_s ? 'Recommended burn' : 'Alternative burn'));
    card.appendChild(el('div', 'ksim-plan__headline', planSentence(o)));
    card.appendChild(el('div', 'ksim-plan__sub',
      `${fmtLead(o.lead_s)} before closest approach${o.lead_label === '1 orbit' ? ' (one orbit)' : ''}`));
    const res = el('div', 'ksim-plan__result');
    const cell = (k, from, to, title) => {
      const c = el('div', 'ksim-plan__cell');
      if (title) c.title = title;
      c.appendChild(el('div', 'ksim-plan__k', k));
      const v = el('div', 'ksim-plan__v');
      v.appendChild(el('span', 'ksim-plan__from', from));
      v.appendChild(el('span', 'ksim-plan__arrow', '→'));
      v.appendChild(el('span', 'ksim-plan__to', to));
      c.appendChild(v);
      return c;
    };
    res.appendChild(cell('Miss distance', FMT.dist(beforeMiss(d.baseline.miss_m)), FMT.dist(o.new_miss_m)));
    res.appendChild(cell('Probability', FMT.pc(d.baseline.pc), FMT.pc(o.new_pc),
      'Assumed-uncertainty probability of collision (Pc)'));
    card.appendChild(res);
    const meta = el('div', 'ksim-plan__meta');
    if (o.meets_pc) meta.appendChild(el('span', 'chip chip--ok', 'Below the elevated threshold'));
    const fuel = el('span', 'ksim-plan__fuel',
      `About ${o.propellant_g_per_100kg < 1 ? o.propellant_g_per_100kg.toFixed(2) : o.propellant_g_per_100kg.toFixed(1)} g of propellant per 100 kg`);
    fuel.title = `Rocket equation at Isp ${d.assumptions.isp_s} s (hydrazine class)`;
    meta.appendChild(fuel);
    card.appendChild(meta);
    const act = el('div', 'ksim-plan__actions');
    refs.planApply = btn('btn btn--primary', 'Apply to simulation');
    act.appendChild(refs.planApply);
    addL(refs.planApply, 'click', () => applyPlanOption(o));
    card.appendChild(act);
    out.appendChild(card);

    if (opts.length > 1) {
      const list = el('div', 'ksim-plan__opts');
      list.appendChild(el('div', 'ksim-plan__optshd', 'Other burn times'));
      for (const x of opts) {
        const b = btn(`ksim-plan__opt${x === o ? ' is-active' : ''}`, null,
          `${planSentence(x)}, ${fmtLead(x.lead_s)} before closest approach`);
        b.setAttribute('aria-pressed', String(x === o));
        b.appendChild(el('span', 'ksim-plan__optlead', x.lead_label === '1 orbit' ? '1 orbit' : `${x.lead_label} before`));
        b.appendChild(el('span', 'ksim-plan__optdv', `${fmtDv(x.dv_ms)} m/s`));
        b.appendChild(el('span', 'ksim-plan__optmiss', FMT.dist(x.new_miss_m)));
        addL(b, 'click', () => { plan.pick = x.lead_s; renderPlan(); collectFocusables(); });
        list.appendChild(b);
      }
      out.appendChild(list);
    }
    const method = el('div', 'ksim-plan__method', `${d.method.charAt(0).toUpperCase()}${d.method.slice(1)}.`);
    method.title = d.criteria.rule;
    out.appendChild(method);
  }

  async function requestPlan() {
    if (!plan || plan.status === 'loading') return plan ? plan.promise : null;
    const myToken = token;
    plan.status = 'loading';
    plan.pick = null;
    refs.planBtn.disabled = true;
    refs.planBtn.textContent = 'Planning…';
    renderPlan();
    plan.promise = (async () => {
      let data = null;
      let err = null;
      try { data = await postJson(PLAN_PATH(openConjId), {}); } catch (e) { err = e; }
      if (myToken !== token || !overlay || !plan) return null;
      refs.planBtn.disabled = false;
      refs.planBtn.textContent = 'Plan again';
      if (err) { plan.status = 'error'; plan.err = err; } else {
        plan.status = 'ready';
        plan.data = data;
        if (data.recommended) {
          plan.pick = data.recommended.lead_s;
          /* the manual what-if starts from the plan's burn until touched */
          if (mnv && !mnv.touched && !mnv.busy && !sim.maneuver) {
            const o = data.recommended;
            mnv.dv = o.dv_ms;
            mnv.dir = o.dv_ms < 0 ? -1 : 1;
            renderDv();
            syncDirButtons();
            const bms = Date.parse(o.burn_iso);
            const pre = MNV_BURNS.find((h) => Math.abs(tcaMs - h * 3600000 - bms) < 1500);
            setBurn(pre != null ? pre : null, bms);
          }
        }
      }
      renderPlan();
      collectFocusables();
      return data;
    })();
    return plan.promise;
  }

  /* load an option into the what-if controls, simulate it, replay the pass */
  async function applyPlanOption(o) {
    if (!o || !mnv || mnv.busy) return null;
    mnv.dv = o.dv_ms;
    mnv.dir = o.dv_ms < 0 ? -1 : 1;
    renderDv();
    syncDirButtons();
    const burnMs = Date.parse(o.burn_iso);
    const preset = MNV_BURNS.find((h) => Math.abs(tcaMs - h * 3600000 - burnMs) < 1500);
    setBurn(preset != null ? preset : null, burnMs);
    if (refs.planApply) {
      refs.planApply.disabled = true;
      refs.planApply.textContent = 'Applying…';
    }
    const resp = await applyManeuver();
    if (!refs || !refs.planApply) return resp;
    refs.planApply.disabled = false;
    refs.planApply.textContent = resp ? 'Applied — replaying' : 'Apply to simulation';
    if (resp) {
      if (sim.view !== 'rel') setView('rel');
      setZoomMode('AUTO');
      setSpeed(REPLAY_SPEED);
      setTRel(Math.max(-W, -REPLAY_FROM_S), true);
      setPlaying(true);
      /* stacked layout (phones/tablets): bring the replay into view */
      const relEl = refs.relCanvas && refs.relCanvas.closest('.ksim__rel');
      if (relEl && window.innerWidth <= 1100 && relEl.scrollIntoView) {
        try { relEl.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (_) { /* old engine */ }
      }
    }
    return resp;
  }

  function buildManeuver() {
    const sec = el('section', 'panel ksim-avoid');
    const hd = panelHd('Avoidance', 'Screening-grade');
    sec.appendChild(hd);
    const bd = el('div', 'ksim-avoid__bd');
    bd.appendChild(el('p', 'ksim-avoid__intro',
      'Find the smallest along-track burn that makes this pass safe.'));
    refs.planBtn = btn('btn btn--primary btn--lg ksim-avoid__plan', 'Plan avoidance');
    bd.appendChild(refs.planBtn);
    refs.planOut = el('div', 'ksim-plan');
    refs.planOut.hidden = true;
    bd.appendChild(refs.planOut);

    /* manual what-if controls, collapsed under a disclosure row */
    refs.mnvHd = btn('ksim-mnv__hd');
    refs.mnvHd.setAttribute('aria-expanded', 'false');
    refs.mnvHd.appendChild(el('span', '', 'Adjust manually'));
    refs.mnvCaret = el('span', 'ksim-mnv__caret');
    refs.mnvCaret.innerHTML = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5L7.5 6 4 9.5" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    refs.mnvHd.appendChild(refs.mnvCaret);
    bd.appendChild(refs.mnvHd);

    refs.mnvBody = el('div', 'ksim-mnv__bd');
    refs.mnvBody.hidden = true;

    const dvRow = el('div', 'ksim-mnv__row');
    dvRow.appendChild(el('label', 'ksim-mnv__lbl', 'Burn size'));
    const stepper = el('div', 'ksim-mnv__stepper');
    refs.mnvMinus = btn('btn btn--xs', '−', 'Smaller burn');
    refs.mnvDv = el('input', 'input ksim-mnv__dv');
    refs.mnvDv.type = 'text';
    refs.mnvDv.inputMode = 'decimal';
    refs.mnvDv.setAttribute('aria-label', 'Burn size in meters per second');
    refs.mnvPlus = btn('btn btn--xs', '+', 'Larger burn');
    stepper.appendChild(refs.mnvMinus);
    stepper.appendChild(refs.mnvDv);
    stepper.appendChild(el('span', 'ksim-mnv__unit', 'm/s'));
    stepper.appendChild(refs.mnvPlus);
    dvRow.appendChild(stepper);
    refs.mnvBody.appendChild(dvRow);

    const dirRow = el('div', 'ksim-mnv__row');
    dirRow.appendChild(el('span', 'ksim-mnv__lbl', 'Direction'));
    const dseg = el('div', 'seg');
    refs.mnvPro = btn('btn btn--xs', 'Prograde');
    refs.mnvRetro = btn('btn btn--xs', 'Retrograde');
    dseg.appendChild(refs.mnvPro);
    dseg.appendChild(refs.mnvRetro);
    dirRow.appendChild(dseg);
    refs.mnvDir = el('span', 'ksim-mnv__dir sr-only', '');
    dirRow.appendChild(refs.mnvDir);
    refs.mnvBody.appendChild(dirRow);

    const bRow = el('div', 'ksim-mnv__row');
    bRow.appendChild(el('span', 'ksim-mnv__lbl', 'Burn time'));
    const burns = el('div', 'seg ksim-mnv__burns');
    refs.mnvBurnBtns = MNV_BURNS.map((h) => {
      const b = btn('btn btn--xs', `${h} h`, `${h} hours before closest approach`);
      addL(b, 'click', () => setBurn(h));
      burns.appendChild(b);
      return b;
    });
    bRow.appendChild(burns);
    refs.mnvBody.appendChild(bRow);
    refs.mnvBurnNote = el('div', 'ksim-mnv__note', '');
    refs.mnvBody.appendChild(refs.mnvBurnNote);

    const act = el('div', 'ksim-mnv__actions');
    refs.mnvApply = btn('btn', 'Simulate burn');
    refs.mnvClear = btn('btn btn--ghost', 'Clear');
    act.appendChild(refs.mnvApply);
    act.appendChild(refs.mnvClear);
    refs.mnvBody.appendChild(act);
    refs.mnvHint = el('div', 'ksim-mnv__note', 'Set a burn size above 0 m/s to simulate it.');
    refs.mnvHint.hidden = true;
    refs.mnvBody.appendChild(refs.mnvHint);

    refs.mnvErr = el('div', 'ksim-plan__note');
    refs.mnvErr.hidden = true;
    refs.mnvBody.appendChild(refs.mnvErr);

    const k = kvGrid();
    refs.mnvRes = k.g;
    refs.mnvRes.classList.add('ksim-mnv__res');
    refs.mnvMiss = k.row('New miss distance');
    refs.mnvPc = k.row('New probability', 'Assumed-uncertainty Pc after the burn');
    refs.mnvRes.hidden = true;
    refs.mnvBody.appendChild(refs.mnvRes);
    refs.mnvBody.appendChild(el('div', 'ksim-mnv__method',
      'Two-body differential model with assumed uncertainty; not an operational maneuver plan.'));

    bd.appendChild(refs.mnvBody);
    sec.appendChild(bd);
    addL(refs.planBtn, 'click', requestPlan);
    addL(refs.mnvHd, 'click', () => toggleManeuver());
    addL(refs.mnvMinus, 'click', () => stepDv(-1));
    addL(refs.mnvPlus, 'click', () => stepDv(1));
    addL(refs.mnvDv, 'input', onDvInput);
    addL(refs.mnvDv, 'change', () => { renderDv(); syncDirButtons(); });
    addL(refs.mnvPro, 'click', () => setDirection(1));
    addL(refs.mnvRetro, 'click', () => setDirection(-1));
    addL(refs.mnvApply, 'click', applyManeuver);
    addL(refs.mnvClear, 'click', clearManeuver);
    return sec;
  }

  /* --- EXPORT WEBM -------------------------------------------------------------
     Scripted -120 -> +60 s pass at x15 recorded from an offscreen 1280x720
     canvas (captureStream 30 fps + MediaRecorder, vp9 -> vp8 -> plain webm).
     The pass drives the SHARED sim from the main rAF loop, so the live view
     keeps rendering (it mirrors the recording); the transport is disabled and
     ESC cancels. Both finish and cancel restore the exact prior transport
     state. Chunks/cancel flag live in the recorder's closure, so a close()
     mid-export can drop `exp` without breaking the onstop handler. */

  function detectExportSupport() {
    if (!viewsOk || typeof SimViews.drawComposite !== 'function') {
      return { ok: false, why: 'Video export is unavailable: renderers not loaded' };
    }
    if (typeof MediaRecorder === 'undefined') {
      return { ok: false, why: 'Video export needs MediaRecorder, which this browser lacks' };
    }
    let probe = null;
    try { probe = document.createElement('canvas'); } catch (_) { /* below */ }
    if (!probe || typeof probe.captureStream !== 'function') {
      return { ok: false, why: 'Video export needs canvas capture, which this browser lacks' };
    }
    return { ok: true, why: '' };
  }

  /* vp9 -> vp8 -> unqualified webm; isTypeSupported may itself be absent */
  function makeRecorder(stream) {
    const can = typeof MediaRecorder.isTypeSupported === 'function'
      ? (m) => { try { return MediaRecorder.isTypeSupported(m); } catch (_) { return false; } }
      : () => true;
    for (const mime of EXP_MIMES) {
      if (!can(mime)) continue;
      try { return { rec: new MediaRecorder(stream, { mimeType: mime }), mime }; } catch (_) { /* next */ }
    }
    return { rec: new MediaRecorder(stream), mime: 'video/webm' };
  }

  function setTransportDisabled(dis) {
    refs.playBtn.disabled = dis;
    refs.speedBtns.forEach((b) => { b.disabled = dis; });
    if (refs.planBtn) refs.planBtn.disabled = dis;
    refs.autoBtn.disabled = dis;
    refs.expBtn.disabled = dis || !exportOk;
    refs.footer.classList.toggle('is-recording', dis);
  }

  function startExport() {
    if (exp || !ready || !exportOk) return;
    const myToken = token;
    const state = {
      id: openConjId,
      t0: Math.max(-W, EXP_T0),
      t1: Math.min(W, EXP_T1),
      t: Math.max(-W, EXP_T0),
      prior: {
        tRel: sim.tRel,
        playing: sim.playing,
        speed: sim.speed,
        zoomMode: sim.zoom.mode,
        pxPerKm: sim.zoom.pxPerKm,
        zoomTarget,
      },
      canvas: null,
      recorder: null,
      mime: 'video/webm',
      cancelled: false,
    };
    setPlaying(false);            /* pause the live view state */
    const canvas = document.createElement('canvas');
    canvas.width = EXP_W;
    canvas.height = EXP_H;
    state.canvas = canvas;
    let made;
    try {
      made = makeRecorder(canvas.captureStream(EXP_FPS));
    } catch (_) {
      restoreTransport(state.prior);
      return;
    }
    state.recorder = made.rec;
    state.mime = made.mime;
    const chunks = [];
    made.rec.ondataavailable = (ev) => {
      if (ev && ev.data && (ev.data.size == null || ev.data.size > 0)) chunks.push(ev.data);
    };
    made.rec.onstop = () => {
      /* runs after exp is already null - everything it needs is closed over */
      if (state.cancelled || myToken !== token || !chunks.length) return;
      try {
        const blob = new Blob(chunks, { type: state.mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `kessler-encounter-${state.id}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (_) { /* download blocked - nothing to clean up */ }
    };
    exp = state;
    try { made.rec.start(); } catch (_) { exp = null; restoreTransport(state.prior); return; }
    setTransportDisabled(true);
    refs.expMeta.textContent = 'Recording 0 s';
    refs.expMeta.hidden = false;
    setTRel(state.t0, true);      /* seek to the pass start, trail rebuilt */
  }

  function restoreTransport(p) {
    setSpeed(p.speed);
    setZoomMode(p.zoomMode);
    zoomTarget = p.zoomTarget;
    sim.zoom.pxPerKm = p.pxPerKm;
    setTRel(p.tRel, true);
    setPlaying(p.playing);
  }

  /* save=true -> completed pass (onstop downloads); false -> ESC cancel */
  function endExport(save) {
    if (!exp) return;
    const state = exp;
    exp = null;
    state.cancelled = !save;
    try {
      if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop();
    } catch (_) { /* recorder already dead */ }
    refs.expMeta.hidden = true;
    refs.expMeta.textContent = '';
    setTransportDisabled(false);
    restoreTransport(state.prior);
  }

  /* --- NARRATE + VOICE ---------------------------------------------------------
     Beats from POST /assistant/narrate scheduled against sim tRel: the
     caption (a DOM line overlaying the relative view bottom - never canvas)
     shows the latest beat with t_rel_s <= tRel, swapped when the next beat's
     time is reached and re-synced on any seek. VOICE plays each beat as an
     ElevenLabs clip fetched from POST /assistant/tts (server-keyed,
     server-cached); voice stops on pause, close, backward seek and
     toggle-off. A narrate 404 hides both buttons (endpoint not deployed);
     a tts 503 disables VOICE for the session (no server key). */

  function cancelSpeech() {
    if (voiceTimer != null) {   /* pending inter-beat gap dies with the beat */
      try { clearTimeout(voiceTimer); } catch (_) { /* no timers here */ }
      voiceTimer = null;
    }
    if (audioEl) {
      try { audioEl.pause(); } catch (_) { /* already dead */ }
      audioEl = null;
    }
  }

  function readAiNoKey() {
    try { return sessionStorage.getItem(AI_NOKEY_KEY) === '1'; } catch (_) { return false; }
  }

  /* Narrate needs Kessler AI (Anthropic key). Known missing - from this
     session's first 503 or the assistant's shared flag - disables Narrate
     and Voice (voice reads the narration) with a plain tooltip. */
  function syncAiButtons() {
    if (!refs || !refs.narBtn) return;
    const gone = aiGone || readAiNoKey();
    refs.narBtn.disabled = gone;
    refs.narBtn.title = gone ? AI_NEEDS_TITLE : 'Narrate the pass with Kessler AI';
    if (gone && nar && nar.on) narrationOff();
    const vGone = gone || voiceGone;
    refs.voiceBtn.disabled = vGone;
    refs.voiceBtn.title = gone ? AI_NEEDS_TITLE
      : voiceGone ? TTS_GONE_TITLE : 'Read the narration aloud';
    if (vGone && nar && nar.voice) {
      nar.voice = false;
      refs.voiceBtn.classList.remove('is-active');
      refs.voiceBtn.setAttribute('aria-pressed', 'false');
      cancelSpeech();
    }
    collectFocusables();
  }

  function onAiAvailability(ev) {
    const avail = !!(ev && ev.detail && ev.detail.available);
    if (avail) aiGone = false;
    syncAiButtons();
  }

  /* the server said 503 - no key configured; disable for the session */
  function voiceUnavailable() {
    voiceGone = true;
    if (nar) nar.voice = false;
    cancelSpeech();
    if (refs && refs.voiceBtn) {
      refs.voiceBtn.classList.remove('is-active');
      refs.voiceBtn.setAttribute('aria-pressed', 'false');
      refs.voiceBtn.disabled = true;
      refs.voiceBtn.title = TTS_GONE_TITLE;
    }
  }

  /* one clip per beat text, deduped in flight and cached as an object URL -
     combined with the server's disk cache a beat is synthesized exactly once */
  function fetchBeatAudio(text, sig) {
    if (voiceGone) return Promise.reject(new Error('tts-unavailable'));
    if (!sig) return Promise.reject(new Error('tts-unsigned'));
    if (audioUrls.has(text)) return Promise.resolve(audioUrls.get(text));
    if (audioReqs.has(text)) return audioReqs.get(text);
    const p = fetch(TTS_PATH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, sig }),
    }).then((r) => {
      if (r.status === 503) { voiceUnavailable(); throw new Error('tts-unavailable'); }
      if (!r.ok) throw new Error(`tts ${r.status}`);
      return r.blob();
    }).then((blob) => {
      const url = URL.createObjectURL(blob);
      audioUrls.set(text, url);
      audioReqs.delete(text);
      return url;
    }).catch((e) => {
      audioReqs.delete(text);
      throw e;
    });
    audioReqs.set(text, p);
    return p;
  }

  /* warm every beat's clip in order so playback lands on the beat */
  function prefetchBeats() {
    if (voiceGone || !nar || nar.status !== 'ready') return;
    let chain = Promise.resolve();
    for (const b of nar.beats) {
      chain = chain
        .then(() => fetchBeatAudio(b.text, b.sig))
        .catch(() => { /* one failed clip must not block the rest */ });
    }
  }

  function speakBeat(text, sig) {
    if (voiceGone) return;
    cancelSpeech();             /* swap semantics - one live clip */
    const myToken = token;
    /* 250 ms breath between beats - calmer than back-to-back speech */
    voiceTimer = setTimeout(() => {
      voiceTimer = null;
      fetchBeatAudio(text, sig).then((url) => {
        /* stale if the sim closed, voice toggled off or playback paused
           while the clip was fetching */
        if (myToken !== token || !nar || !nar.voice || !sim || !sim.playing) return;
        const a = new Audio(url);
        a.addEventListener('ended', () => {
          if (audioEl === a) audioEl = null;
          spokenDone.add(text);
        });
        audioEl = a;
        a.play().catch(() => { /* autoplay denied - captions carry on */ });
      }).catch(() => { /* clip unavailable - captions carry on */ });
    }, VOICE_GAP_MS);
  }

  /* speak the beat under the cursor NOW - dead air killed. Fires when
     VOICE arms, when narration turns on with voice already armed, and on
     resume-after-pause (which skips beats whose clip already finished). */
  function speakCurrent(force) {
    if (!nar || !nar.on || !nar.voice || nar.status !== 'ready') return;
    if (!sim || !sim.playing) return;
    const b = nar.idx >= 0 ? nar.beats[nar.idx] : null;
    if (!b) return;
    if (!force && spokenDone.has(b.text)) return;
    speakBeat(b.text, b.sig);
  }

  function setCaption(text) {
    if (!refs || !refs.caption) return;
    refs.caption.textContent = text || '';
    refs.caption.hidden = !text;
  }

  /* latest beat at or before t; -1 before the first beat */
  function beatIndexFor(t) {
    let i = -1;
    for (let k = 0; k < nar.beats.length; k++) {
      if (nar.beats[k].t_rel_s <= t) i = k; else break;
    }
    return i;
  }

  /* any seek: recompute the current beat. A seek that crosses a beat
     boundary (either direction) cancels the now-stale clip and, with voice
     on, speaks the beat that owns the new position - a forward scrub never
     leaves the previous beat's audio playing under a newer caption. */
  function resyncBeats(prevTRel) {
    if (!nar) return;
    if (Number.isFinite(prevTRel) && sim.tRel < prevTRel
        && nar.status !== 'ready') cancelSpeech();
    if (!nar.on || nar.status !== 'ready') return;
    const idx = beatIndexFor(sim.tRel);
    if (idx !== nar.idx) {
      cancelSpeech();
      nar.idx = idx;
      setCaption(idx >= 0 ? nar.beats[idx].text : '');
      if (nar.voice && idx >= 0) speakCurrent(true);
    } else {
      setCaption(nar.idx >= 0 ? nar.beats[nar.idx].text : '');
    }
  }

  /* rAF: advance to the next beat when its time arrives (next-beat swap) */
  function tickBeats() {
    if (!nar || !nar.on || nar.status !== 'ready') return;
    let changed = false;
    while (nar.idx + 1 < nar.beats.length
      && nar.beats[nar.idx + 1].t_rel_s <= sim.tRel) {
      nar.idx += 1;
      changed = true;
    }
    if (changed) {
      const b = nar.beats[nar.idx];
      setCaption(b.text);
      if (nar.voice) speakBeat(b.text, b.sig);
    }
  }

  function narrationOff() {
    nar.on = false;
    refs.narBtn.classList.remove('is-active');
    refs.narBtn.setAttribute('aria-pressed', 'false');
    setCaption('');
    cancelSpeech();
  }

  async function toggleNarrate() {
    if (!nar || nar.gone || nar.status === 'loading' || aiGone) return;
    if (nar.on) { narrationOff(); return; }
    if (nar.status === 'ready') {   /* beats cached for the session */
      nar.on = true;
      refs.narBtn.classList.add('is-active');
      refs.narBtn.setAttribute('aria-pressed', 'true');
      resyncBeats(NaN);
      speakCurrent(true);           /* voice armed -> speak from here */
      return;
    }
    nar.status = 'loading';
    setCaption('Writing the narration…');
    const myToken = token;
    let resp = null;
    let err = null;
    try {
      resp = await postJson(NARRATE_PATH, { conjunction_id: openConjId });
    } catch (e) { err = e; }
    if (myToken !== token || !overlay || !nar) return;   /* closed meanwhile */
    if (err) {
      if (err.status === 404) {
        /* endpoint absent (older server) - hide the feature */
        nar.gone = true;
        nar.status = 'idle';
        refs.narBtn.hidden = true;
        refs.voiceBtn.hidden = true;
        setCaption('');
        collectFocusables();
        return;
      }
      nar.status = 'error';
      if (err.status === 503) {
        /* no Anthropic key: one calm line, then Narrate + Voice go quiet */
        aiGone = true;
        try { sessionStorage.setItem(AI_NOKEY_KEY, '1'); } catch (_) { /* blocked */ }
        nar.status = 'idle';
        setCaption(AI_NOKEY_TEXT);
        syncAiButtons();
        setTimeout(() => { if (nar && !nar.on) setCaption(''); }, 6000);
        return;
      }
      setCaption(`Narration is unavailable right now (${err.detail || err.message || 'unknown error'}).`);
      /* the banner must not ride over the hero view for the whole session */
      setTimeout(() => { if (nar && nar.status === 'error') setCaption(''); }, 8000);
      return;
    }
    const beats = (resp && Array.isArray(resp.beats) ? resp.beats : [])
      .filter((b) => b && Number.isFinite(b.t_rel_s) && typeof b.text === 'string' && b.text)
      .map((b) => ({ t_rel_s: b.t_rel_s, text: b.text, sig: typeof b.sig === 'string' ? b.sig : null }))
      .sort((a, b) => a.t_rel_s - b.t_rel_s);
    if (!beats.length) {
      nar.status = 'error';
      setCaption('Narration came back empty.');
      setTimeout(() => { if (nar && nar.status === 'error') setCaption(''); }, 8000);
      return;
    }
    nar.beats = beats;
    nar.status = 'ready';
    nar.on = true;
    refs.narBtn.classList.add('is-active');
      refs.narBtn.setAttribute('aria-pressed', 'true');
    if (nar.voice) prefetchBeats();   /* voice already armed - warm the clips */
    resyncBeats(NaN);
    speakCurrent(true);               /* no dead air waiting for a swap */
  }

  function toggleVoice() {
    if (!nar || nar.gone || voiceGone || aiGone) return;
    nar.voice = !nar.voice;
    refs.voiceBtn.classList.toggle('is-active', nar.voice);
    refs.voiceBtn.setAttribute('aria-pressed', String(nar.voice));
    if (!nar.voice) { cancelSpeech(); return; }   /* toggle-off cancels voice */
    /* voice reads the narration: arming it with narration off starts the
       narration too, so the toggle never reads "on" while nothing plays */
    if (!nar.on) { toggleNarrate(); return; }
    prefetchBeats();
    speakCurrent(true);               /* audible immediately, not next beat */
  }

  /* A2: the latest TLEs and the latest message disagree (server-side
     check on the ephemeris, the same rule the avoidance plan uses for
     geometry_mismatch) - say so calmly at open, before any number is read. */
  function buildMismatchNotice() {
    if (!ephem || ephem.geometry_consistent !== false) return null;
    const box = el('div', 'ksim-notice');
    box.setAttribute('role', 'status');
    const ic = el('span', 'ksim-notice__icon');
    ic.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 9v5M10 6.2v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
    box.appendChild(ic);
    const txt = el('div', 'ksim-notice__txt');
    txt.appendChild(el('strong', '', 'The latest tracking data disagree with this event’s last message. '));
    const ts = ephem.tca_state || {};
    txt.appendChild(document.createTextNode(
      `The message put the pair ${FMT.dist(ephem.latest_message_miss_m)} apart; the newest orbits, which this replay uses, put them ${FMT.dist(ts.sep_m)} apart. Treat the event by its message until fresher tracking arrives.`));
    box.appendChild(txt);
    const x = btn('ksim-notice__close', null, 'Dismiss');
    x.innerHTML = ICON_CLOSE;
    addL(x, 'click', () => {
      if (box.parentNode) box.parentNode.removeChild(box);
      refs.stage.classList.remove('has-notice');
      collectFocusables();
    });
    box.appendChild(x);
    return box;
  }

  function buildLayout() {
    refs.stage.className = 'ksim__main';
    clearChildren(refs.stage);
    const notice = buildMismatchNotice();
    if (notice) {
      refs.stage.classList.add('has-notice');
      refs.stage.appendChild(notice);
    }

    /* left - the space stage (dark in both themes) with glass overlays */
    const rel = el('section', 'ksim__rel ksim-space');
    rel.setAttribute('aria-label', 'Encounter view');
    refs.relCanvas = mkCanvas('ksim__relcanvas');
    rel.appendChild(viewsOk ? refs.relCanvas : nodata('Renderers unavailable'));

    const top = el('div', 'ksim-ov ksim-ov--top');
    const tl = el('div', 'ksim-ov__stack');
    refs.phaseLbl = el('div', 'ksim-pill ksim-phase', '');
    refs.phaseLbl.setAttribute('aria-live', 'polite');
    tl.appendChild(refs.phaseLbl);
    refs.burnBadge = el('div', 'ksim-pill ksim-badge');
    refs.burnBadge.hidden = true;
    tl.appendChild(refs.burnBadge);
    top.appendChild(tl);
    const tr = el('div', 'ksim-ov__stack ksim-ov__stack--r');
    const vseg = el('div', 'seg ksim-glass-seg');
    vseg.setAttribute('role', 'group');
    vseg.setAttribute('aria-label', 'View');
    refs.viewRelBtn = btn('btn btn--xs is-active', 'Encounter plane');
    refs.viewRelBtn.setAttribute('aria-pressed', 'true');
    refs.viewCamBtn = btn('btn btn--xs', 'Space camera');
    refs.viewCamBtn.setAttribute('aria-pressed', 'false');
    vseg.appendChild(refs.viewRelBtn);
    vseg.appendChild(refs.viewCamBtn);
    tr.appendChild(vseg);
    refs.axisCap = el('div', 'ksim-axiscap',
      'Looking along the approach: the plane perpendicular to relative velocity');
    tr.appendChild(refs.axisCap);
    top.appendChild(tr);
    rel.appendChild(top);

    const bot = el('div', 'ksim-ov ksim-ov--bottom');
    const hero = el('div', 'ksim-hero');
    refs.heroLbl = el('div', 'ksim-hero__label', 'Separation');
    hero.appendChild(refs.heroLbl);
    refs.sepVal = el('div', 'ksim-hero__value', '—');
    hero.appendChild(refs.sepVal);
    /* plain-language line - what will happen, in words */
    refs.plain = el('div', 'ksim-plain', '');
    hero.appendChild(refs.plain);
    bot.appendChild(hero);
    /* narration caption - controller-owned DOM line over the view bottom */
    refs.caption = el('div', 'ksim-caption');
    refs.caption.hidden = true;
    refs.caption.setAttribute('aria-live', 'polite');
    bot.appendChild(refs.caption);
    refs.zoomBox = el('div', 'ksim-zoom');
    refs.zoomInfo = el('span', 'ksim-zoom__info', '');
    refs.autoBtn = btn('btn btn--xs is-active', 'Auto zoom');
    refs.autoBtn.setAttribute('aria-pressed', 'true');
    refs.autoBtn.title = 'Scroll on the view to zoom manually';
    refs.zoomBox.appendChild(refs.zoomInfo);
    refs.zoomBox.appendChild(refs.autoBtn);
    bot.appendChild(refs.zoomBox);
    rel.appendChild(bot);
    refs.stage.appendChild(rel);
    addL(refs.viewRelBtn, 'click', () => setView('rel'));
    addL(refs.viewCamBtn, 'click', () => setView('cam'));

    /* right - avoidance first (the decision), then the evidence */
    const side = el('div', 'ksim__side');
    /* WHAT-IF / AVOIDANCE - future events only (hidden when tca < now) */
    if (tcaMs >= Date.now()) side.appendChild(buildManeuver());
    else {
      const past = el('section', 'panel ksim-avoid');
      past.appendChild(panelHd('Avoidance'));
      past.appendChild(el('p', 'ksim-avoid__intro ksim-avoid__intro--past',
        'This closest approach has passed. Replays are for review; burns can only be planned for upcoming events.'));
      side.appendChild(past);
    }

    const rng = el('section', 'panel');
    rng.appendChild(panelHd('Separation over time', 'Log scale'));
    const rngBd = el('div', 'ksim__rngbd');
    refs.rangeCanvas = mkCanvas('ksim__rngcanvas');
    rngBd.appendChild(viewsOk ? refs.rangeCanvas : nodata('Renderers unavailable'));
    rng.appendChild(rngBd);
    side.appendChild(rng);

    const ctx = el('section', 'panel');
    ctx.appendChild(panelHd('Where they are', 'Earth-centered'));
    const ctxBd = el('div', 'ksim__ctxbd');
    refs.ctxCanvas = mkCanvas('ksim__ctxcanvas');
    ctxBd.appendChild(viewsOk ? refs.ctxCanvas : nodata('Renderers unavailable'));
    ctx.appendChild(ctxBd);
    side.appendChild(ctx);

    const rd = el('section', 'panel ksim__rdpanel');
    rd.appendChild(panelHd('Details'));
    rd.appendChild(buildReadouts());
    side.appendChild(rd);

    refs.stage.appendChild(side);

    /* footer - transport bar */
    refs.footer = el('footer', 'ksim__ft');
    refs.playBtn = btn('ksim__play');
    refs.footer.appendChild(refs.playBtn);
    renderPlayBtn();
    refs.tVal = el('span', 'ksim__tval', fmtVsTca(sim.tRel));
    refs.footer.appendChild(refs.tVal);
    refs.footer.appendChild(buildTimeline());
    const spd = el('div', 'seg ksim__speeds');
    spd.setAttribute('role', 'group');
    spd.setAttribute('aria-label', 'Playback speed');
    refs.speedBtns = SPEEDS.map((s) => {
      const b = btn('btn btn--xs', `${s < 1 ? s.toFixed(1) : s}×`, `Speed ${s} times`);
      const on = s === sim.speed;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', String(on));
      spd.appendChild(b);
      addL(b, 'click', () => { if (!exp) setSpeed(s); });
      return b;
    });
    refs.footer.appendChild(spd);
    /* export - meta shows "Recording <n> s" while the pass runs */
    refs.expMeta = el('span', 'ksim__expmeta', '');
    refs.expMeta.hidden = true;
    refs.footer.appendChild(refs.expMeta);
    refs.expBtn = btn('btn btn--xs ksim__expbtn', 'Export video');
    refs.expBtn.title = 'Record a 12-second WebM of the pass';
    const sup = detectExportSupport();
    exportOk = sup.ok;
    exportWhy = sup.why;
    if (!exportOk) {
      refs.expBtn.disabled = true;
      refs.expBtn.title = exportWhy;
    }
    refs.footer.appendChild(refs.expBtn);
    overlay.appendChild(refs.footer);

    /* header narration buttons become live with the layout */
    refs.narBtn.hidden = false;
    refs.voiceBtn.hidden = false;
    syncAiButtons();            /* known no-key / no-voice -> disabled + tooltip */
    addL(document, 'kessler:ai-availability', onAiAvailability);

    addL(refs.playBtn, 'click', () => { if (!exp) setPlaying(!sim.playing); });
    addL(refs.autoBtn, 'click', () => { if (!exp) setZoomMode('AUTO'); });
    addL(refs.expBtn, 'click', startExport);
    if (viewsOk) addL(refs.relCanvas, 'wheel', onWheel, { passive: false });
    if (mnv && refs.mnvDv) {
      renderDv();
      syncDirButtons();
      setBurn(MNV_BURN_DEFAULT);
    }
  }

  /* tab-trap cycle: every visible, enabled control in DOM order */
  function collectFocusables() {
    if (!overlay) return;
    focusables = Array.from(overlay.querySelectorAll(
      'button, input, [tabindex="0"]'))
      .filter((b) => !b.disabled && b.getClientRects().length > 0);
  }

  function fillHeader() {
    refs.pair.textContent = '';
    refs.pair.appendChild(el('span', '', conj.primary.name));
    refs.pair.appendChild(el('span', 'ksim__and', 'and'));
    refs.pair.appendChild(el('span', '', conj.secondary.name));
    const lvl = conj.risk_level || 'nominal';
    refs.chip.hidden = false;
    refs.chip.className = `chip chip--${lvl}`;
    refs.chip.textContent = lvl.charAt(0).toUpperCase() + lvl.slice(1);
    const eff = conj.predicted_final_risk != null ? conj.predicted_final_risk : conj.pc_latest;
    const t = new Date(conj.tca);
    const when = Number.isFinite(t.getTime())
      ? `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]}, ${conj.tca.slice(11, 19)} UTC`
      : '—';
    refs.hdMeta.textContent =
      `Closest approach ${when} · Probability ${FMT.pc(eff)} · Model ${conj.model_version || '—'}`;
    refs.hdMeta.title = `TCA ${FMT.utc(conj.tca)}`;
  }

  /* --- open / load / close ---------------------------------------------------------- */

  function initSim() {
    /* per-sample separations (km, grid order) for drawRange - precomputed
       once per load (contract) so the renderer never recomputes per frame */
    const rangeCache = new Array(N);
    for (let j = 0; j < N; j++) {
      const a = ephem.primary.r[j];
      const b = ephem.secondary.r[j];
      rangeCache[j] = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    sim = {
      ephem,
      tRel: -W,
      pr: [0, 0, 0],
      sr: [0, 0, 0],
      relKm: [0, 0, 0],
      sepKm: 0,
      speed: DEFAULT_SPEED,
      playing: true,
      view: 'rel',        /* main-panel mode: 'rel' | 'cam' */
      zoom: { mode: 'AUTO', pxPerKm: PXKM_MIN },
      trailRel: [],
      basis: computeBasis(),
      rangeCache,
      pathCache: null,    /* flyby path - built by ensurePathCache */
      pulse: null,        /* TCA pulse - normalized 0..1, null idle */
      maneuver: null,
      fmtDist: (m) => FMT.dist(m),   /* canvas labels use the page formatter (C9) */
      header: {
        primaryName: conj.primary.name,
        secondaryName: conj.secondary.name,
        riskLevel: conj.risk_level || null,
      },
    };
    mnv = {
      dv: MNV_DV_DEFAULT, dir: 1, burnH: MNV_BURN_DEFAULT, touched: false,
      burnMs: tcaMs - MNV_BURN_DEFAULT * 3600000, busy: false, openUi: false,
      ghostRel: null, ghostKm: 0,
    };
    plan = { status: 'idle', data: null, err: null, pick: null, promise: null };
    nar = {
      on: false, voice: false, status: 'idle', beats: [], idx: -1, gone: false,
    };
    pulseT = null;
    pulseArmed = true;
    const ts0 = ephem.tca_state;
    missM = ts0 && Number.isFinite(ts0.sep_m)
      ? ts0.sep_m : vMag(relAt(0)) * 1000;   /* plain-language miss distance */
    /* in-plane radius of the TCA miss point: AUTO zoom never zooms in
       past it, so the closest-approach ghost stays on screen */
    const relT = relAt(0);
    sim.tcaInPlaneKm = Math.hypot(vDot(relT, sim.basis.e1), vDot(relT, sim.basis.e2));
    interpTo(sim.tRel);
    rebuildTrail();
  }

  async function load(myToken) {
    setStage('loading');
    let c, e;
    try {
      [c, e] = await Promise.all([
        API.conjunction(openConjId),
        getJson(`/conjunctions/${openConjId}/ephemeris?window_s=${WINDOW_S}`),
      ]);
    } catch (err) {
      if (myToken !== token || !overlay) return;
      setStage('error', (err && err.message) || String(err));
      return;
    }
    if (myToken !== token || !overlay) return;   /* closed while fetching */
    conj = c;
    ephem = e;
    W = Math.max(1, Math.round(ephem.window_s || WINDOW_S));
    const pr = (ephem.primary && ephem.primary.r) || [];
    const sr = (ephem.secondary && ephem.secondary.r) || [];
    N = Math.min(pr.length, sr.length);
    tcaMs = new Date(ephem.tca || conj.tca).getTime();
    fillHeader();
    if (N < 2) { setStage('degrade'); return; }
    initSim();
    buildLayout();
    zoomTarget = autoZoomTarget();
    sim.zoom.pxPerKm = zoomTarget;
    ready = true;
    collectFocusables();
    lastTs = null;
    rafId = requestAnimationFrame(frame);
    if (applyPlanOnLoad) {
      applyPlanOnLoad = false;
      planAvoidance({ apply: true });
    }
  }

  /* Public: compute the avoidance plan on the open simulator; with
     {apply:true} the recommended burn is loaded, simulated and replayed.
     Resolves to the plan payload (null when unavailable). */
  async function planAvoidance(opts) {
    if (!overlay || !ready || !refs || !refs.planBtn) return null;
    const data = await requestPlan();
    if (data && opts && opts.apply && data.recommended) {
      await applyPlanOption(data.recommended);
    }
    return data;
  }

  /* Safe to call right after page load (e.g. a /?sim=<id> deep link): until
     the DOM and the renderers are ready the request is queued (last call
     wins) and replayed on DOMContentLoaded. */
  function depsReady() {
    return document.readyState !== 'loading' && !!document.body &&
      typeof API !== 'undefined' && typeof FMT !== 'undefined';
  }

  async function open(opts) {
    const id = opts && opts.conjunctionId;
    if (id == null) return;
    if (!depsReady()) {
      const first = pendingOpen == null;
      pendingOpen = opts;
      if (first) {
        document.addEventListener('DOMContentLoaded', () => {
          const o = pendingOpen;
          pendingOpen = null;
          if (o) open(o);
        }, { once: true });
      }
      return;
    }
    if (overlay && openConjId === id) {          /* idempotent */
      if (opts.applyPlan) {
        if (ready) planAvoidance({ apply: true });
        else applyPlanOnLoad = true;             /* still loading: apply on ready */
      }
      return;
    }
    if (overlay) close();                        /* switching events (resets the flag) */
    applyPlanOnLoad = !!opts.applyPlan;
    openConjId = id;
    const myToken = ++token;
    prevFocus = document.activeElement || null;
    prevBodyOverflow = document.body.style.overflow || '';
    document.body.style.overflow = 'hidden';
    document.body.classList.add('ksim-open');   /* assistant launcher steps away */
    listeners = [];
    viewsOk = typeof SimViews !== 'undefined' && !!SimViews;
    buildOverlay(id);
    if (overlay.focus) overlay.focus();
    await load(myToken);
  }

  function close() {
    if (!overlay) return;
    if (exp) {
      /* closing mid-export = cancel: onstop's closure sees cancelled=true */
      const state = exp;
      exp = null;
      state.cancelled = true;
      try {
        if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop();
      } catch (_) { /* recorder already dead */ }
    }
    cancelSpeech();                       /* voice stops with the overlay */
    nar = null;
    for (const url of audioUrls.values()) {
      try { URL.revokeObjectURL(url); } catch (_) { /* already released */ }
    }
    audioUrls = new Map();
    audioReqs = new Map();
    spokenDone = new Set();
    pulseT = null;
    pulseArmed = true;
    token++;
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
    for (const [t, ty, fn, opts] of listeners) t.removeEventListener(ty, fn, opts);
    listeners = null;
    if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    overlay = null;
    refs = null;
    focusables = null;
    conj = null;
    ephem = null;
    sim = null;
    mnv = null;
    plan = null;
    applyPlanOnLoad = false;
    ready = false;
    openConjId = null;
    dragging = false;
    lastTs = null;
    document.body.style.overflow = prevBodyOverflow;
    document.body.classList.remove('ksim-open');
    if (prevFocus && prevFocus.focus) { try { prevFocus.focus(); } catch (_) { /* gone */ } }
    prevFocus = null;
  }

  return { open, close, planAvoidance };
})();

/* A page script that runs before this file (dashboard.js loads first) can
   leave a request in window.KSIM_PENDING = {conjunctionId, applyPlan}; it is
   opened as soon as the simulator exists. */
if (typeof window !== 'undefined' && window.KSIM_PENDING && window.KSIM_PENDING.conjunctionId != null) {
  const pendingSim = window.KSIM_PENDING;
  window.KSIM_PENDING = null;
  KSim.open(pendingSim);
}
