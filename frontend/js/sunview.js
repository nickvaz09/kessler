/* Kessler - the Sun -> Earth view (the Overview's "Space weather" lens).
   Global: KSunView.mount(canvas) -> view
     view.set(params, snap) live conditions {speed, density, bz, kp, ratio}; eased
                           unless snap (the first reading is shown as it is)
     view.setInsets({left, right, top, bottom})   floating UI to keep clear of
     view.start() / view.stop() / view.simulate(seconds) / view.destroy()

   It draws what NOAA reports now - no scenarios, no staged events.

   One canvas, back to front: the Sun · solar wind (a Rankine half-body flow
   around the magnetopause, so the wind really is deflected by the shield) ·
   bow shock · magnetopause · field lines (dipole, compressed on the day
   side, stretched into the tail; open lobes from the polar caps) · GEO belt
   · upper atmosphere (thickens and warms with the storm's density ratio) ·
   Earth (the real day/night Earth, Sun to the left, with this moment's
   season and dipole tilt) · auroral ovals around the geomagnetic poles ·
   particles precipitating down the field into the ovals · a LEO satellite ·
   labels · bloom (a blurred additive pass).

   Distances are compressed (displayed radius = R_earth * r^0.78) so Earth
   and the shield fit one frame. The transform is monotonic, so the ordering
   - LEO < GEO < shield < bow shock - is exact: the GEO belt is drawn outside
   the shield exactly when the Shue standoff puts the magnetopause inside
   6.6 Earth radii. */
(function () {
  'use strict';

  const TWO_PI = Math.PI * 2;
  const DEG = Math.PI / 180;
  const KAPPA = 0.78;
  const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);
  const GEO_RE = 6.62;
  const SHOCK_X = 1.3;                     // bow shock standoff / magnetopause standoff
  const MAG_POLE = [80.8 * DEG, -72.7 * DEG];   // geomagnetic north pole (IGRF, 2025)
  const CAM_TILT = 0.30;                   // camera above the ecliptic (rad)
  const SAT_INC = 53 * DEG;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;

  function hash(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
  function noise(x) {                      // smooth 1-D value noise in [0, 1]
    const i = Math.floor(x), f = x - i;
    const u = f * f * (3 - 2 * f);
    return lerp(hash(i), hash(i + 1), u);
  }

  /* the same low-precision Sun and GMST the globe uses */
  function gmstRad(ms) {
    const d = (ms - J2000_MS) / 86400000;
    return ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
  }
  function sunEci(ms) {
    const d = (ms - J2000_MS) / 86400000;
    const g = (357.529 + 0.98560028 * d) * DEG;
    const L = (280.459 + 0.98564736 * d) * DEG + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
    const eps = (23.439 - 0.00000036 * d) * DEG;
    return [Math.cos(L), Math.cos(eps) * Math.sin(L), Math.sin(eps) * Math.sin(L)];
  }

  /* Shue et al. 1998 standoff (Earth radii) from the solar wind */
  function standoffRe(density, speed, bz) {
    const p = 1.6726e-6 * Math.max(0.5, density) * speed * speed;
    return (10.22 + 1.29 * Math.tanh(0.184 * ((bz || 0) + 8.14))) * Math.pow(p, -1 / 6.6);
  }

  /* Rankine half-body (uniform flow + source at Earth): the dividing
     streamline r(phi) = b (pi - phi) / sin(phi), phi from +X (downstream);
     nose at r = b, asymptotic half-width pi*b - a close match to the Shue
     magnetopause, and an exact potential flow for the wind around it. */
  function bodyR(b, phi) {
    const s = Math.sin(phi);
    if (s < 1e-5) return phi > 1.5 ? b : 1e9;
    return b * (Math.PI - phi) / s;
  }
  function inBody(X, Y, b) {
    return Math.hypot(X, Y) < bodyR(b, Math.atan2(Math.abs(Y), X));
  }

  function mount(canvas) {
    const ctx = canvas.getContext('2d');
    const glow = document.createElement('canvas');
    const gctx = glow.getContext('2d');
    const glow2 = document.createElement('canvas');      // the blurred glow, same low res
    const g2ctx = glow2.getContext('2d');
    const earthC = document.createElement('canvas');
    const ectx = earthC.getContext('2d');
    const canBlur = typeof gctx.filter === 'string';
    const reduce = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
    const font = (getComputedStyle(document.documentElement).getPropertyValue('--k-sans') || '').trim()
      || 'Inter, system-ui, -apple-system, sans-serif';

    let W = 0, H = 0, dpr = 1, gs = 0.5;
    let inset = { left: 0, right: 0, top: 0, bottom: 0 };
    let running = false, raf = 0, lastTs = 0, T = 0;

    /* live conditions, eased toward each new NOAA reading */
    const cur = { speed: 420, density: 5, bz: 0, kp: 1.3, ratio: 1.0 };
    const tgt = Object.assign({}, cur);

    /* derived each frame */
    let G = null;                           // layout
    let standoff = 10.5, rShield = 0, rShock = 0;
    let earthAt = -1e9;                     // T of the last earth-texture render
    let orient = null;                      // camera + dipole, refreshed each second

    /* --- particles: solar wind ---------------------------------------------- */
    const NW = 420, HL = 9;
    const wx = new Float32Array(NW), wy = new Float32Array(NW), wSeed = new Float32Array(NW);
    const wh = new Float32Array(NW * HL * 2);   // history ring per particle
    const wSheath = new Uint8Array(NW);
    let wHead = 0, nWind = 0, windReady = false, wTick = 0;

    /* precipitation: particles riding field lines into the ovals */
    const NP = 44;
    const pS = new Float32Array(NP), pLine = new Uint8Array(NP), pHemi = new Int8Array(NP), pSpd = new Float32Array(NP);
    let pAlive = 0;

    /* field-line geometry, rebuilt when the shield moves > 1 px */
    let lines = [], linesKey = '';
    let precipPaths = [];

    /* insets: floating UI to frame around; capRight/capBottom: a card in the
       top-left corner the labels must not run under */
    let layoutDirty = true;
    function setInsets(o) {
      inset = Object.assign({ left: 0, right: 0, top: 0, bottom: 0, capRight: 0, capBottom: 0 }, o || {});
      layoutDirty = true;                      // applied at the next step, never mid-step
    }

    function resize() {
      const r = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
      /* a pixel budget, not just a DPR cap: an ultrawide at 2x would be a
         17-megapixel canvas redrawn every frame */
      const d = Math.min(2, window.devicePixelRatio || 1, Math.sqrt(8.5e6 / Math.max(1, w * h)));
      if (w === W && h === H && Math.abs(d - dpr) < 1e-3) return false;
      W = w; H = h; dpr = d;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      glow.width = glow2.width = Math.max(1, Math.round(W * gs));
      glow.height = glow2.height = Math.max(1, Math.round(H * gs));
      G = null; earthAt = -1e9; windReady = false;
      return true;
    }

    function layout() {
      const fx0 = inset.left, fx1 = W - inset.right;
      const fy0 = inset.top, fy1 = H - inset.bottom;
      const fw = Math.max(240, fx1 - fx0), fh = Math.max(200, fy1 - fy0);
      const sunX = fx0 + clamp(fw * 0.05, 30, 150);
      /* Earth sits a fixed gap of wind past the quiet-day bow shock, so the
         frame never jumps when the shield moves */
      const shockK = Math.pow(10.8 * SHOCK_X, KAPPA);
      const gap = clamp(fw * 0.2, 90, 560);
      let R = clamp(Math.min(fw * 0.05, fh * 0.085), 18, 96);
      while (R > 18 && sunX + gap + R * shockK > fx0 + fw * 0.7) R -= 1;
      const ex = sunX + gap + R * shockK, ey = fy0 + fh * 0.53;
      return { fx0, fx1, fy0, fy1, fw, fh, R, ex, ey, sunR: fh * 1.9, sunX, sunY: ey,
        fs: clamp(R * 0.19, 11, 15) };
    }
    const disp = (rRe) => G.R * Math.pow(Math.max(rRe, 0), KAPPA);

    /* --- orientation: Sun to the left, this moment's tilts ---------------------- */
    function orientation() {
      const now = Date.now();
      const s = sunEci(now);
      const gm = gmstRad(now);
      const spin = Math.PI - Math.atan2(s[1], s[0]);         // sun azimuth -> screen left
      const cS = Math.cos(spin), sS = Math.sin(spin), cT = Math.cos(CAM_TILT), sT = Math.sin(CAM_TILT);
      const view = (v) => {                                   // ECI -> camera (x right, z up, d toward us)
        const x1 = v[0] * cS - v[1] * sS, y1 = v[0] * sS + v[1] * cS;
        return [x1, y1 * sT + v[2] * cT, -(y1 * cT - v[2] * sT)];
      };
      const sv = view(s);
      const rot = -Math.atan2(sv[1], -sv[0]);                 // turn the projected Sun exactly left
      /* geomagnetic north pole: ECEF -> ECI (rotate by GMST) */
      const la = MAG_POLE[0], lo = MAG_POLE[1] + gm;
      const D = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
      const dv = view(D);
      const cR = Math.cos(rot), sR = Math.sin(rot);
      const ax = dv[0] * cR + dv[1] * sR, az = -dv[0] * sR + dv[1] * cR;
      const psi = Math.atan2(-ax, az);                        // dipole lean toward the Sun (screen)
      return { now, s, gm, spin, cS, sS, cT, sT, rot, cR, sR, D, psi };
    }

    /* ECI direction (unit sphere, radius rr in Earth radii, display-scaled) -> screen */
    const pj = { x: 0, y: 0, d: 0 };
    function project(v, rrDisp) {
      const o = orient;
      const x1 = v[0] * o.cS - v[1] * o.sS, y1 = v[0] * o.sS + v[1] * o.cS;
      const z2 = y1 * o.sT + v[2] * o.cT;
      const d = -(y1 * o.cT - v[2] * o.sT);
      const lx = x1 * rrDisp, ly = -z2 * rrDisp;
      pj.x = G.ex + lx * o.cR - ly * o.sR;
      pj.y = G.ey + lx * o.sR + ly * o.cR;
      pj.d = d;
      return pj;
    }

    /* --- params ------------------------------------------------------------- */
    /* new readings ease in; the first reading (snap) is shown as it is */
    function set(p, snap) {
      const q = p || {};
      for (const k of ['speed', 'density', 'bz', 'kp', 'ratio']) {
        if (Number.isFinite(q[k])) { tgt[k] = q[k]; if (snap) cur[k] = q[k]; }
      }
    }

    /* ======================================================================
       Step
       ==================================================================== */
    function step(dt) {
      T += dt;
      if (!G) return;
      const k = 1 - Math.exp(-dt / 1.1);
      for (const key in tgt) cur[key] += (tgt[key] - cur[key]) * k;
    }

    function eff() { return cur; }

    /* ======================================================================
       Solar wind
       ==================================================================== */
    function windCount() {
      const area = clamp((G.fw * G.fh) / (980 * 600), 0.35, 1.6);
      const dens = clamp(eff().density / 14, 0.45, 1.35);
      return Math.min(NW, Math.round(250 * area * dens + 40));
    }
    function spawnWind(i, anywhere) {
      const xL = -G.ex - 30, xR = W - G.ex + 20;
      for (let tries = 0; tries < 8; tries++) {
        const X = anywhere ? lerp(xL, xR * 0.2, Math.random()) : xL - Math.random() * 80;
        const Y = (Math.random() * 2 - 1) * (H * 0.62);
        if (!inBody(X, Y, rShield * 1.02)) { wx[i] = X; wy[i] = Y; break; }
        wx[i] = xL; wy[i] = (Math.random() < 0.5 ? -1 : 1) * H * 0.6;
      }
      wSeed[i] = Math.random() * 1000;
      for (let h = 0; h < HL; h++) { const j = (i * HL + h) * 2; wh[j] = wx[i]; wh[j + 1] = wy[i]; }
    }
    function stepWind(dt) {
      const want = windCount();
      if (!windReady) { for (let i = 0; i < NW; i++) spawnWind(i, true); windReady = true; nWind = want; }
      nWind += Math.sign(want - nWind) * Math.min(Math.abs(want - nWind), 3);
      const e = eff();
      const base = (e.speed / 420) * Math.max(70, G.fw * 0.12) * (reduce ? 0.3 : 1);
      const b = rShield, bs = rShock;
      const rec = (wTick++ % 3) === 0;          // trail sample every third step
      if (rec) wHead = (wHead + 1) % HL;
      for (let i = 0; i < nWind; i++) {
        let X = wx[i], Y = wy[i];
        const r2 = X * X + Y * Y + 1;
        let u = 1 + b * X / r2, v = b * Y / r2;
        const sheath = inBody(X, Y, bs) ? 1 : 0;
        wSheath[i] = sheath;
        let spd = base;
        if (sheath) {
          spd *= 0.62;
          const n = wSeed[i] + T * 1.4;
          u += (noise(n) - 0.5) * 0.35; v += (noise(n + 57.3) - 0.5) * 0.45;
        }
        X += u * spd * dt; Y += v * spd * dt;
        if (inBody(X, Y, b)) {                  // never inside the shield: slide along it
          const phi = Math.atan2(Math.abs(Y), X), rr = bodyR(b, phi) * 1.012;
          const m = rr / Math.hypot(X, Y); X *= m; Y *= m;
        }
        wx[i] = X; wy[i] = Y;
        if (rec) { const j = (i * HL + wHead) * 2; wh[j] = X; wh[j + 1] = Y; }
        if (X > W - G.ex + 40 || Math.abs(Y) > H * 0.72) spawnWind(i, false);
      }
    }
    function drawWind() {
      const ex = G.ex, ey = G.ey;
      /* threads: four fading segments down the sampled history, then the
         live head segment; batched per (class, segment) - ten strokes */
      const seg = [[8, 6], [6, 4], [4, 2], [2, 0]];   // history offsets, oldest first
      const alpha = [0.035, 0.09, 0.17, 0.3];
      ctx.save();
      ctx.lineCap = 'round';
      for (let cls = 0; cls < 2; cls++) {
        const rgb = cls ? '255, 232, 210' : '255, 190, 130';
        const k = cls ? 1.25 : 1;
        ctx.lineWidth = cls ? 1.15 : 1.0;
        for (let s = 0; s < 5; s++) {
          ctx.strokeStyle = `rgba(${rgb}, ${(s < 4 ? alpha[s] : 0.5) * k})`;
          ctx.beginPath();
          for (let i = 0; i < nWind; i++) {
            if (wSheath[i] !== cls) continue;
            if (s < 4) {
              const a = (i * HL + ((wHead - seg[s][0] + HL * 2) % HL)) * 2;
              const c = (i * HL + ((wHead - seg[s][1] + HL * 2) % HL)) * 2;
              ctx.moveTo(ex + wh[a], ey + wh[a + 1]);
              ctx.lineTo(ex + wh[c], ey + wh[c + 1]);
            } else {
              const a = (i * HL + wHead) * 2;
              ctx.moveTo(ex + wh[a], ey + wh[a + 1]);
              ctx.lineTo(ex + wx[i], ey + wy[i]);
            }
          }
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    /* ======================================================================
       Shields: magnetopause and bow shock
       ==================================================================== */
    function bodyPath(c, b, sx, sy, scale, phiMin) {
      const n = 90;
      c.beginPath();
      for (let k = 0; k <= n; k++) {           // upper flank -> nose
        const phi = lerp(phiMin, Math.PI, k / n);
        const r = bodyR(b, phi);
        c.lineTo((sx + r * Math.cos(phi)) * scale, (sy - r * Math.sin(phi)) * scale);
      }
      for (let k = n; k >= 0; k--) {           // nose -> lower flank
        const phi = lerp(phiMin, Math.PI, k / n);
        const r = bodyR(b, phi);
        c.lineTo((sx + r * Math.cos(phi)) * scale, (sy + r * Math.sin(phi)) * scale);
      }
    }
    function phiOff(b) {                       // flank angle where the curve leaves the frame
      const lim = Math.max(W, H) * 1.1;
      let lo = 0.05, hi = Math.PI - 0.01;
      for (let k = 0; k < 24; k++) { const m = (lo + hi) / 2; if (bodyR(b, m) > lim) lo = m; else hi = m; }
      return lo;
    }
    function drawShields() {
      const { ex, ey } = G;
      const e = eff();
      /* the shield's inside: a faint cool wash, strongest along the boundary */
      const pm = phiOff(rShield);
      bodyPath(ctx, rShield, ex, ey, 1, pm);
      const wash = ctx.createRadialGradient(ex, ey, rShield * 0.35, ex, ey, rShield * 1.9);
      wash.addColorStop(0, 'rgba(70, 130, 255, 0)');
      wash.addColorStop(0.42, 'rgba(70, 130, 255, 0.025)');
      wash.addColorStop(0.53, 'rgba(90, 160, 255, 0.075)');
      wash.addColorStop(1, 'rgba(70, 130, 255, 0.0)');
      ctx.fillStyle = wash;
      ctx.fill();
      /* bow shock: heated plasma, warm and faint, fading toward the flanks */
      const bpm = phiOff(rShock);
      const fadeArc = (c, b, scale, rgb, a0, width) => {
        const n = 60;
        c.lineWidth = width * scale;
        for (let side = -1; side <= 1; side += 2) {
          for (let k = 0; k < n; k++) {
            const p0 = lerp(Math.PI, bpm, k / n), p1 = lerp(Math.PI, bpm, (k + 1) / n);
            const f = 1 - k / n;
            c.strokeStyle = `rgba(${rgb}, ${a0 * f * f})`;
            c.beginPath();
            const r0 = bodyR(b, p0), r1 = bodyR(b, p1);
            c.moveTo((ex + r0 * Math.cos(p0)) * scale, (ey + side * r0 * Math.sin(p0)) * scale);
            c.lineTo((ex + r1 * Math.cos(p1)) * scale, (ey + side * r1 * Math.sin(p1)) * scale);
            c.stroke();
          }
        }
      };
      const heat = 0.55 + 0.45 * clamp((e.speed - 380) / 500, 0, 1);
      fadeArc(ctx, rShock, 1, '255, 214, 178', 0.30 * heat, 1.1);
      fadeArc(gctx, rShock, gs, '255, 190, 140', 0.35 * heat, 5);
      /* magnetopause: crisp line + glow + a slow shimmer travelling along it */
      ctx.save();
      ctx.lineJoin = 'round';
      bodyPath(ctx, rShield, ex, ey, 1, pm);
      ctx.strokeStyle = 'rgba(132, 192, 255, 0.10)'; ctx.lineWidth = 7; ctx.stroke();
      ctx.strokeStyle = 'rgba(150, 205, 255, 0.62)'; ctx.lineWidth = 1.3; ctx.stroke();
      ctx.setLineDash([G.fh * 0.09, G.fh * 0.5]);
      ctx.lineDashOffset = -T * G.fh * 0.06;
      ctx.strokeStyle = 'rgba(205, 232, 255, 0.55)'; ctx.lineWidth = 1.8; ctx.stroke();
      ctx.restore();
      bodyPath(gctx, rShield, ex, ey, gs, pm);
      gctx.strokeStyle = 'rgba(110, 175, 255, 0.42)'; gctx.lineWidth = 6 * gs * 2; gctx.stroke();
    }

    /* ======================================================================
       Field lines
       ==================================================================== */
    const CLOSED_L = [1.55, 2.1, 2.8, 3.7, 4.9, 6.3, 8.1];
    function buildLines() {
      const key = `${Math.round(rShield)}|${Math.round(G.R)}|${orient ? orient.psi.toFixed(2) : 0}|${Math.round(G.ex)},${Math.round(G.ey)}`;
      if (key === linesKey) return;
      linesKey = key;
      lines = [];
      precipPaths = [];
      const psi = orient ? orient.psi : 0;
      const cP = Math.cos(psi), sP = Math.sin(psi);
      const R = G.R, b = rShield;
      /* closed dipole loops, three families (front, plane, back) */
      for (const fam of [{ phi: 0, a: 1 }, { phi: 0.85, a: 0.42 }, { phi: -0.85, a: 0.30 }]) {
        for (const L of (fam.phi === 0 ? CLOSED_L : [2.8, 4.9, 8.1])) {
          for (const side of [-1, 1]) {        // -1 day (sunward, left), +1 night
            const lam0 = Math.acos(Math.sqrt(1 / L));
            const pts = [];
            const n = 64;
            for (let k = 0; k <= n; k++) {
              const lam = lerp(-lam0, lam0, k / n);
              const rr = L * Math.cos(lam) * Math.cos(lam);
              /* dipole frame: x outward (side), z along the axis */
              let x = side * rr * Math.cos(lam) * Math.cos(fam.phi);
              let z = rr * Math.sin(lam);
              /* lean with the dipole tilt */
              const X0 = x * cP - z * sP, Z0 = x * sP + z * cP;
              const r = Math.hypot(X0, Z0) || 1e-6;
              let rd = R * Math.pow(r, KAPPA);
              let X = X0 / r * rd, Y = -Z0 / r * rd;
              if (X < 0) {                     // day side: soft-clamp inside the shield
                const lim = 0.93 * bodyR(b, Math.atan2(Math.abs(Y), X));
                const q = Math.hypot(X, Y);
                const qq = q / Math.pow(1 + Math.pow(q / lim, 5), 1 / 5);
                X *= qq / q; Y *= qq / q;
              } else {                         // night side: stretch into the tail
                const w = Math.pow(clamp(X / (Math.hypot(X, Y) || 1), 0, 1), 1.4);
                const s = (L / 8.1) * (L / 8.1);
                X *= 1 + 0.65 * w * s;
                Y *= 1 - 0.18 * w * (L / 8.1);
              }
              pts.push(X, Y);
            }
            lines.push({ pts, a: fam.a * (0.55 + 0.45 * (L / 8.1)), front: fam.phi > 0, plane: fam.phi === 0, L, side });
          }
        }
      }
      /* open lobes: polar cap -> over the pole -> down the tail */
      for (const hemi of [-1, 1]) {            // +1 north (screen up)
        for (let k = 0; k < 4; k++) {
          const colat = (8 + k * 3.2) * DEG;   // from the dipole axis
          const fx = Math.sin(colat), fz = Math.cos(colat);
          const ax0 = fx * cP - hemi * fz * sP, az0 = fx * sP + hemi * fz * cP;
          const p0x = ax0 * R * 1.0, p0y = -az0 * R * 1.0;
          const lobe = R * (1.55 + k * 0.42);
          const pts = [];
          const P = [[p0x, p0y], [p0x + R * (0.35 + 0.2 * k), p0y - hemi * R * (1.6 + 0.4 * k)],
            [R * (3.2 + k * 0.8), -hemi * lobe], [W, -hemi * lobe * (1 + 0.04 * k)]];
          for (let s = 0; s <= 48; s++) {
            const t = s / 48, u = 1 - t;
            pts.push(u * u * u * P[0][0] + 3 * u * u * t * P[1][0] + 3 * u * t * t * P[2][0] + t * t * t * P[3][0],
              u * u * u * P[0][1] + 3 * u * u * t * P[1][1] + 3 * u * t * t * P[2][1] + t * t * t * P[3][1]);
          }
          lines.push({ pts, a: 0.5 - k * 0.07, front: false, plane: true, open: true });
        }
      }
      /* precipitation rides the night-side plane loops that land on the oval */
      for (const ln of lines) {
        if (ln.plane && !ln.open && ln.L >= 3.7 && ln.side === 1) precipPaths.push(ln);
      }
    }
    function strokeLine(c, pts, scale, ox, oy) {
      c.beginPath();
      c.moveTo((ox + pts[0]) * scale, (oy + pts[1]) * scale);
      for (let k = 2; k < pts.length; k += 2) c.lineTo((ox + pts[k]) * scale, (oy + pts[k + 1]) * scale);
    }
    function drawLines(front) {
      const { ex, ey } = G;
      ctx.save();
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      for (const ln of lines) {
        if (ln.front !== front) continue;
        const a = ln.a * (ln.open ? 0.5 : 0.62);
        strokeLine(ctx, ln.pts, 1, ex, ey);
        ctx.strokeStyle = `rgba(108, 160, 240, ${a * 0.55})`;
        ctx.lineWidth = ln.plane ? 1.15 : 0.9;
        ctx.stroke();
        if (ln.plane && (ln.open || ln.L >= 4.9)) {
          ctx.setLineDash([3, 30]);
          ctx.lineDashOffset = -T * (ln.open ? 14 : 9) * (ln.side === 1 ? 1 : -1);
          ctx.strokeStyle = `rgba(175, 210, 255, ${a * 0.6})`;
          ctx.lineWidth = 1.3;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      ctx.restore();
      if (!front) {
        gctx.save();
        gctx.lineWidth = 2.2;
        for (const ln of lines) {
          if (!ln.plane) continue;
          strokeLine(gctx, ln.pts, gs, ex, ey);
          gctx.strokeStyle = `rgba(80, 140, 240, ${ln.a * 0.22})`;
          gctx.stroke();
        }
        gctx.restore();
      }
    }

    /* precipitation: bright beads sliding down the outer loops to the poles */
    function stepPrecip(dt) {
      const e = eff();
      const rate = clamp((e.kp - 4) / 5, 0, 1) * 0.7;
      const want = Math.round(NP * clamp(rate, 0, 1));
      while (pAlive < want && precipPaths.length) {
        const i = pAlive++;
        pS[i] = Math.random() * 0.5;
        pLine[i] = Math.floor(Math.random() * precipPaths.length);
        pHemi[i] = Math.random() < 0.5 ? -1 : 1;
        pSpd[i] = 0.16 + Math.random() * 0.22;
      }
      if (pAlive > want) pAlive = Math.max(want, pAlive - 2);
      for (let i = 0; i < pAlive; i++) {
        pS[i] += pSpd[i] * dt * 0.6 * (reduce ? 0.4 : 1);
        if (pS[i] >= 1) { pS[i] = 0; pLine[i] = Math.floor(Math.random() * precipPaths.length); pHemi[i] = -pHemi[i]; }
      }
    }
    function drawPrecip() {
      if (!pAlive || !precipPaths.length) return;
      const { ex, ey } = G;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (let i = 0; i < pAlive; i++) {
        const ln = precipPaths[pLine[i] % precipPaths.length];
        const n = ln.pts.length / 2 - 1, mid = n / 2;
        /* from the loop's apex (equator) down to one footpoint */
        const s = pS[i] * pS[i];
        const f = pHemi[i] > 0 ? mid * (1 - s) : mid + (n - mid) * s;
        const k = Math.min(n - 1, Math.floor(f)), u = f - k;
        const x = ex + lerp(ln.pts[2 * k], ln.pts[2 * k + 2], u);
        const y = ey + lerp(ln.pts[2 * k + 1], ln.pts[2 * k + 3], u);
        const a = 0.35 + 0.65 * pS[i];
        ctx.fillStyle = `rgba(140, 255, 196, ${0.55 * a})`;
        ctx.beginPath(); ctx.arc(x, y, 1.2 + 0.8 * pS[i], 0, TWO_PI); ctx.fill();
        gctx.fillStyle = `rgba(100, 255, 170, ${0.5 * a})`;
        gctx.beginPath(); gctx.arc(x * gs, y * gs, 2.4 * gs * 2, 0, TWO_PI); gctx.fill();
      }
      ctx.restore();
    }

    /* ======================================================================
       The Sun
       ==================================================================== */
    function drawSun() {
      const { sunR, sunX, sunY, fh } = G;
      const cx = sunX - sunR;
      /* corona: a wide warm glow beyond the limb */
      const cor = ctx.createRadialGradient(cx, sunY, sunR * 0.985, cx, sunY, sunR + G.fw * 0.42);
      cor.addColorStop(0, 'rgba(255, 196, 120, 0.42)');
      cor.addColorStop(0.12, 'rgba(255, 160, 80, 0.16)');
      cor.addColorStop(0.45, 'rgba(255, 130, 60, 0.035)');
      cor.addColorStop(1, 'rgba(255, 120, 50, 0)');
      ctx.fillStyle = cor;
      ctx.fillRect(0, 0, sunX + G.fw * 0.45, H);
      /* streamers: soft elongated glows off the limb, breathing slowly */
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (let k = 0; k < 7; k++) {
        const y0 = sunY + (k - 3) * fh * 0.16 + (noise(k * 9.1 + T * 0.05) - 0.5) * fh * 0.06;
        const len = G.fw * (0.10 + 0.12 * noise(k * 3.7 + T * 0.04));
        const wid = fh * (0.05 + 0.04 * hash(k));
        ctx.save();
        ctx.translate(sunX, y0);
        ctx.rotate((k - 3) * 0.05);
        ctx.scale(len / wid, 1);
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, wid);
        g.addColorStop(0, `rgba(255, 200, 140, ${0.06 + 0.035 * noise(k + T * 0.2)})`);
        g.addColorStop(1, 'rgba(255, 200, 140, 0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(0, 0, wid, 0, TWO_PI); ctx.fill();
        ctx.restore();
      }
      ctx.restore();
      /* the disk: limb-darkened, a sliver of a huge sphere */
      const disk = ctx.createRadialGradient(cx, sunY, 0, cx, sunY, sunR);
      disk.addColorStop(0, '#fff9ea');
      disk.addColorStop(0.9, '#ffe6ad');
      disk.addColorStop(0.965, '#ffc46a');
      disk.addColorStop(0.99, '#ff9440');
      disk.addColorStop(1, '#f2732a');
      ctx.fillStyle = disk;
      ctx.beginPath(); ctx.arc(cx, sunY, sunR, 0, TWO_PI); ctx.fill();
      /* surface texture: faint drifting cells near the limb */
      ctx.save();
      ctx.beginPath(); ctx.arc(cx, sunY, sunR, 0, TWO_PI); ctx.clip();
      ctx.globalCompositeOperation = 'multiply';
      for (let k = 0; k < 26; k++) {
        const y = (hash(k * 1.7) * 1.2 - 0.1) * H;
        const x = sunX - (6 + hash(k * 3.1) * G.fw * 0.035);
        const r = 8 + hash(k * 5.3) * 26;
        ctx.fillStyle = `rgba(240, 150, 70, ${0.07 + 0.05 * noise(k + T * 0.3)})`;
        ctx.beginPath(); ctx.arc(x, y, r, 0, TWO_PI); ctx.fill();
      }
      ctx.restore();
      /* glow layer */
      const gg = gctx.createRadialGradient(cx * gs, sunY * gs, sunR * gs * 0.99, cx * gs, sunY * gs, (sunR + G.fw * 0.12) * gs);
      gg.addColorStop(0, 'rgba(255, 200, 120, 0.55)');
      gg.addColorStop(1, 'rgba(255, 160, 80, 0)');
      gctx.fillStyle = gg;
      gctx.fillRect(0, 0, (sunX + G.fw * 0.15) * gs, H * gs);
    }

    /* ======================================================================
       Earth, its air, the aurora, the satellite, GEO
       ==================================================================== */
    function renderEarthTexture() {
      const R = G.R, m = Math.ceil(R * 0.1) + 2;
      const size = Math.ceil((R + m) * 2);
      if (earthC.width !== Math.round(size * dpr)) {
        earthC.width = Math.round(size * dpr); earthC.height = Math.round(size * dpr);
      }
      ectx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ectx.clearRect(0, 0, size, size);
      let ok = false;
      if (typeof KEarthTexture !== 'undefined' && KEarthTexture.ready()) {
        ok = KEarthTexture.drawHemisphere(ectx, { cx: size / 2, cy: size / 2, radiusPx: R,
          spin: orient.spin, tilt: CAM_TILT, gmstRad: orient.gm, sunDirTEME: orient.s });
      }
      if (!ok) {                                   // texture not loaded yet: a shaded sphere
        const g = ectx.createRadialGradient(size / 2 - R * 0.45, size / 2, R * 0.1, size / 2, size / 2, R);
        g.addColorStop(0, '#3d6fb0'); g.addColorStop(0.55, '#1b3a66'); g.addColorStop(1, '#050b16');
        ectx.fillStyle = g;
        ectx.beginPath(); ectx.arc(size / 2, size / 2, R, 0, TWO_PI); ectx.fill();
      }
      earthAt = T;
      return ok;
    }
    function drawEarth() {
      const R = G.R, m = Math.ceil(R * 0.1) + 2, size = Math.ceil((R + m) * 2);
      ctx.save();
      ctx.translate(G.ex, G.ey);
      ctx.rotate(orient.rot);
      ctx.drawImage(earthC, -size / 2, -size / 2, size, size);
      ctx.restore();
    }
    function drawAir(under) {
      const { ex, ey, R } = G;
      const e = eff();
      const heat = clamp((e.ratio - 1) / 2.4, 0, 1);
      const th = 1.045 + 0.06 * heat;     // the swollen shell
      const cool = [120, 178, 255], warm = [255, 150, 105];
      const c = cool.map((v, i) => Math.round(lerp(v, warm[i], 0.75 * heat)));
      if (under) {
        const g = ctx.createRadialGradient(ex, ey, R * 0.99, ex, ey, R * th);
        g.addColorStop(0, `rgba(${c}, ${0.46 + 0.2 * heat})`);
        g.addColorStop(0.4, `rgba(${c}, ${0.14 + 0.12 * heat})`);
        g.addColorStop(1, `rgba(${c}, 0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(ex, ey, R * th, 0, TWO_PI); ctx.fill();
        gctx.strokeStyle = `rgba(${c}, ${0.16 + 0.3 * heat})`;
        gctx.lineWidth = Math.max(2, R * (th - 1) * gs * 2);
        gctx.beginPath(); gctx.arc(ex * gs, ey * gs, R * (1 + (th - 1) * 0.5) * gs, 0, TWO_PI); gctx.stroke();
      } else {
        const g = ctx.createRadialGradient(ex, ey, R * 0.9, ex, ey, R);
        g.addColorStop(0, `rgba(${c}, 0)`);
        g.addColorStop(0.8, `rgba(${c}, ${0.03 + 0.04 * heat})`);
        g.addColorStop(1, `rgba(${c}, ${0.26 + 0.1 * heat})`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(ex, ey, R, 0, TWO_PI); ctx.fill();
      }
    }

    /* auroral ovals: geomagnetic colatitude grows with Kp (the oval moves
       toward the equator in a storm), brighter and wider on the night side */
    function ovalColat(kp) {
      return (90 - clamp(67.5 - 2.4 * kp, 44, 68)) * DEG;
    }
    function drawAurora() {
      const o = orient, e = eff();
      const D = o.D, S = o.s;
      /* magnetic frame: noon direction (sun, perpendicular to the axis), dusk */
      const dot = D[0] * S[0] + D[1] * S[1] + D[2] * S[2];
      let X = [S[0] - dot * D[0], S[1] - dot * D[1], S[2] - dot * D[2]];
      const xl = Math.hypot(X[0], X[1], X[2]) || 1; X = X.map((v) => v / xl);
      const Y = [D[1] * X[2] - D[2] * X[1], D[2] * X[0] - D[0] * X[2], D[0] * X[1] - D[1] * X[0]];
      const th0 = ovalColat(e.kp);
      const inten = clamp(0.25 + e.kp / 9 * 0.75, 0, 1);
      const n = 150;
      const rr = G.R * 1.025;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const hemi of [1, -1]) {
        for (let k = 0; k < n; k++) {
          const phi = (k / n) * TWO_PI;              // 0 = noon
          const night = (1 - Math.cos(phi)) / 2;
          const th = th0 + (2.8 * night - 1.4) * DEG;
          const wid = (2.4 + 3.6 * night) * DEG;
          const shimmer = reduce ? 0.8 : 0.55 + 0.45 * noise(k * 0.37 + T * 0.9 + (hemi > 0 ? 0 : 50));
          const a = inten * (0.35 + 0.65 * night) * shimmer;
          for (let layer = -1; layer <= 1; layer++) {
            const t = th + layer * wid * 0.45;
            const st = Math.sin(t), ct = Math.cos(t) * hemi;
            const v = [ct * D[0] + st * (Math.cos(phi) * X[0] + Math.sin(phi) * Y[0]),
              ct * D[1] + st * (Math.cos(phi) * X[1] + Math.sin(phi) * Y[1]),
              ct * D[2] + st * (Math.cos(phi) * X[2] + Math.sin(phi) * Y[2])];
            project(v, rr);
            if (pj.d < 0) {                          // behind: only past the limb
              const dx = pj.x - G.ex, dy = pj.y - G.ey;
              if (dx * dx + dy * dy < G.R * G.R * 1.0) continue;
            }
            const aa = a * (layer === 0 ? 1 : 0.5) * (pj.d < 0 ? 0.45 : 1);
            const red = e.kp >= 6.5 ? clamp((e.kp - 6) / 3, 0, 1) : 0;
            const rgb = layer === -1 && red > 0 ? `255, ${Math.round(120 + 90 * (1 - red))}, 140` : '96, 255, 170';
            ctx.fillStyle = `rgba(${rgb}, ${0.13 * aa})`;
            ctx.beginPath(); ctx.arc(pj.x, pj.y, Math.max(1, G.R * 0.032), 0, TWO_PI); ctx.fill();
            if (layer === 0 && (k & 1) === 0) {
              gctx.fillStyle = `rgba(70, 255, 150, ${0.2 * aa})`;
              gctx.beginPath(); gctx.arc(pj.x * gs, pj.y * gs, G.R * 0.07 * gs, 0, TWO_PI); gctx.fill();
            }
          }
        }
      }
      ctx.restore();
    }

    /* GEO belt: a dotted ring; the part outside the shield turns warm */
    function drawGeo(front) {
      const rd = disp(GEO_RE);
      const n = 180;
      ctx.save();
      for (let k = 0; k < n; k++) {
        const a = (k / n) * TWO_PI;
        const v = [Math.cos(a), Math.sin(a), 0];
        project(v, rd);
        if ((pj.d >= 0) !== front) continue;
        const X = pj.x - G.ex, Y = pj.y - G.ey;
        const out = !inBody(X, Y, rShield);
        const behindEarth = pj.d < 0 && X * X + Y * Y < G.R * G.R;
        if (behindEarth) continue;
        ctx.fillStyle = out ? 'rgba(255, 130, 96, 0.85)' : `rgba(170, 190, 220, ${front ? 0.42 : 0.22})`;
        ctx.fillRect(pj.x - 0.8, pj.y - 0.8, out ? 2.2 : 1.6, out ? 2.2 : 1.6);
        if (out && k % 3 === 0) {
          gctx.fillStyle = 'rgba(255, 120, 80, 0.5)';
          gctx.fillRect(pj.x * gs - 1, pj.y * gs - 1, 2, 2);
        }
      }
      ctx.restore();
    }
    function geoExposed() {
      const rd = disp(GEO_RE);
      return rd > rShield * 0.995;
    }

    /* a LEO satellite (altitude exaggerated) with a fading trail */
    function satPos(t, out) {
      const per = 16;                                 // seconds per lap on screen
      const u = (t / per) * TWO_PI;
      const c = Math.cos(u), s = Math.sin(u);
      /* orbit plane: inclined 53 deg, node toward the viewer-left */
      const node = 2.3;
      const xo = c, yo = s * Math.cos(SAT_INC), zo = s * Math.sin(SAT_INC);
      out[0] = xo * Math.cos(node) - yo * Math.sin(node);
      out[1] = xo * Math.sin(node) + yo * Math.cos(node);
      out[2] = zo;
      return out;
    }
    const satV = [0, 0, 0];
    function drawSat(front) {
      const rd = G.R * 1.24;
      ctx.save();
      ctx.lineCap = 'round';
      for (let k = 22; k >= 0; k--) {
        satPos(T - k * 0.05, satV);
        project(satV, rd);
        if ((pj.d >= 0) !== front) continue;
        if (pj.d < 0 && (pj.x - G.ex) ** 2 + (pj.y - G.ey) ** 2 < G.R * G.R) continue;
        const a = (1 - k / 23);
        ctx.fillStyle = `rgba(225, 238, 255, ${0.5 * a * a})`;
        ctx.fillRect(pj.x - 0.9, pj.y - 0.9, 1.8, 1.8);
      }
      satPos(T, satV);
      project(satV, rd);
      const vis = (pj.d >= 0) === front && !(pj.d < 0 && (pj.x - G.ex) ** 2 + (pj.y - G.ey) ** 2 < G.R * G.R);
      if (vis) {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(pj.x, pj.y, 2.4, 0, TWO_PI); ctx.fill();
        gctx.fillStyle = 'rgba(220, 235, 255, 0.7)';
        gctx.beginPath(); gctx.arc(pj.x * gs, pj.y * gs, 3 * gs * 2, 0, TWO_PI); gctx.fill();
        satScreen.x = pj.x; satScreen.y = pj.y; satScreen.ok = true;
      }
      ctx.restore();
    }
    const satScreen = { x: 0, y: 0, ok: false };

    /* ======================================================================
       Labels
       ==================================================================== */
    function label(text, ax, ay, tx, ty, opts) {
      const o = Object.assign({}, opts || {});
      ctx.save();
      ctx.font = `${o.weight || 500} ${o.size || 12}px ${font}`;
      /* keep the text inside the visible frame: flip to the anchor's other
         side, then clamp */
      const w = Math.max(ctx.measureText(text).width,
        o.sub ? ctx.measureText(o.sub).width * 0.92 : 0);
      const xMax = W - inset.right - 8, xMin = inset.left + 8;
      if ((o.align || 'left') === 'left' && tx + w > xMax) {
        if (o.leader !== false && ax - (tx - ax) - w >= xMin) { tx = ax - (tx - ax); o.align = 'right'; }
        else tx = Math.max(xMin, xMax - w);
      } else if (o.align === 'right' && tx - w < xMin) {
        if (o.leader !== false && ax + (ax - tx) + w <= xMax) { tx = ax + (ax - tx); o.align = 'left'; }
        else { tx = Math.min(xMax, xMin + w); }
      }
      ty = clamp(ty, inset.top + 12, H - inset.bottom - 22);
      ctx.textBaseline = 'middle';
      ctx.textAlign = o.align || 'left';
      const a = o.alpha == null ? 1 : o.alpha;
      if (o.leader !== false) {
        ctx.strokeStyle = `rgba(220, 230, 245, ${0.32 * a})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx + (o.align === 'right' ? 6 : -6), ty); ctx.stroke();
        ctx.fillStyle = `rgba(230, 238, 250, ${0.85 * a})`;
        ctx.beginPath(); ctx.arc(ax, ay, 1.8, 0, TWO_PI); ctx.fill();
      }
      ctx.fillStyle = o.color || `rgba(232, 238, 248, ${0.86 * a})`;
      ctx.fillText(text, tx, ty);
      if (o.sub) {
        ctx.font = `400 ${(o.size || 12) - 1}px ${font}`;
        ctx.fillStyle = `rgba(200, 212, 230, ${0.6 * a})`;
        ctx.fillText(o.sub, tx, ty + (o.size || 12) + 2);
      }
      ctx.restore();
    }
    function drawLabels() {
      const { ex, ey, R, fx0, fy0, fh, fw } = G;
      const e = eff();
      const small = fw < 620;
      const sz = small ? 11 : Math.round(G.fs);
      /* shield: at the nose, above the axis */
      const phiL = Math.PI - 0.62;
      const rL = bodyR(rShield, phiL);
      const sx = ex + rL * Math.cos(phiL), sy = ey - rL * Math.sin(phiL);
      label('Magnetic shield', sx, sy, sx + 18, Math.max(fy0 + 18, sy - fh * 0.1),
        { size: sz, sub: `${standoff.toFixed(1)} Earth radii from Earth`, weight: 600 });
      /* bow shock: below the axis */
      const phiB = Math.PI - 0.5;
      const rB = bodyR(rShock, phiB);
      const bx = ex + rB * Math.cos(phiB), by = ey + rB * Math.sin(phiB);
      if (!small) label('Bow shock', bx, by, bx - 14, Math.min(G.fy1 - 18, by + fh * 0.12), { size: sz, align: 'right', alpha: 0.85 });
      /* wind: upstream, below the caption card when one covers the corner */
      const wx0 = Math.max(fx0 + 16, G.sunX + fw * 0.06);
      let wyy = fy0 + fh * 0.2;
      if (wx0 < inset.capRight && wyy < inset.capBottom + 24) wyy = inset.capBottom + 34;
      label(`Solar wind · ${Math.round(e.speed)} km/s`, 0, 0, wx0, wyy,
        { size: sz, leader: false, weight: 600, color: 'rgba(255, 220, 180, 0.92)' });
      /* aurora: the north oval (narrow frames: the word only, tucked close) */
      const auY = ey - R * 1.08, auX = ex + R * 0.1;
      label('Aurora', auX, auY, ex + R * (small ? 1.1 : 1.6), ey - R * (small ? 2.9 : 2.15),
        { size: sz, weight: 600, color: 'rgba(140, 255, 196, 0.95)',
          sub: small ? undefined : 'storm energy pouring into the poles' });
      /* upper atmosphere */
      const airX = ex - R * 0.74, airY = ey + R * 0.74;
      label(`Upper atmosphere ×${e.ratio.toFixed(1)}`, airX, airY, ex - R * 1.1, ey + R * 2.1,
        { size: sz, align: 'right', weight: 600,
          color: e.ratio > 1.25 ? 'rgba(255, 176, 140, 0.95)' : 'rgba(186, 208, 255, 0.92)',
          sub: small ? undefined
            : e.ratio > 1.25 ? 'denser air at 550 km than a quiet day' : 'about as dense as a quiet day' });
      /* GEO */
      const gr = disp(GEO_RE);
      project([0.0, 1, 0], gr);
      const exposed = geoExposed();
      if (!small || exposed) {
        label(exposed ? 'GEO belt · outside the shield' : 'GEO belt', ex + gr * 0.95, ey + gr * 0.12,
          ex + gr * 0.95 + 14, ey + gr * 0.12 + 22,
          { size: sz - 1, alpha: exposed ? 1 : 0.7, color: exposed ? 'rgba(255, 150, 120, 0.95)' : undefined });
      }
      /* satellite drag */
      if (satScreen.ok) {
        ctx.save();
        ctx.font = `600 ${sz - 1}px ${font}`;
        ctx.textBaseline = 'middle';
        ctx.fillStyle = e.ratio > 1.25 ? 'rgba(255, 186, 150, 0.95)' : 'rgba(225, 235, 250, 0.85)';
        const dx = satScreen.x - ex, dy = satScreen.y - ey, dl = Math.hypot(dx, dy) || 1;
        const lx = satScreen.x + dx / dl * 14, ly = satScreen.y + dy / dl * 12;
        ctx.textAlign = dx >= 0 ? 'left' : 'right';
        ctx.fillText(`Drag ×${e.ratio.toFixed(1)}`, lx, ly);
        ctx.restore();
      }
    }

    /* ======================================================================
       Frame
       ==================================================================== */
    /* physics for one step: eased conditions, the shields, the particles */
    function advance(dt) {
      if (resize() || !G || layoutDirty) { G = layout(); layoutDirty = false; }
      if (!orient || Date.now() - orient.now > 1000) { orient = orientation(); linesKey = ''; }
      step(dt);
      const e = eff();
      standoff = standoffRe(e.density, e.speed, e.bz);
      rShield = disp(standoff);
      rShock = disp(standoff * SHOCK_X);
      buildLines();
      if (dt > 0) { stepWind(dt); stepPrecip(dt); }
    }

    function draw(dt) {
      advance(dt);
      render();
    }

    function render() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      gctx.setTransform(1, 0, 0, 1, 0, 0);
      gctx.clearRect(0, 0, glow.width, glow.height);

      drawSun();
      drawWind();
      drawShields();
      drawGeo(false);
      drawLines(false);
      drawSat(false);
      drawAir(true);
      if (T - earthAt > 1 || earthAt < 0) renderEarthTexture();
      drawEarth();
      drawAir(false);
      drawAurora();
      drawPrecip();
      drawLines(true);
      drawGeo(true);
      satScreen.ok = false;
      drawSat(true);
      /* bloom: blur at the glow layer's low resolution, then upscale */
      let src = glow;
      if (canBlur) {
        g2ctx.setTransform(1, 0, 0, 1, 0, 0);
        g2ctx.clearRect(0, 0, glow2.width, glow2.height);
        g2ctx.filter = `blur(${Math.max(2, G.R * 0.16 * gs).toFixed(1)}px)`;
        g2ctx.drawImage(glow, 0, 0);
        g2ctx.filter = 'none';
        src = glow2;
      }
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = canBlur ? 0.95 : 0.35;
      ctx.drawImage(src, 0, 0, W, H);
      ctx.restore();
      drawLabels();
    }

    function frame(ts) {
      raf = 0;
      if (!running) return;
      const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0;
      lastTs = ts;
      draw(dt);
      raf = requestAnimationFrame(frame);
    }
    function start() {
      if (running) return;
      running = true; lastTs = 0;
      raf = requestAnimationFrame(frame);
    }
    function stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    }
    /* stills for checks and screenshots (a hidden tab gets no frames) */
    function simulate(seconds) {
      const n = Math.max(1, Math.round(seconds * 60));
      for (let k = 0; k < n; k++) advance(1 / 60);
      render();
    }
    function onVis() { if (document.hidden) { if (running) { stop(); running = 'paused'; } } else if (running === 'paused') { running = false; start(); } }
    document.addEventListener('visibilitychange', onVis);
    if (typeof KEarthTexture !== 'undefined') {
      KEarthTexture.load().then(() => { earthAt = -1e9; }).catch(() => { /* the shaded sphere stands in */ });
    }

    return {
      set, setInsets, start, stop, simulate,
      state() {
        if (!G) advance(0);                      // before the first frame
        return { standoff, geoExposed: G ? geoExposed() : false, ...eff() };
      },
      destroy() { stop(); document.removeEventListener('visibilitychange', onVis); },
    };
  }

  window.KSunView = { mount, standoffRe };
})();
