/* Kessler deck - two living scenes.

   KDeckSwarm.mount(canvas, {reduced, offset, scale}) -> scene
     Every tracked object around Earth, each on its own orbit (time sped up).
     scene.setCatalog(rows)   rows from GET /fleet/positions (TEME state
                              vectors); until then, an illustrative catalog
     scene.layout() / step(dt) / render() / isLive()

   KDeckStation.mount(canvas, {reduced, labels}) -> scene
     15 November 2021: the space station, small, on its orbit around Earth,
     and the debris cloud on a crossing orbit streaming past it each pass.
     scene.load() / layout() / restart() / step(dt) / render(); scene.meet is
     the second the station reaches the cloud

   KDeckCrossing.mount(canvas, {reduced}) -> scene
     10 February 2009: Iridium 33 and Cosmos 2251 on their real orbits over
     the night side, meeting 789 km above Siberia at 16:56 UTC; the two
     debris clouds run on along their orbits.
     scene.load() / layout() / restart() / step(dt) / render()

   Positions are drawn with compressed altitudes (display radius grows with
   the log of altitude), so low orbit reads as the thick cloud it is and
   GEO still fits the frame. */
(function () {
  'use strict';

  const TWO_PI = Math.PI * 2;
  const DEG = Math.PI / 180;
  const MU = 398600.4418, RE = 6378.137;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  /* display radius, in Earth radii, for an orbital radius in km */
  const dispR = (rKm) => 1 + 0.42 * Math.log(1 + Math.max(0, rKm - RE) / 400);

  function seeded(seed) {
    let s = seed >>> 0;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  }

  /* ======================================================================
     The catalog around Earth
     ==================================================================== */
  function mountSwarm(canvas, opts) {
    const o = opts || {};
    const reduce = !!o.reduced;
    const ctx = canvas.getContext('2d');
    const glow = document.createElement('canvas');
    const gctx = glow.getContext('2d');
    const glow2 = document.createElement('canvas');
    const g2ctx = glow2.getContext('2d');
    const canBlur = typeof gctx.filter === 'string';
    const globe = KDeckGlobe.sphere({ tau: 0.36, omega: -0.16, sun: [-0.62, 0.3, 0.72],
      spin0: -30 * DEG, period: 240, reduced: reduce });
    const { E1, E2, N3 } = globe.basis;
    const ACC = 170;                                   // orbits run 170x faster than life

    let W = 0, H = 0, dpr = 1, gs = 0.5, T = 0, G = null;

    /* one entry per object: its orbit plane (P toward perigee or the epoch
       position, Q 90 degrees along the motion), both in view coordinates */
    let N = 0, live = false;
    let Px, Py, Pz, Qx, Qy, Qz, A, Ecc, Rate, M0, Typ;
    function alloc(n) {
      Px = new Float32Array(n); Py = new Float32Array(n); Pz = new Float32Array(n);
      Qx = new Float32Array(n); Qy = new Float32Array(n); Qz = new Float32Array(n);
      A = new Float32Array(n); Ecc = new Float32Array(n); Rate = new Float32Array(n);
      M0 = new Float32Array(n); Typ = new Uint8Array(n);
    }
    /* inertial (equatorial, z north) -> view */
    function toView(x, y, z, out, k) {
      out[k] = E1[0] * x + E2[0] * y + N3[0] * z;
      out[k + 1] = E1[1] * x + E2[1] * y + N3[1] * z;
      out[k + 2] = E1[2] * x + E2[2] * y + N3[2] * z;
    }
    const tv = new Float32Array(6);
    function putOrbit(i, p, q, a, e, m0, typ) {
      toView(p[0], p[1], p[2], tv, 0); toView(q[0], q[1], q[2], tv, 3);
      Px[i] = tv[0]; Py[i] = tv[1]; Pz[i] = tv[2]; Qx[i] = tv[3]; Qy[i] = tv[4]; Qz[i] = tv[5];
      A[i] = a; Ecc[i] = e; Rate[i] = Math.sqrt(MU / (a * a * a)) * ACC; M0[i] = m0; Typ[i] = typ;
    }
    /* classical elements -> plane vectors (P to perigee, Q along the motion) */
    function fromElements(i, a, e, inc, raan, argp, m0, typ) {
      const cO = Math.cos(raan), sO = Math.sin(raan), ci = Math.cos(inc), si = Math.sin(inc);
      const cw = Math.cos(argp), sw = Math.sin(argp);
      const p = [cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si];
      const q = [-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si];
      putOrbit(i, p, q, a, e, m0, typ);
    }

    /* the real catalog: state vectors -> orbits */
    function setCatalog(rows) {
      if (!Array.isArray(rows) || rows.length < 1000) return false;
      const n = Math.min(rows.length, 40000);
      alloc(n);
      let k = 0;
      for (let j = 0; j < rows.length && k < n; j++) {
        const r = rows[j];
        const x = +r[3], y = +r[4], z = +r[5], vx = +r[6], vy = +r[7], vz = +r[8];
        if (![x, y, z, vx, vy, vz].every(Number.isFinite)) continue;
        const rr = Math.hypot(x, y, z), v2 = vx * vx + vy * vy + vz * vz;
        const a = 1 / (2 / rr - v2 / MU);
        if (!(a > RE) || a > 60000) continue;
        const hx = y * vz - z * vy, hy = z * vx - x * vz, hz = x * vy - y * vx;
        const hl = Math.hypot(hx, hy, hz) || 1;
        const rv = x * vx + y * vy + z * vz;
        const c1 = v2 - MU / rr;
        const ex = (c1 * x - rv * vx) / MU, ey = (c1 * y - rv * vy) / MU, ez = (c1 * z - rv * vz) / MU;
        const e = Math.hypot(ex, ey, ez);
        if (e >= 0.97) continue;
        let p, m0;
        if (e < 0.03) {                                // near-circular: P at the epoch position
          p = [x / rr, y / rr, z / rr]; m0 = 0;
        } else {                                       // P at perigee, the mean anomaly from the true one
          p = [ex / e, ey / e, ez / e];
          const cnu = clamp((p[0] * x + p[1] * y + p[2] * z) / rr, -1, 1);
          let nu = Math.acos(cnu);
          if (rv < 0) nu = TWO_PI - nu;
          const E = 2 * Math.atan(Math.sqrt((1 - e) / (1 + e)) * Math.tan(nu / 2));
          m0 = E - e * Math.sin(E);
        }
        const hxu = hx / hl, hyu = hy / hl, hzu = hz / hl;
        const q = [hyu * p[2] - hzu * p[1], hzu * p[0] - hxu * p[2], hxu * p[1] - hyu * p[0]];
        const t = r[2] | 0;
        putOrbit(k++, p, q, a, e < 0.03 ? 0 : e, m0, t === 0 ? 0 : t === 2 ? 2 : 1);
      }
      N = k; live = true;
      return true;
    }

    /* until the real catalog arrives (or offline): an illustrative one with
       the same make-up (shells, debris bands, MEO navigation, GEO, transfer) */
    function synthCatalog() {
      const rnd = seeded(33166);
      const gauss = () => (rnd() + rnd() + rnd() - 1.5) * 0.8;
      const groups = [
        // count, alt km [lo, hi], inclinations (deg), type (0 sat, 1 debris, 2 rocket body), ecc
        [7200, [530, 570], [53, 53.2, 43, 70, 97.6], 0],
        [700, [1180, 1220], [87.9], 0],
        [3400, [400, 900], [51.6, 97.5, 98.2, 65, 82, 45, 28], 0],
        [9800, [600, 1100], [98.8, 74, 82, 65, 86.4, 71], 1],
        [2400, [600, 1600], [71, 82, 98, 51.6, 63], 2],
        [1400, [300, 2000], [10, 30, 60, 90, 100], 1],
      ];
      const meo = [[31, 20200, 55], [24, 19100, 64.8], [28, 23222, 56], [24, 21528, 55], [160, 20000, 60]];
      const geo = 1600, gto = 650, molniya = 60;
      let n = groups.reduce((s, g) => s + g[0], 0) + meo.reduce((s, g) => s + g[0], 0) + geo + gto + molniya;
      alloc(n);
      let i = 0;
      for (const [count, [lo, hi], incs, typ] of groups) {
        for (let k = 0; k < count; k++) {
          const alt = lerp(lo, hi, rnd());
          const inc = (incs[Math.floor(rnd() * incs.length)] + gauss() * 1.2) * DEG;
          fromElements(i++, RE + alt, 0, inc, rnd() * TWO_PI, 0, rnd() * TWO_PI, typ);
        }
      }
      for (const [count, alt, inc] of meo) {
        for (let k = 0; k < count; k++) {
          fromElements(i++, RE + alt + gauss() * 300, 0, (inc + gauss()) * DEG,
            (Math.floor(rnd() * 6) / 6) * TWO_PI + gauss() * 0.05, 0, rnd() * TWO_PI, count > 100 ? 1 : 0);
        }
      }
      for (let k = 0; k < geo; k++) {
        const dead = rnd() < 0.55;
        fromElements(i++, 42164 + gauss() * (dead ? 300 : 20), 0, (dead ? rnd() * 15 : rnd() * 0.3) * DEG,
          rnd() * TWO_PI, 0, rnd() * TWO_PI, dead ? 1 : 0);
      }
      for (let k = 0; k < gto; k++) {
        const rp = RE + 250 + rnd() * 350, ra = 42164 + gauss() * 400;
        fromElements(i++, (rp + ra) / 2, (ra - rp) / (ra + rp), (7 + rnd() * 21) * DEG,
          rnd() * TWO_PI, rnd() * TWO_PI, rnd() * TWO_PI, rnd() < 0.7 ? 2 : 1);
      }
      for (let k = 0; k < molniya; k++) {
        const rp = RE + 600, ra = RE + 39700;
        fromElements(i++, (rp + ra) / 2, (ra - rp) / (ra + rp), 63.4 * DEG,
          rnd() * TWO_PI, 270 * DEG, rnd() * TWO_PI, 0);
      }
      N = i; live = false;
    }
    synthCatalog();

    /* ------------------------------------------------------------- layout */
    function layout() {
      const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
      const d = Math.min(2, window.devicePixelRatio || 1, Math.sqrt(8.5e6 / (w * h)));
      if (w === W && h === H && Math.abs(d - dpr) < 1e-3 && G) return false;
      W = w; H = h; dpr = d;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      glow.width = glow2.width = Math.max(1, Math.round(W * gs));
      glow.height = glow2.height = Math.max(1, Math.round(H * gs));
      const S = Math.min(W, H * 16 / 9);
      const R = clamp(Math.min(H * 0.112, S * 0.066) * (o.scale || 1), 30, 220);
      G = { S, R, ex: W / 2 + S * (o.offset == null ? 0.2 : o.offset), ey: H * 0.5 };
      globe.size(R, dpr);
      return true;
    }

    /* ---------------------------------------------------------- positions */
    let SX = new Float32Array(0), SY = new Float32Array(0), SZ = new Float32Array(0);
    function positions() {
      if (SX.length < N) { SX = new Float32Array(N); SY = new Float32Array(N); SZ = new Float32Array(N); }
      const { ex, ey, R } = G;
      for (let i = 0; i < N; i++) {
        const M = M0[i] + Rate[i] * T;
        let c, s, rd;
        const e = Ecc[i];
        if (e === 0) {
          c = Math.cos(M); s = Math.sin(M); rd = dispR(A[i]);
        } else {                                       // three Newton steps on Kepler's equation
          let E = M + e * Math.sin(M);
          for (let k = 0; k < 3; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
          const cE = Math.cos(E), sE = Math.sin(E);
          const xo = cE - e, yo = Math.sqrt(1 - e * e) * sE;
          const ro = Math.hypot(xo, yo) || 1;
          c = xo / ro; s = yo / ro; rd = dispR(A[i] * (1 - e * cE));
        }
        const vx = c * Px[i] + s * Qx[i], vy = c * Py[i] + s * Qy[i], vz = c * Pz[i] + s * Qz[i];
        SX[i] = ex + vx * rd * R; SY[i] = ey - vy * rd * R; SZ[i] = vz;
      }
    }

    /* -------------------------------------------------------------- draw */
    const COLORS = [
      ['rgba(150, 196, 255, 0.95)', 'rgba(150, 196, 255, 0.35)'],     // satellites
      ['rgba(214, 222, 232, 0.8)', 'rgba(214, 222, 232, 0.28)'],      // debris
      ['rgba(236, 240, 246, 0.9)', 'rgba(236, 240, 246, 0.3)'],       // rocket bodies
    ];
    function drawDots(front) {
      const { ex, ey, R } = G;
      const R2 = R * R;
      const sz = front ? 1.4 : 1.1, h = sz / 2;
      for (let pass = 0; pass < (front ? 2 : 1); pass++) {   // front: beside Earth, then over its face
        for (let t = 0; t < 3; t++) {
          ctx.fillStyle = COLORS[t][front ? 0 : 1];
          ctx.globalAlpha = pass ? 0.42 : 1;
          ctx.beginPath();
          for (let i = 0; i < N; i++) {
            if (Typ[i] !== t) continue;
            const z = SZ[i];
            if ((z >= 0) !== front) continue;
            const x = SX[i], y = SY[i];
            const dx = x - ex, dy = y - ey, inDisc = dx * dx + dy * dy < R2;
            if (!front ? inDisc : inDisc !== !!pass) continue;
            ctx.rect(x - h, y - h, sz, sz);
          }
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }
    function drawAir() {
      const { ex, ey, R } = G;
      /* a rim, starting transparent: inside a radial gradient's inner circle
         the first stop fills everything, which washed the whole disc */
      const g = ctx.createRadialGradient(ex, ey, R * 0.95, ex, ey, R * 1.08);
      g.addColorStop(0, 'rgba(120, 178, 255, 0)');
      g.addColorStop(0.35, 'rgba(120, 178, 255, 0.3)');
      g.addColorStop(0.55, 'rgba(120, 178, 255, 0.1)');
      g.addColorStop(1, 'rgba(120, 178, 255, 0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(ex, ey, R * 1.09, 0, TWO_PI); ctx.fill();
      gctx.strokeStyle = 'rgba(110, 170, 255, 0.35)';
      gctx.lineWidth = Math.max(2, R * 0.08 * gs * 2);
      gctx.beginPath(); gctx.arc(ex * gs, ey * gs, R * 1.02 * gs, 0, TWO_PI); gctx.stroke();
    }

    function step(dt) {
      if (!G) layout();
      T += dt;
    }
    function render() {
      if (!G) layout();
      positions();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      gctx.setTransform(1, 0, 0, 1, 0, 0);
      gctx.clearRect(0, 0, glow.width, glow.height);
      drawDots(false);
      globe.draw(ctx, G.ex, G.ey, T);
      drawAir();
      drawDots(true);
      let src = glow;
      if (canBlur) {
        g2ctx.setTransform(1, 0, 0, 1, 0, 0);
        g2ctx.clearRect(0, 0, glow2.width, glow2.height);
        g2ctx.filter = `blur(${Math.max(2, G.R * 0.1 * gs).toFixed(1)}px)`;
        g2ctx.drawImage(glow, 0, 0);
        g2ctx.filter = 'none';
        src = glow2;
      }
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = canBlur ? 0.9 : 0.35;
      ctx.drawImage(src, 0, 0, W, H);
      ctx.restore();
    }

    return {
      layout, step, render, setCatalog,
      load: () => globe.load(),
      isLive: () => live,
      count: () => N,
      simulate(seconds) {
        const n = Math.max(1, Math.round(seconds * 30));
        for (let k = 0; k < n; k++) step(1 / 30);
        render();
      },
    };
  }

  /* ======================================================================
     The International Space Station, as it was on 15 November 2021
     ==================================================================== */
  /* x starboard, y forward (the direction of flight), z up (away from Earth);
     metres, from NASA's reference figures: a 109 m truss, eight 35 m solar
     wings of two blankets each, the S1/P1 radiators, the US and Russian
     modules (Nauka since July 2021), Crew Dragon Endurance on Harmony's
     forward port and Soyuz MS-19 on Rassvet, below Zarya. */
  function issModel() {
    const P = [];
    const cyl = (a, b, r, mat, r2, tag) => P.push({ k: 'cyl', a, b, r, r2: r2 == null ? r : r2, mat, tag });
    const quad = (pts, mat, grid, tag) => P.push({ k: 'quad', pts, mat, grid, tag });
    /* the truss, in segments so it sorts against what sits on it */
    for (let k = 0; k < 16; k++) {
      const x0 = -54.5 + k * 109 / 16;
      P.push({ k: 'box', min: [x0, -2.3, 2.2], max: [x0 + 109 / 16, 2.3, 6.2], mat: 'truss' });
    }
    /* solar wings on P6, P4, S4, S6: forward and aft, two blankets either side
       of a mast, tilted on their gimbals toward the Sun */
    const beta = 28 * DEG;
    const wd = [Math.cos(beta), 0, Math.sin(beta)];
    for (const xm of [-50.4, -37.6, 37.6, 50.4]) {
      for (const sy of [1, -1]) {
        const y0 = 3.4 * sy, y1 = 36.9 * sy;
        for (const [o0, o1] of [[-5.8, -1.1], [1.1, 5.8]]) {
          const c = (off, y) => [xm + wd[0] * off, y, 4.2 + wd[2] * off];
          quad([c(o0, y0), c(o1, y0), c(o1, y1), c(o0, y1)], 'array', [17, 1]);
        }
        P.push({ k: 'line', a: [xm, y0 * 0.6, 4.2], b: [xm, y1, 4.2], mat: 'mast' });
      }
    }
    /* S1 and P1 radiators: three wings a side, angled aft and down */
    const rd = [0, -Math.cos(30 * DEG), -Math.sin(30 * DEG)];
    for (const sx of [1, -1]) {
      for (let k = 0; k < 3; k++) {
        const x0 = sx * (10.8 + k * 3.9), x1 = x0 + sx * 2.9;
        const L = 22.5;
        quad([[x0, -2.3, 3.0], [x1, -2.3, 3.0], [x1, -2.3 + rd[1] * L, 3.0 + rd[2] * L], [x0, -2.3 + rd[1] * L, 3.0 + rd[2] * L]],
          'radiator', [8, 0]);
      }
    }
    /* US segment */
    cyl([0, 4.3, 0], [0, 11.5, 0], 2.2, 'module');                 // Harmony
    cyl([0, -4.3, 0], [0, 4.3, 0], 2.15, 'module');                // destiny
    cyl([0, -9.8, 0], [0, -4.3, 0], 2.3, 'module');                // Unity
    cyl([2.2, 7.9, 0], [9.1, 7.9, 0], 2.25, 'module');             // Columbus
    cyl([-2.2, 7.9, 0], [-13.4, 7.9, 0], 2.2, 'module');           // Kibo
    P.push({ k: 'box', min: [-19.6, 5.3, -0.9], max: [-13.4, 10.5, 0.7], mat: 'platform' });  // Kibo's exposed facility
    cyl([-6.2, 7.9, 2.2], [-6.2, 7.9, 6.4], 2.2, 'module');        // Kibo's logistics module
    cyl([-2.3, -7.05, 0], [-9.0, -7.05, 0], 2.25, 'module');       // tranquility
    cyl([2.3, -7.05, 0], [7.8, -7.05, 0], 2.0, 'module');          // Quest
    cyl([0, -7.05, -2.3], [0, -7.05, -8.7], 2.2, 'module');        // Leonardo
    cyl([-5.6, -7.05, -2.25], [-5.6, -7.05, -3.7], 1.48, 'module', 1.2); // the Cupola
    /* Russian segment */
    cyl([0, -11.8, 0], [0, -9.8, 0], 1.4, 'module', 1.2);          // PMA-1
    cyl([0, -24.4, 0], [0, -11.8, 0], 2.05, 'ru');                 // Zarya
    cyl([0, -37.5, 0], [0, -24.4, 0], 2.07, 'ru', 1.6);            // Zvezda
    cyl([0, -27.0, 2.0], [0, -27.0, 6.1], 1.27, 'ru');             // Poisk
    cyl([0, -27.0, -2.0], [0, -27.0, -15.0], 2.1, 'ru');           // Nauka
    cyl([0, -18.0, -2.0], [0, -18.0, -8.0], 1.17, 'ru');           // Rassvet
    for (const sx of [1, -1]) {                                    // Zvezda's wings
      const c = (x, y) => [x, y, 0.8];
      quad([c(sx * 2.4, -33.1), c(sx * 13.1, -33.1), c(sx * 13.1, -36.4), c(sx * 2.4, -36.4)], 'array', [0, 5]);
    }
    /* who was docked that morning */
    cyl([0, 11.5, 0], [0, 13.5, 0], 1.0, 'module');                // PMA-2
    cyl([0, 13.5, 0], [0, 17.4, 0], 0.95, 'dragon', 1.95, 'dragon'); // crew Dragon, nose docked
    cyl([0, 17.4, 0], [0, 20.2, 0], 1.85, 'trunk');                // its trunk
    cyl([0, -18.0, -8.0], [0, -18.0, -10.8], 1.13, 'soyuz', 1.13, 'soyuz');  // Soyuz orbital module
    cyl([0, -18.0, -10.8], [0, -18.0, -13.0], 1.0, 'soyuz', 1.5);  // descent module
    cyl([0, -18.0, -13.0], [0, -18.0, -15.9], 1.36, 'soyuzsm');    // service module
    for (const sx of [1, -1]) {
      quad([[sx * 1.4, -18.7, -14.6], [sx * 5.4, -18.7, -14.6], [sx * 5.4, -17.3, -14.6], [sx * 1.4, -17.3, -14.6]], 'array', [0, 3]);
    }
    return P;
  }

  /* base colours (albedo); light does the rest */
  const MATS = {
    module: [228, 232, 236], ru: [212, 217, 208], dragon: [244, 245, 246], trunk: [58, 66, 78],
    soyuz: [178, 188, 152], soyuzsm: [132, 142, 124],
  };
  const norm3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  function issRenderer() {
    const parts = issModel();
    function render(c, cx, cy, s, cam, L) {
      const light = norm3(L);
      const half = norm3([light[0] + cam.back[0], light[1] + cam.back[1], light[2] + cam.back[2]]);
      const S = (v) => { const q = cam.p(v); return [cx + q[0] * s, cy - q[1] * s, q[2], q[3]]; };
      const items = [];
      for (const pt of parts) {
        let depth;
        if (pt.k === 'cyl' || pt.k === 'line') depth = (cam.p(pt.a)[2] + cam.p(pt.b)[2]) / 2;
        else if (pt.k === 'quad') depth = pt.pts.reduce((sum, q) => sum + cam.p(q)[2], 0) / 4;
        else depth = cam.p([(pt.min[0] + pt.max[0]) / 2, (pt.min[1] + pt.max[1]) / 2, (pt.min[2] + pt.max[2]) / 2])[2];
        items.push([depth, pt]);
      }
      items.sort((a, b) => a[0] - b[0]);
      const anchors = {};
      c.save();
      c.lineJoin = 'round';
      for (const [, pt] of items) {
        if (pt.k === 'cyl') drawCyl(c, pt, S, s, cam, light, half, anchors);
        else if (pt.k === 'quad') drawQuad(c, pt, S, cam, light, half);
        else if (pt.k === 'box') drawBox(c, pt, S, cam, light);
        else if (pt.k === 'line') {
          const a = S(pt.a), b = S(pt.b);
          c.strokeStyle = 'rgba(190, 198, 208, 0.8)'; c.lineWidth = Math.max(0.7, s * 0.1 * a[3]);
          c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
        }
      }
      c.restore();
      return anchors;
    }
    const shadeRGB = (rgb, k) => `rgb(${rgb.map((v) => Math.round(clamp(v * k, 0, 255))).join(',')})`;
    /* a lit cylinder (or cone): across its width the surface normal turns from
       one silhouette edge to the other; each stop is that normal's Lambert and
       Blinn-Phong light */
    function drawCyl(c, pt, S, s, cam, light, half, anchors) {
      const rgb = MATS[pt.mat] || MATS.module;
      const a = S(pt.a), b = S(pt.b);
      const r0 = pt.r * s * a[3], r1 = pt.r2 * s * b[3];
      const ax = norm3([pt.b[0] - pt.a[0], pt.b[1] - pt.a[1], pt.b[2] - pt.a[2]]);
      const f = cam.facing(ax);
      let dx = b[0] - a[0], dy = b[1] - a[1];
      const len = Math.hypot(dx, dy);
      const shine = pt.mat === 'trunk' ? 0.12 : pt.mat === 'soyuz' || pt.mat === 'soyuzsm' ? 0.18 : 0.4;
      const lit = (n) => 0.44 + 0.7 * Math.max(0, dot3(n, light));
      const spec = (n) => shine * Math.pow(Math.max(0, dot3(n, half)), 28);
      if (len > 0.5) {
        dx /= len; dy /= len;
        /* the side vector and the camera-facing normal, in 3-D */
        const side = norm3(cross3(ax, cam.back));
        const fwd = norm3([cam.back[0] - f * ax[0], cam.back[1] - f * ax[1], cam.back[2] - f * ax[2]]);
        const sx = dot3(side, cam.right), sy = -dot3(side, cam.up);
        const sl = Math.hypot(sx, sy) || 1;
        const ux = sx / sl, uy = sy / sl;
        const g = c.createLinearGradient(a[0] - ux * r0, a[1] - uy * r0, a[0] + ux * r0, a[1] + uy * r0);
        for (let k = 0; k <= 8; k++) {
          const w = -1 + k / 4, h = Math.sqrt(Math.max(0, 1 - w * w));
          const n = [side[0] * w + fwd[0] * h, side[1] * w + fwd[1] * h, side[2] * w + fwd[2] * h];
          const sp = spec(n) * 255;
          const col = rgb.map((v) => Math.round(clamp(v * lit(n) + sp, 0, 255)));
          g.addColorStop(k / 8, `rgb(${col.join(',')})`);
        }
        c.fillStyle = g;
        c.beginPath();
        c.moveTo(a[0] - ux * r0, a[1] - uy * r0); c.lineTo(b[0] - ux * r1, b[1] - uy * r1);
        c.lineTo(b[0] + ux * r1, b[1] + uy * r1); c.lineTo(a[0] + ux * r0, a[1] + uy * r0);
        c.closePath(); c.fill();
        c.strokeStyle = 'rgba(8, 12, 18, 0.35)'; c.lineWidth = 0.6; c.stroke();
        /* seams around the hull */
        if (len > s * 3 && pt.mat !== 'trunk') {
          c.strokeStyle = 'rgba(30, 38, 50, 0.22)'; c.lineWidth = Math.max(0.5, s * 0.05);
          for (const t of [0.34, 0.67]) {
            const px = a[0] + (b[0] - a[0]) * t, py = a[1] + (b[1] - a[1]) * t, rr = r0 + (r1 - r0) * t;
            c.beginPath(); c.moveTo(px - ux * rr, py - uy * rr); c.lineTo(px + ux * rr, py + uy * rr); c.stroke();
          }
        }
      }
      /* the end that faces the camera, lit as a flat disc */
      const e = f >= 0 ? b : a, er = f >= 0 ? r1 : r0;
      const nE = f >= 0 ? ax : [-ax[0], -ax[1], -ax[2]];
      const minor = Math.max(0.12, Math.abs(f)) * er;
      const ang = len > 0.5 ? Math.atan2(dy, dx) : 0;
      c.fillStyle = shadeRGB(rgb, 0.3 + 0.72 * Math.max(0, dot3(nE, light)));
      c.beginPath(); c.ellipse(e[0], e[1], minor, er, ang, 0, TWO_PI); c.fill();
      c.strokeStyle = 'rgba(8, 12, 18, 0.35)'; c.lineWidth = 0.6; c.stroke();
      if (pt.tag) anchors[pt.tag] = { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, r: Math.max(r0, r1) };
    }
    function drawQuad(c, pt, S, cam, light, half) {
      const q = pt.pts.map(S);
      const e1 = [pt.pts[1][0] - pt.pts[0][0], pt.pts[1][1] - pt.pts[0][1], pt.pts[1][2] - pt.pts[0][2]];
      const e2 = [pt.pts[3][0] - pt.pts[0][0], pt.pts[3][1] - pt.pts[0][1], pt.pts[3][2] - pt.pts[0][2]];
      let n = norm3(cross3(e1, e2));
      if (dot3(n, cam.back) < 0) n = [-n[0], -n[1], -n[2]];
      const dif = Math.abs(dot3(n, light));
      if (pt.mat === 'array') {
        /* cells: deep blue; a sheen sweeps where the Sun would glint */
        const k = 0.55 + 0.6 * dif;
        const glint = Math.pow(Math.max(0, dot3(n, half)), 6);
        const g = c.createLinearGradient(q[0][0], q[0][1], q[2][0], q[2][1]);
        g.addColorStop(0, shadeRGB([24, 46, 82], k));
        g.addColorStop(0.5, shadeRGB([42 + 70 * glint, 74 + 70 * glint, 122 + 60 * glint], k));
        g.addColorStop(1, shadeRGB([20, 38, 68], k));
        c.fillStyle = g;
      } else {
        const g = c.createLinearGradient(q[0][0], q[0][1], q[2][0], q[2][1]);
        g.addColorStop(0, shadeRGB([236, 239, 242], 0.42 + 0.62 * dif));
        g.addColorStop(1, shadeRGB([200, 207, 214], 0.36 + 0.6 * dif));
        c.fillStyle = g;
      }
      c.beginPath(); c.moveTo(q[0][0], q[0][1]);
      for (let k = 1; k < 4; k++) c.lineTo(q[k][0], q[k][1]);
      c.closePath(); c.fill();
      const [along, across] = pt.grid || [0, 0];
      c.lineWidth = 0.6;
      c.strokeStyle = pt.mat === 'array' ? 'rgba(130, 160, 204, 0.34)' : 'rgba(110, 122, 136, 0.45)';
      c.beginPath();
      for (let k = 1; k < along; k++) {
        const t = k / along;
        c.moveTo(q[0][0] + (q[3][0] - q[0][0]) * t, q[0][1] + (q[3][1] - q[0][1]) * t);
        c.lineTo(q[1][0] + (q[2][0] - q[1][0]) * t, q[1][1] + (q[2][1] - q[1][1]) * t);
      }
      for (let k = 1; k < across; k++) {
        const t = k / across;
        c.moveTo(q[0][0] + (q[1][0] - q[0][0]) * t, q[0][1] + (q[1][1] - q[0][1]) * t);
        c.lineTo(q[3][0] + (q[2][0] - q[3][0]) * t, q[3][1] + (q[2][1] - q[3][1]) * t);
      }
      c.stroke();
      c.strokeStyle = pt.mat === 'array' ? 'rgba(196, 158, 86, 0.7)' : 'rgba(255, 255, 255, 0.45)';
      c.lineWidth = 0.8;
      c.beginPath(); c.moveTo(q[0][0], q[0][1]);
      for (let k = 1; k < 4; k++) c.lineTo(q[k][0], q[k][1]);
      c.closePath(); c.stroke();
    }
    function drawBox(c, pt, S, cam, light) {
      const [x0, y0, z0] = pt.min, [x1, y1, z1] = pt.max;
      const faces = [
        { n: [0, 0, 1], v: [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]] },
        { n: [0, -1, 0], v: [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]] },
        { n: [0, 1, 0], v: [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]] },
        { n: [-1, 0, 0], v: [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]] },
        { n: [1, 0, 0], v: [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]] },
        { n: [0, 0, -1], v: [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]] },
      ];
      const truss = pt.mat === 'truss';
      for (const fc of faces) {
        if (cam.facing(fc.n) <= 0.02) continue;
        const q = fc.v.map(S);
        const k = 0.34 + 0.7 * Math.max(0, dot3(fc.n, light));
        c.fillStyle = truss ? `rgba(${[132, 142, 154].map((v) => Math.round(v * k)).join(',')}, 0.8)` : shadeRGB([168, 176, 186], k);
        c.beginPath(); c.moveTo(q[0][0], q[0][1]);
        for (let m = 1; m < 4; m++) c.lineTo(q[m][0], q[m][1]);
        c.closePath(); c.fill();
        if (truss) {                                  // the lattice
          c.strokeStyle = `rgba(${Math.round(200 * k + 30)}, ${Math.round(208 * k + 30)}, ${Math.round(218 * k + 30)}, 0.8)`;
          c.lineWidth = 0.75;
          c.beginPath();
          const n = 3;
          for (let m = 0; m <= n; m++) {
            const t = m / n;
            const ax0 = q[0][0] + (q[1][0] - q[0][0]) * t, ay0 = q[0][1] + (q[1][1] - q[0][1]) * t;
            const ax1 = q[3][0] + (q[2][0] - q[3][0]) * t, ay1 = q[3][1] + (q[2][1] - q[3][1]) * t;
            c.moveTo(ax0, ay0); c.lineTo(ax1, ay1);
            if (m < n) {
              const t2 = (m + 1) / n;
              c.moveTo(ax0, ay0);
              c.lineTo(q[3][0] + (q[2][0] - q[3][0]) * t2, q[3][1] + (q[2][1] - q[3][1]) * t2);
            }
          }
          c.stroke();
        }
        c.strokeStyle = 'rgba(10, 14, 20, 0.45)'; c.lineWidth = 0.6;
        c.beginPath(); c.moveTo(q[0][0], q[0][1]);
        for (let m = 1; m < 4; m++) c.lineTo(q[m][0], q[m][1]);
        c.closePath(); c.stroke();
      }
    }
    return { render };
  }

  /* ======================================================================
     15 November 2021: the space station on its orbit, and the debris
     ==================================================================== */
  /* The station (51.6 degrees, drawn small, oriented along its orbit) and the
     Kosmos 1408 debris cloud on its own orbit (82.6 degrees, a little higher).
     The two orbits cross twice a lap, so on each pass the fragments stream by,
     which is why the crew sheltered. Altitudes are exaggerated so the orbits
     clear the globe; a lap takes 27 seconds on screen. */
  function mountStation(canvas, opts) {
    const o = opts || {};
    const reduce = !!o.reduced;
    const ctx = canvas.getContext('2d');
    const globe = KDeckGlobe.sphere({ tau: 0.4, omega: 0.12, sun: [-0.72, 0.32, 0.62],
      spin0: -70 * DEG, period: 240, reduced: reduce, hires: true });
    const { E1, E2, N3 } = globe.basis;
    const SUNV = globe.sun;
    const iss = issRenderer();
    const ISS = { inc: 51.64 * DEG, r: 1.17, per: 27 };
    const DEB = { inc: 82.6 * DEG, r: 1.21, per: 27.8 };
    let W = 0, H = 0, dpr = 1, T = 0, G = null;

    /* orbit planes: pick the shared node offset that puts one crossing high on
       the left, facing us, where the station meets the cloud in plain view */
    function planeVecs(inc, raan) {
      const cO = Math.cos(raan), sO = Math.sin(raan), ci = Math.cos(inc), si = Math.sin(inc);
      const toV = (x, y, z) => [E1[0] * x + E2[0] * y + N3[0] * z, E1[1] * x + E2[1] * y + N3[1] * z, E1[2] * x + E2[2] * y + N3[2] * z];
      return { P: toV(cO, sO, 0), Q: toV(-sO * ci, cO * ci, si), nrm: toV(sO * si, -cO * si, ci) };
    }
    /* and both rings open toward us (not edge-on), so the cloud reads as a band */
    let best = 0, bestD = 0, bestScore = -9;
    for (let dr = 40; dr <= 140; dr += 4) {
      for (let k = 0; k < 360; k += 3) {
        const a = planeVecs(ISS.inc, k * DEG), b = planeVecs(DEB.inc, (k + dr) * DEG);
        const d = norm3(cross3(a.nrm, b.nrm));
        for (const sg of [1, -1]) {
          const v = [d[0] * sg, d[1] * sg, d[2] * sg];
          const sc = -Math.hypot(v[0] + 0.5, v[1] - 0.42) + 0.5 * v[2]
            + 0.7 * Math.min(Math.abs(a.nrm[2]), 0.6) + 0.9 * Math.min(Math.abs(b.nrm[2]), 0.55);
          if (sc > bestScore) { bestScore = sc; best = k; bestD = dr; }
        }
      }
    }
    ISS.pl = planeVecs(ISS.inc, best * DEG);
    DEB.pl = planeVecs(DEB.inc, (best + bestD) * DEG);
    /* the crossing, high on the left and facing us, as an angle on each ring */
    const XV = (() => {
      const d = norm3(cross3(ISS.pl.nrm, DEB.pl.nrm));
      return Math.hypot(d[0] + 0.5, d[1] - 0.42) < Math.hypot(-d[0] + 0.5, -d[1] - 0.42) ? d : d.map((c) => -c);
    })();
    const X0 = Math.atan2(dot3(XV, ISS.pl.Q), dot3(XV, ISS.pl.P));
    const XD = Math.atan2(dot3(XV, DEB.pl.Q), dot3(XV, DEB.pl.P));
    const wI = TWO_PI / ISS.per, wD = TWO_PI / DEB.per;
    /* T = 0: the station just come round from behind the Earth, so a whole
       pass in front of it plays out while the slide is up */
    const zOn = (pl, u) => Math.cos(u) * pl.P[2] + Math.sin(u) * pl.Q[2];
    let U0 = X0;
    for (let k = 0; k < 720 && zOn(ISS.pl, U0) >= 0; k++) U0 -= TWO_PI / 720;
    U0 += 0.3;
    const MEET = (((X0 - U0) % TWO_PI) + TWO_PI) % TWO_PI / wI;   // seconds until it meets the cloud

    /* the cloud: fragments spread around their orbit, thickest where they
       cross the station's path just as it gets there */
    const rnd = seeded(1408);
    const NF = 360;
    const F = [];
    const B0 = XD - wD * MEET;
    for (let k = 0; k < NF; k++) {
      const g = (rnd() + rnd() + rnd() - 1.5) / 1.5;
      F.push({ u: rnd() < 0.55 ? B0 + g * 2.4 : rnd() * TWO_PI, dr: (rnd() - 0.5) * 0.06,
        tilt: (rnd() - 0.5) * 7 * DEG, w: 1 + (rnd() - 0.5) * 0.05, a: 0.36 + 0.46 * rnd() });
    }

    function layout() {
      const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
      const d = Math.min(2, window.devicePixelRatio || 1, Math.sqrt(8.5e6 / (w * h)));
      if (w === W && h === H && Math.abs(d - dpr) < 1e-3 && G) return false;
      W = w; H = h; dpr = d;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      const S = Math.min(W, H * 16 / 9);
      const R = clamp(Math.min(H * 0.2, S * 0.115), 40, 260);
      G = { S, R, ex: W / 2, ey: H * (o.cy || 0.66) };
      globe.size(R, dpr);
      return true;
    }
    const at = (pl, u, r, tilt, out) => {
      const c = Math.cos(u), s2 = Math.sin(u);
      let Q = pl.Q;
      if (tilt) {                                      // a fragment's plane, tipped about P
        const ct = Math.cos(tilt), st = Math.sin(tilt);
        Q = [Q[0] * ct + pl.nrm[0] * st, Q[1] * ct + pl.nrm[1] * st, Q[2] * ct + pl.nrm[2] * st];
      }
      for (let k = 0; k < 3; k++) out[k] = (c * pl.P[k] + s2 * Q[k]) * r;
      return out;
    };
    const v3 = [0, 0, 0];
    const hidden = (v) => v[2] < 0 && v[0] * v[0] + v[1] * v[1] < 1;
    /* one unbroken ring: the far half under the globe, the near half over it,
       sharing their end points, its brightness easing from one to the other
       (a step in brightness reads as a break), drawn in runs of equal alpha */
    const RN = 360, RP = new Float32Array((RN + 1) * 3), LV = 12;
    function ring(pl, r, front, aFar, aNear) {
      for (let k = 0; k <= RN; k++) {
        at(pl, (k / RN) * TWO_PI, r, 0, v3);
        RP[k * 3] = G.ex + v3[0] * G.R; RP[k * 3 + 1] = G.ey - v3[1] * G.R; RP[k * 3 + 2] = v3[2];
      }
      ctx.save();
      ctx.lineWidth = 1;
      ctx.lineCap = 'butt';
      ctx.lineJoin = 'round';
      let level = -1;
      for (let k = 0; k < RN; k++) {
        const z = (RP[k * 3 + 2] + RP[k * 3 + 5]) / 2;
        if ((z >= 0) !== front) { if (level >= 0) ctx.stroke(); level = -1; continue; }
        const lv = Math.round(smooth(-0.3, 0.3, z) * LV);
        if (lv !== level) {
          if (level >= 0) ctx.stroke();
          level = lv;
          ctx.strokeStyle = `rgba(219, 226, 234, ${(aFar + (aNear - aFar) * lv / LV).toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(RP[k * 3], RP[k * 3 + 1]);
        }
        ctx.lineTo(RP[k * 3 + 3], RP[k * 3 + 4]);
      }
      if (level >= 0) ctx.stroke();
      ctx.restore();
    }
    const v4 = [0, 0, 0];
    function debris(front) {
      const w0 = wD;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.strokeStyle = '#d6dee8';
      ctx.lineWidth = 1.1;
      for (const f of F) {
        const u = f.u + w0 * f.w * T;
        at(DEB.pl, u, DEB.r + f.dr, f.tilt, v3);
        if ((v3[2] >= 0) !== front || (front && hidden(v3))) continue;   // the globe hides the far ones
        at(DEB.pl, u - 0.02, DEB.r + f.dr, f.tilt, v4);   // where it was a moment ago
        ctx.globalAlpha = f.a * (0.35 + 0.6 * smooth(-0.3, 0.3, v3[2]));
        ctx.beginPath();
        ctx.moveTo(G.ex + v4[0] * G.R, G.ey - v4[1] * G.R);
        ctx.lineTo(G.ex + v3[0] * G.R, G.ey - v3[1] * G.R);
        ctx.stroke();
      }
      ctx.restore();
    }
    /* the station: oriented along its orbit (forward = velocity, up = away
       from Earth), drawn about 1:250,000 of the globe's scale so it reads */
    function station(front) {
      const u = U0 + wI * T;
      const r = at(ISS.pl, u, 1, 0, [0, 0, 0]);
      const vel = norm3([-Math.sin(u) * ISS.pl.P[0] + Math.cos(u) * ISS.pl.Q[0],
        -Math.sin(u) * ISS.pl.P[1] + Math.cos(u) * ISS.pl.Q[1], -Math.sin(u) * ISS.pl.P[2] + Math.cos(u) * ISS.pl.Q[2]]);
      const pos = [r[0] * ISS.r, r[1] * ISS.r, r[2] * ISS.r];
      if ((pos[2] >= 0) !== front || hidden(pos)) return null;
      const Z = r, Y = vel, X = cross3(Y, Z);           // starboard, forward, zenith in view space
      const cam = {
        right: [X[0], Y[0], Z[0]], up: [X[1], Y[1], Z[1]], back: [X[2], Y[2], Z[2]], D: 1e6,
        p(v) { return [dot3(v, this.right), dot3(v, this.up), dot3(v, this.back), 1]; },
        facing(d) { return dot3(d, this.back); },
      };
      const L = [dot3(SUNV, X), dot3(SUNV, Y), dot3(SUNV, Z)];
      const x = G.ex + pos[0] * G.R, y = G.ey - pos[1] * G.R;
      iss.render(ctx, x, y, G.R * 0.0019, cam, L);
      return { x, y, z: pos[2] };
    }
    function label(text, x, y, a, align) {
      if (a < 0.02) return;
      ctx.save();
      ctx.globalAlpha = a;
      ctx.font = `500 ${Math.round(clamp(G.S * 0.0068, 10, 13))}px "JetBrains Mono", ui-monospace, monospace`;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '2.6px';
      ctx.textBaseline = 'middle';
      ctx.textAlign = align || 'left';
      ctx.fillStyle = '#dbe2ea';
      ctx.shadowColor = 'rgba(5, 7, 12, 0.95)'; ctx.shadowBlur = 10;
      ctx.fillText(text, x, y);
      ctx.restore();
    }
    function step(dt) { if (!G) layout(); T += dt; }
    function render() {
      if (!G) layout();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ring(DEB.pl, DEB.r, false, 0.08, 0.16);
      ring(ISS.pl, ISS.r, false, 0.14, 0.32);
      debris(false);
      station(false);
      globe.draw(ctx, G.ex, G.ey, T);
      /* the air */
      const { ex, ey, R } = G;
      const g = ctx.createRadialGradient(ex, ey, R * 0.95, ex, ey, R * 1.08);
      g.addColorStop(0, 'rgba(120, 178, 255, 0)'); g.addColorStop(0.35, 'rgba(120, 178, 255, 0.34)');
      g.addColorStop(0.6, 'rgba(120, 178, 255, 0.1)'); g.addColorStop(1, 'rgba(120, 178, 255, 0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(ex, ey, R * 1.08, 0, TWO_PI); ctx.fill();
      ring(DEB.pl, DEB.r, true, 0.08, 0.16);
      ring(ISS.pl, ISS.r, true, 0.14, 0.32);
      debris(true);
      const st = station(true);
      if (o.labels !== false) {
        if (st) label('SPACE STATION', st.x + G.R * 0.16, st.y - G.R * 0.05, 0.9 * smooth(0.02, 0.3, st.z));
        /* the cloud's label sits where its ring is in front and clear of the globe */
        if (!G.debAt) {
          let bestU = 0, bestS = -9;
          for (let k = 0; k < 360; k++) {
            at(DEB.pl, (k / 360) * TWO_PI, DEB.r, 0, v3);
            const off = Math.hypot(v3[0], v3[1]);
            const sc = v3[2] > 0 && off > 1.06 ? v3[0] - Math.abs(v3[1]) * 0.3 : -9;
            if (sc > bestS) { bestS = sc; bestU = (k / 360) * TWO_PI; }
          }
          at(DEB.pl, bestU, DEB.r, 0, v3);
          G.debAt = [G.ex + v3[0] * G.R, G.ey - v3[1] * G.R, v3[0] >= 0 ? 1 : -1];
        }
        const [lx, ly, side] = G.debAt;
        label('DEBRIS CLOUD', lx + side * 14, ly, 0.75, side > 0 ? 'left' : 'right');
      }
    }
    return {
      layout, step, render,
      load: () => KDeckGlobe.loadHires(),
      restart() { T = 0; },
      meet: MEET,
      simulate(seconds) {
        const n = Math.max(1, Math.round(seconds * 60));
        for (let k = 0; k < n; k++) step(1 / 60);
        render();
      },
    };
  }

  /* ======================================================================
     10 February 2009, 16:56 UTC: Iridium 33 and Cosmos 2251 over Siberia
     ==================================================================== */
  /* Real geometry. Both orbit planes come from the archived element sets
     (each passes within 0.2 degrees of the collision point), the Earth is
     turned to that minute's sidereal time and the Sun is where it was: the
     collision happened in the polar night, about 789 km above the Taymyr
     Peninsula. The approach is sped up about 90 times. */
  function mountCrossing(canvas, opts) {
    const o = opts || {};
    const reduce = !!o.reduced;
    const ctx = canvas.getContext('2d');
    const J2000 = Date.UTC(2000, 0, 1, 12, 0, 0), TC = Date.UTC(2009, 1, 10, 16, 56, 0);
    const day = (TC - J2000) / 86400000;
    const GMST = ((280.46061837 + 360.98564736629 * day) % 360) * DEG;
    const gA = (357.529 + 0.98560028 * day) * DEG;
    const lamS = (280.459 + 0.98564736 * day) * DEG + (1.915 * Math.sin(gA) + 0.02 * Math.sin(2 * gA)) * DEG;
    const eps = (23.439 - 0.00000036 * day) * DEG;
    const SUN_I = [Math.cos(lamS), Math.cos(eps) * Math.sin(lamS), Math.sin(eps) * Math.sin(lamS)];
    const LAT = 72.5 * DEG, ALPHA = 97.9 * DEG + GMST;       // the collision point's right ascension then
    const ALT = 789 / RE;                                     // altitude, in Earth radii
    /* each orbit through the point: Iridium northbound, Cosmos just past its
       northernmost point, southbound */
    const ORB = [
      { name: 'IRIDIUM 33', inc: 86.3994 * DEG, north: true, rgb: '169, 205, 255', n: 260 },
      { name: 'COSMOS 2251', inc: 74.0355 * DEG, north: false, rgb: '226, 230, 236', n: 420 },
    ].map((b) => {
      const s0 = Math.asin(clamp(Math.sin(LAT) / Math.sin(b.inc), -1, 1));
      const uc = b.north ? s0 : Math.PI - s0;
      const raan = ALPHA - Math.atan2(Math.cos(b.inc) * Math.sin(uc), Math.cos(uc));
      return Object.assign(b, { uc, raan });
    });
    const W_APPROACH = 5.4 * DEG;                             // display angular rate, rad/s
    const T_HIT = 6.2;                                        // seconds after arrival

    let W = 0, H = 0, dpr = 1, T = 0, tIn = 0, G = null, earth = null, earthKey = '';
    let deb = null;                                           // fragments, made at the hit
    const rnd = seeded(2251);

    function layout() {
      const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
      const d = Math.min(2, window.devicePixelRatio || 1, Math.sqrt(8.5e6 / (w * h)));
      if (w === W && h === H && Math.abs(d - dpr) < 1e-3 && G) return false;
      W = w; H = h; dpr = d;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      const S = Math.min(W, H * 16 / 9);
      const yL = H * (o.limb || 0.57), R = S * 1.25;
      const cx = W / 2, cy = yL + R;
      const yX = H * (o.cross || 0.45);                        // where the two meet on screen
      const vy = clamp((cy - yX) / ((1 + ALT) * R), -0.99, 0.99);
      const tau = LAT - Math.asin(vy);
      const delta = -Math.PI / 2 - ALPHA;                      // turns the point to face us
      const N3 = [0, Math.cos(tau), Math.sin(tau)], E1 = [1, 0, 0], E2 = [0, Math.sin(tau), -Math.cos(tau)];
      G = { S, R, cx, cy, yL, tau, delta, N3, E1, E2, spin: GMST + delta };
      G.sun = view(SUN_I);
      earth = null; earthKey = '';
      return true;
    }
    /* inertial -> view (x right, y up, z toward us) */
    function view(v, out) {
      const { delta, E1, E2, N3 } = G;
      const x = v[0] * Math.cos(delta) - v[1] * Math.sin(delta), y = v[0] * Math.sin(delta) + v[1] * Math.cos(delta), z = v[2];
      const r = out || [0, 0, 0];
      for (let c = 0; c < 3; c++) r[c] = E1[c] * x + E2[c] * y + N3[c] * z;
      return r;
    }
    const tmp = [0, 0, 0], tmpV = [0, 0, 0];
    function orbitPoint(b, u, rr, out) {
      const cO = Math.cos(b.raan), sO = Math.sin(b.raan), ci = Math.cos(b.inc), si = Math.sin(b.inc);
      const cu = Math.cos(u), su = Math.sin(u);
      tmp[0] = rr * (cu * cO - su * ci * sO);
      tmp[1] = rr * (cu * sO + su * ci * cO);
      tmp[2] = rr * (su * si);
      return view(tmp, out);
    }
    const scr = (v) => [G.cx + v[0] * G.R, G.cy - v[1] * G.R, v[2]];
    const hidden = (v) => v[2] < 0 && v[0] * v[0] + v[1] * v[1] < 1;   // behind the Earth

    function paintEarth() {
      const key = `${W}x${H}@${dpr}`;
      if (earthKey === key && earth) return;
      if (!KDeckGlobe.ready) return;
      const top = Math.floor(G.yL - 3);
      const img = KDeckGlobe.paint({
        w: W, h: H - top, scale: Math.min(dpr, 1.5), cx: G.cx, cy: G.cy - top, R: G.R,
        tau: G.tau, omega: 0, spin: G.spin, sun: G.sun, night: 1.25, hires: true, hiresOnly: true,
      });
      if (img) { earth = { img, top }; earthKey = key; }
    }
    function drawEarth() {
      paintEarth();
      if (earth) ctx.drawImage(earth.img, 0, earth.top, W, H - earth.top);
      const { cx, cy, R } = G;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const g = ctx.createRadialGradient(cx, cy, R * 0.985, cx, cy, R * 1.02);
      g.addColorStop(0, 'rgba(110, 160, 255, 0)');
      g.addColorStop(0.43, 'rgba(110, 160, 255, 0.5)');
      g.addColorStop(0.62, 'rgba(110, 160, 255, 0.16)');
      g.addColorStop(1, 'rgba(110, 160, 255, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, G.yL - R * 0.03, W, H - G.yL + R * 0.03);
      ctx.restore();
    }
    function uNow(b) { return b.uc + W_APPROACH * (T - tIn - T_HIT); }
    function drawOrbit(b) {
      ctx.save();
      ctx.strokeStyle = `rgba(${b.rgb}, 0.32)`;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      let pen = false, prev = null;
      for (let k = 0; k <= 720; k++) {
        const v = orbitPoint(b, b.uc + (k / 2 - 180) * DEG, 1 + ALT, [0, 0, 0]);
        const vis = !hidden(v);
        if (prev && vis !== prev.vis) {
          /* cut the segment exactly where it meets the limb */
          let lo = prev.v, hi = v;
          for (let it = 0; it < 12; it++) {
            const m = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
            if (!hidden(m) === prev.vis) lo = m; else hi = m;
          }
          const [x, y] = scr(prev.vis ? lo : hi);
          if (prev.vis) { ctx.lineTo(x, y); pen = false; } else { ctx.moveTo(x, y); pen = true; }
        }
        if (vis) {
          const [x, y] = scr(v);
          if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
        }
        prev = { v, vis };
      }
      ctx.stroke();
      ctx.restore();
    }
    function makeDebris() {
      deb = ORB.map((b) => {
        const f = [];
        for (let k = 0; k < b.n; k++) {
          const g = (rnd() + rnd() + rnd() - 1.5) / 1.5;
          f.push({ dw: W_APPROACH * (0.12 + 0.22 * g), du: (rnd() - 0.5) * 0.4 * DEG,
            dr: (rnd() - 0.5) * 0.012, tilt: (rnd() - 0.5) * 2.2 * DEG, a: 0.45 + 0.5 * rnd() });
        }
        return f;
      });
    }
    function drawSats() {
      const t = T - tIn;
      for (let i = 0; i < 2; i++) {
        const b = ORB[i];
        if (t < T_HIT) {
          /* the satellite and a short trail */
          for (let k = 14; k >= 0; k--) {
            const v = orbitPoint(b, uNow(b) - k * 0.55 * DEG, 1 + ALT, tmpV);
            if (hidden(v)) continue;
            const [x, y] = scr(v);
            const a = 1 - k / 15;
            ctx.fillStyle = `rgba(${b.rgb}, ${0.55 * a * a})`;
            ctx.beginPath(); ctx.arc(x, y, 1.2 + 0.6 * a, 0, TWO_PI); ctx.fill();
          }
          const v = orbitPoint(b, uNow(b), 1 + ALT, tmpV);
          if (!hidden(v)) {
            const [x, y] = scr(v);
            const g = ctx.createRadialGradient(x, y, 0, x, y, 10);
            g.addColorStop(0, `rgba(${b.rgb}, 0.9)`); g.addColorStop(1, `rgba(${b.rgb}, 0)`);
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 10, 0, TWO_PI); ctx.fill();
            ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x, y, 2.4, 0, TWO_PI); ctx.fill();
            label(b.name, x, y, i === 0 ? 1 : -1, smooth(T_HIT - 0.2, T_HIT - 1.2, t) * smooth(0, 0.6, t));
          }
        } else {
          if (!deb) makeDebris();
          /* the two clouds: each runs on along its own orbit, spreading */
          const dt = t - T_HIT;
          const fade = 0.25 + 0.75 * smooth(0, 0.5, dt);
          ctx.fillStyle = `rgba(${b.rgb}, 1)`;
          for (const f of deb[i]) {
            const u = b.uc + f.du + f.dw * dt + W_APPROACH * 0.35 * dt;
            const v = orbitPoint({ raan: b.raan + f.tilt, inc: b.inc + f.tilt * 0.5 }, u, 1 + ALT + f.dr, tmpV);
            if (hidden(v)) continue;
            const [x, y] = scr(v);
            ctx.globalAlpha = f.a * fade;
            ctx.fillRect(x - 0.9, y - 0.9, 1.8, 1.8);
          }
          ctx.globalAlpha = 1;
        }
      }
    }
    function label(text, x, y, side, a) {
      if (a < 0.02) return;
      ctx.save();
      ctx.globalAlpha = a;
      ctx.font = `500 ${Math.round(clamp(G.S * 0.0072, 10, 14))}px "JetBrains Mono", ui-monospace, monospace`;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '2.6px';
      ctx.textBaseline = 'middle';
      ctx.textAlign = side > 0 ? 'left' : 'right';
      ctx.fillStyle = '#dbe2ea';
      ctx.shadowColor = 'rgba(5, 7, 12, 0.95)'; ctx.shadowBlur = 10;
      ctx.fillText(text, x + side * 16, y);
      ctx.restore();
    }
    function drawHit() {
      const t = T - tIn;
      const b = ORB[0];
      const v = orbitPoint(b, b.uc, 1 + ALT, tmpV);
      const [x, y] = scr(v);
      if (t < T_HIT) return;
      const age = t - T_HIT;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const ring = smooth(0, 1.4, age), a = 1 - smooth(0.2, 2.2, age);
      ctx.strokeStyle = `rgba(220, 235, 255, ${0.85 * a})`; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 4 + 60 * ring, 0, TWO_PI); ctx.stroke();
      const core = 1 - smooth(0, 1.2, age);
      const g = ctx.createRadialGradient(x, y, 0, x, y, 46);
      g.addColorStop(0, `rgba(255, 255, 255, ${core})`);
      g.addColorStop(0.25, `rgba(190, 215, 255, ${0.5 * core})`);
      g.addColorStop(1, 'rgba(150, 190, 255, 0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 46, 0, TWO_PI); ctx.fill();
      ctx.restore();
    }
    function step(dt) { if (!G) layout(); T += dt; }
    function render() {
      if (!G) layout();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      drawEarth();
      for (const b of ORB) drawOrbit(b);
      drawHit();
      drawSats();
    }
    return {
      layout, step, render,
      load: () => KDeckGlobe.loadHires().then(() => { earth = null; earthKey = ''; }),
      restart() { tIn = T; deb = null; },
      simulate(seconds) {
        const n = Math.max(1, Math.round(seconds * 60));
        for (let k = 0; k < n; k++) step(1 / 60);
        render();
      },
    };
  }

  window.KDeckSwarm = { mount: mountSwarm };
  window.KDeckStation = { mount: mountStation };
  window.KDeckCrossing = { mount: mountCrossing };
})();
