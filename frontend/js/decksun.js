/* Kessler deck - the space weather slide: the Sun, the solar wind bending
   around Earth's magnetic shield, the aurora and the storm-heated air.
   Global: KDeckSun.mount(canvas, {reduced, leaders}) -> scene
     scene.layout()     re-measure the canvas (true when the size changed)
     scene.prepare()    build the Sun's surface and Earth's imagery, in small
                        slices so the deck never stutters (returns a promise)
     scene.step(dt)     advance the scene by dt seconds
     scene.render()     draw one frame
     scene.anchors()    where the slide's labels go (CSS px, canvas space)

   An illustration of a strong (G3) storm, drawn with the physics of the
   Overview's Space weather lens (js/sunview.js): the wind flows around the
   magnetopause as a Rankine half-body, its standoff is Shue et al. (1998)
   for the storm's wind, the field is a dipole compressed on the day side and
   stretched into the tail, and the auroral ovals ring the geomagnetic poles,
   pushed toward the equator by the storm. The Sun is brought in close and
   distances are compressed (r^0.78) so it all fits one frame. Earth is the
   real Blue Marble / Black Marble imagery (js/deck-earth.js). */
(function () {
  'use strict';

  const TWO_PI = Math.PI * 2;
  const DEG = Math.PI / 180;
  const KAPPA = 0.78;
  const SHOCK_X = 1.3;
  const MAG_POLE = [80.8 * DEG, -72.7 * DEG];      // geomagnetic north pole (IGRF, 2025)
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

  /* the storm on stage: fast, dense wind with a southward field (G3, Kp 7) */
  const STORM = { speed: 800, density: 15, bz: -15, kp: 7 };
  function standoffRe(density, speed, bz) {
    const p = 1.6726e-6 * Math.max(0.5, density) * speed * speed;
    return (10.22 + 1.29 * Math.tanh(0.184 * (bz + 8.14))) * Math.pow(p, -1 / 6.6);
  }
  const STANDOFF = standoffRe(STORM.density, STORM.speed, STORM.bz);   // ~6 Earth radii

  /* 1-D value noise for slow animation */
  function hash(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
  function noise(x) {
    const i = Math.floor(x), f = x - i;
    const u = f * f * (3 - 2 * f);
    return lerp(hash(i), hash(i + 1), u);
  }

  /* 2-D value noise on a seeded 256 lattice, for the Sun's surface */
  const PERM = new Uint8Array(512), VAL = new Float32Array(256);
  (function seed() {
    let s = 20260210 >>> 0;
    const r = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
    for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
    for (let i = 0; i < 256; i++) VAL[i] = r();
  })();
  function vn(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const X = xi & 255, Y = yi & 255;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = VAL[PERM[PERM[X] + Y]], b = VAL[PERM[PERM[X + 1] + Y]];
    const c = VAL[PERM[PERM[X] + Y + 1]], d = VAL[PERM[PERM[X + 1] + Y + 1]];
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }
  function fbm(x, y, oct) {
    let s = 0, a = 0.5, f = 1, n = 0;
    for (let k = 0; k < oct; k++) { s += a * vn(x * f, y * f); n += a; a *= 0.5; f *= 2.07; }
    return s / n;
  }

  /* Rankine half-body: r(phi) = b (pi - phi) / sin(phi), phi from +X (downstream) */
  function bodyR(b, phi) {
    const s = Math.sin(phi);
    if (s < 1e-5) return phi > 1.5 ? b : 1e9;
    return b * (Math.PI - phi) / s;
  }
  function inBody(X, Y, b) { return Math.hypot(X, Y) < bodyR(b, Math.atan2(Math.abs(Y), X)); }

  /* the Sun's colour ramp: deep orange network -> gold -> white-hot */
  const SUN_RAMP = [
    [0.00, [122, 30, 6]], [0.34, [196, 66, 14]], [0.58, [246, 118, 30]],
    [0.80, [255, 172, 72]], [0.96, [255, 222, 150]], [1.12, [255, 246, 222]],
  ];
  function sunColor(I, out) {
    const R = SUN_RAMP;
    if (I <= R[0][0]) { out[0] = R[0][1][0]; out[1] = R[0][1][1]; out[2] = R[0][1][2]; return; }
    for (let k = 1; k < R.length; k++) {
      if (I <= R[k][0] || k === R.length - 1) {
        const t = clamp((I - R[k - 1][0]) / (R[k][0] - R[k - 1][0]), 0, 1);
        for (let c = 0; c < 3; c++) out[c] = R[k - 1][1][c] + (R[k][1][c] - R[k - 1][1][c]) * t;
        return;
      }
    }
  }

  function mount(canvas, opts) {
    const o = opts || {};
    const reduce = !!o.reduced;
    const ctx = canvas.getContext('2d');
    const glow = document.createElement('canvas');
    const gctx = glow.getContext('2d');
    const glow2 = document.createElement('canvas');
    const g2ctx = glow2.getContext('2d');
    const canBlur = typeof gctx.filter === 'string';
    const sunA = document.createElement('canvas');     // the surface, two states that cross-fade
    const sunB = document.createElement('canvas');

    let W = 0, H = 0, dpr = 1, gs = 0.5, T = 0;
    let G = null;                                      // layout
    let sunKey = '', sunReady = false, sunJob = 0, sunAt = -1;
    let rShield = 0, rShock = 0;

    /* ---------------------------------------------------------------- layout */
    function layout() {
      const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
      /* a pixel budget, not just a DPR cap (an ultrawide at 2x is 17 MP) */
      const d = Math.min(2, window.devicePixelRatio || 1, Math.sqrt(8.5e6 / (w * h)));
      if (w === W && h === H && Math.abs(d - dpr) < 1e-3 && G) return false;
      W = w; H = h; dpr = d;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      glow.width = glow2.width = Math.max(1, Math.round(W * gs));
      glow.height = glow2.height = Math.max(1, Math.round(H * gs));
      const S = Math.min(W, H * 16 / 9);               // the 16:9 stage inside any window
      const cx = W / 2, ey = H * (o.axis || 0.5);
      const R = clamp(Math.min(H * 0.058, S * 0.033), 20, 104);
      const ex = cx + S * 0.205;
      const sunR = H * 0.95;
      const limbX = cx - S * 0.355;
      /* the wind and the shield fade out before the type above and below */
      G = { S, R, ex, ey, sunR, sunX: limbX - sunR, limbX, fade0: H * 0.19, fade1: H * 0.31 };
      rShield = R * Math.pow(STANDOFF, KAPPA);
      rShock = R * Math.pow(STANDOFF * SHOCK_X, KAPPA);
      globe.size(R, dpr);
      windReady = false; linesKey = '';
      if (sunKey !== `${W}x${H}@${dpr.toFixed(3)}`) { sunReady = false; }
      return true;
    }
    const fadeY = (y) => 1 - smooth(G.fade0, G.fade1, Math.abs(y - G.ey));

    /* ------------------------------------------------------- the Sun's surface */
    /* EUV-like: a mottled network over a limb-darkened disc, a few bright
       active regions; built once per layout, in row slices */
    function buildSun(sync) {
      const key = `${W}x${H}@${dpr.toFixed(3)}`;
      const job = ++sunJob;
      const sc = Math.min(dpr, 1.25);                   // surface resolution (px per CSS px)
      const wC = Math.ceil(G.limbX + 3), hC = H;
      const w = Math.max(1, Math.round(wC * sc)), h = Math.max(1, Math.round(hC * sc));
      const imgA = new ImageData(w, h), imgB = new ImageData(w, h);
      const R = G.sunR, sx = G.sunX, sy = G.ey;
      const F = 24;                                     // network cells per radian
      const ar = [[1.12, 0.30, 0.10], [1.33, -0.24, 0.08], [1.46, 0.07, 0.06], [1.24, -0.06, 0.05]];
      const col = [0, 0, 0];
      let y = 0;
      return new Promise((resolve) => {
        function slice() {
          if (job !== sunJob) { resolve(false); return; }
          const t0 = performance.now();
          for (; y < h && (sync || performance.now() - t0 < 7); y++) {
            const py = (y + 0.5) / sc - sy;
            for (let x = 0; x < w; x++) {
              const px = (x + 0.5) / sc - sx;
              const d2 = (px * px + py * py) / (R * R);
              const p = (y * w + x) * 4;
              if (d2 >= 1) continue;
              const mu = Math.sqrt(1 - d2);
              const lat = Math.asin(py / R), lon = Math.atan2(px / R, mu);
              const u = lon * F, v = lat * F;
              /* bright network lanes (ridged) over cloudy fbm */
              const n1 = fbm(u, v, 5);
              const rid = 1 - Math.abs(2 * fbm(u * 1.9 + 31.7, v * 1.9 - 12.3, 3) - 1);
              const n2 = fbm(u * 1.07 + 57.1, v * 1.07 + 19.3, 5);
              let act = 0;
              for (const a of ar) {
                const dl = (lon - a[0]) / a[2], dt = (lat - a[1]) / a[2];
                act += Math.exp(-(dl * dl + dt * dt));
              }
              const limb = 0.66 + 0.34 * Math.pow(mu, 0.45);
              const edge = smooth(0, 0.004, 1 - Math.sqrt(d2));     // anti-aliased rim
              for (let s = 0; s < 2; s++) {
                const nn = s ? 0.5 * n1 + 0.5 * n2 : n1;
                const I = limb * (0.56 + 0.34 * nn + 0.2 * rid * rid) + act * (0.22 + 0.4 * nn);
                sunColor(I, col);
                const img = s ? imgB : imgA;
                img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2];
                img.data[p + 3] = 255 * edge;
              }
            }
          }
          if (y < h) { setTimeout(slice, 0); return; }
          for (const [c, img] of [[sunA, imgA], [sunB, imgB]]) {
            c.width = w; c.height = h;
            c.getContext('2d').putImageData(img, 0, 0);
          }
          sunKey = key; sunReady = true; sunAt = -1;
          resolve(true);
        }
        slice();
      });
    }

    function drawSun() {
      const { sunR, sunX, ey, limbX, S } = G;
      /* corona: a wide warm glow beyond the limb */
      const cor = ctx.createRadialGradient(sunX, ey, sunR * 0.985, sunX, ey, sunR + S * 0.5);
      cor.addColorStop(0, 'rgba(255, 190, 110, 0.5)');
      cor.addColorStop(0.07, 'rgba(255, 150, 70, 0.2)');
      cor.addColorStop(0.3, 'rgba(255, 120, 50, 0.06)');
      cor.addColorStop(1, 'rgba(255, 110, 40, 0)');
      ctx.fillStyle = cor;
      ctx.fillRect(0, 0, limbX + S * 0.52, H);
      /* streamers: fine rays off the limb, slowly breathing */
      const thMax = Math.asin(clamp((H * 0.62) / sunR, 0, 0.99));
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';
      const n = 110;
      for (let k = 0; k < n; k++) {
        const th = lerp(-thMax - 0.06, thMax + 0.06, (k + hash(k) * 0.8) / n);
        const len = S * (0.04 + 0.2 * Math.pow(noise(k * 0.61 + T * 0.12), 2.2)) * (1 - 0.35 * Math.abs(th) / thMax);
        const a = 0.05 + 0.07 * noise(k * 1.37 + T * 0.38);
        const c = Math.cos(th), s = Math.sin(th);
        const x0 = sunX + c * (sunR - 1), y0 = ey + s * (sunR - 1);
        const x1 = sunX + c * (sunR + len), y1 = ey + s * (sunR + len);
        const g = ctx.createLinearGradient(x0, y0, x1, y1);
        g.addColorStop(0, `rgba(255, 214, 160, ${a})`);
        g.addColorStop(1, 'rgba(255, 190, 120, 0)');
        ctx.strokeStyle = g;
        ctx.lineWidth = 1 + 2.2 * hash(k * 7.3);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        if ((k & 1) === 0) {
          gctx.strokeStyle = `rgba(255, 170, 100, ${a * 1.4})`;
          gctx.lineWidth = 7 * gs * 2;
          gctx.beginPath(); gctx.moveTo(x0 * gs, y0 * gs); gctx.lineTo((sunX + c * (sunR + len * 0.7)) * gs, (ey + s * (sunR + len * 0.7)) * gs); gctx.stroke();
        }
      }
      ctx.restore();
      /* the disc: a plain one until the surface is built, then the surface
         fades in over it (never a pop) */
      const show = !sunReady ? 0 : reduce ? 1 : sunAt < 0 ? 0 : clamp((T - sunAt) / 0.7, 0, 1);
      if (show < 1) {
        const disk = ctx.createRadialGradient(sunX, ey, sunR * 0.6, sunX, ey, sunR);
        disk.addColorStop(0, '#ffb04a'); disk.addColorStop(0.92, '#f2802c'); disk.addColorStop(1, '#c9521a');
        ctx.fillStyle = disk;
        ctx.beginPath(); ctx.arc(sunX, ey, sunR, 0, TWO_PI); ctx.fill();
      }
      if (show > 0) {
        const wD = Math.ceil(G.limbX + 3);
        ctx.globalAlpha = show;
        ctx.drawImage(sunA, 0, 0, sunA.width, sunA.height, 0, 0, wD, H);
        const boil = reduce ? 0.5 : 0.5 + 0.5 * Math.sin(T * 0.62);
        ctx.globalAlpha = show * boil;
        ctx.drawImage(sunB, 0, 0, sunB.width, sunB.height, 0, 0, wD, H);
        ctx.globalAlpha = 1;
      }
      /* the chromosphere: a thin bright rim, spicules, prominences */
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const rim = ctx.createRadialGradient(sunX, ey, sunR * 0.992, sunX, ey, sunR * 1.012);
      rim.addColorStop(0, 'rgba(255, 150, 70, 0)');
      rim.addColorStop(0.55, 'rgba(255, 196, 120, 0.55)');
      rim.addColorStop(1, 'rgba(255, 160, 80, 0)');
      ctx.fillStyle = rim;
      ctx.fillRect(0, 0, limbX + 20, H);
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(255, 150, 80, 0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k < 360; k++) {
        const th = lerp(-thMax, thMax, k / 359);
        const f = noise(k * 0.9 + T * 1.5);
        const len = 2 + 6 * f * f;
        const c = Math.cos(th), s = Math.sin(th);
        ctx.moveTo(sunX + c * (sunR - 1), ey + s * (sunR - 1));
        ctx.lineTo(sunX + c * (sunR + len), ey + s * (sunR + len));
      }
      ctx.stroke();
      drawProminences();
      ctx.restore();
      /* bloom source: a band around the limb (inside a radial gradient's inner
         circle the first stop fills everything, so the band starts at zero) */
      const r0 = sunR * 0.95, r1 = sunR + S * 0.1, at = (sunR - r0) / (r1 - r0);
      const gg = gctx.createRadialGradient(sunX * gs, ey * gs, r0 * gs, sunX * gs, ey * gs, r1 * gs);
      gg.addColorStop(0, 'rgba(255, 170, 90, 0)');
      gg.addColorStop(at, 'rgba(255, 190, 110, 0.7)');
      gg.addColorStop(1, 'rgba(255, 150, 70, 0)');
      gctx.fillStyle = gg;
      gctx.fillRect(0, 0, (limbX + S * 0.12) * gs, H * gs);
    }

    /* prominences: arches of plasma held up by the Sun's field, never still.
       Each one rises out of the limb; its threads waver and plasma streams
       along them (bright runs sliding from one footpoint to the other); the
       arch breathes and sways; then it drains back or lifts off, and a new
       one grows elsewhere on the limb. Nothing on the limb holds still. */
    const PROM_N = 4;                                  // alive at once
    const proms = [];
    let promSeed = 20260303 >>> 0;
    const prand = () => ((promSeed = (Math.imul(promSeed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const limbRange = () => Math.asin(clamp((H * 0.46) / G.sunR, 0.05, 0.99)) - 0.05;
    function newProm(age) {
      const lim = limbRange();
      const big = prand();
      const w = 0.045 + 0.085 * big;                   // footpoint spread, radians of limb
      let th = 0;
      for (let tries = 0; tries < 30; tries++) {
        th = (prand() * 2 - 1) * (lim - w * 0.6);
        if (proms.every((q) => Math.abs(q.th - th) > (q.w + w) * 0.75)) break;
      }
      const n = 16 + Math.round(12 * big);
      const threads = [];
      for (let k = 0; k < n; k++) {
        threads.push({ j: k / (n - 1) - 0.5, r1: prand(), r2: prand(), ph: prand() * 100,
          dir: prand() < 0.5 ? -1 : 1, speed: 26 + 34 * prand(), dash: 10 + 22 * prand() });
      }
      return { th, w, h: 0.022 + 0.05 * Math.pow(prand(), 0.8) * (0.6 + 0.6 * big),
        life: 11 + 12 * prand(), age, erupt: prand() < 0.35, lean: (prand() - 0.5) * 0.5,
        seed: prand() * 100, drift: (prand() - 0.5) * 0.01, threads };
    }
    /* on arrival they are already mid-life, so the limb is busy from the start */
    function ensureProms() {
      while (proms.length < PROM_N) proms.push(newProm(proms.length ? 1.5 + prand() * 7 : 4));
    }
    function stepProms(dt) {
      ensureProms();
      for (let q = proms.length - 1; q >= 0; q--) {
        const p = proms[q];
        p.age += dt;
        p.th += p.drift * dt;                          // footpoints creep along the limb
        if (p.age >= p.life) { proms.splice(q, 1); proms.push(newProm(0)); }
      }
    }
    const bez = (a, b, c, d, t) => { const u = 1 - t; return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d; };
    const PX = new Float32Array(19), PY = new Float32Array(19);
    function drawProminences() {
      ensureProms();
      const { sunR, sunX, ey } = G;
      const pt = (th, r) => [sunX + Math.cos(th) * r, ey + Math.sin(th) * r];
      const NN = 18;
      ctx.lineCap = 'round';
      for (const p of proms) {
        const u = p.age / p.life;
        const grow = smooth(0, 0.16, u);
        const lift = p.erupt ? Math.pow(smooth(0.6, 1, u), 1.5) : 0;
        const fade = 1 - smooth(p.erupt ? 0.74 : 0.78, 1, u);
        const env = grow * fade;
        if (env < 0.01) continue;
        /* the arch breathes and sways the whole time */
        const breathe = 1 + 0.14 * Math.sin(T * 0.9 + p.seed) + 0.06 * Math.sin(T * 2.1 + p.seed * 1.7);
        const hScale = (0.35 + 0.65 * grow) * (1 + 3.2 * lift) * breathe
          * (p.erupt ? 1 : 1 - 0.25 * smooth(0.78, 1, u));
        const spread = 1 + 0.6 * lift;
        const lean = p.lean + 0.18 * Math.sin(T * 0.55 + p.seed * 0.7);
        for (let k = 0; k < p.threads.length; k++) {
          const t = p.threads[k], j = t.j;
          const sway = noise(p.seed + t.ph + T * 0.45) - 0.5;
          const t1 = p.th - p.w * spread * (0.5 + 0.22 * j + 0.08 * t.r1);
          const t2 = p.th + p.w * spread * (0.5 - 0.22 * j + 0.08 * t.r2);
          const hh = sunR * p.h * hScale * (0.55 + 0.5 * t.r2 + 0.22 * sway) * (1 - 0.45 * Math.abs(j));
          const [ax, ay] = pt(t1, sunR - 2), [bx, by] = pt(t2, sunR - 2);
          const [c1x, c1y] = pt(t1 + p.w * (0.08 + 0.16 * sway), sunR + hh * (1.32 + lean));
          const [c2x, c2y] = pt(t2 - p.w * (0.08 - 0.16 * sway), sunR + hh * (1.32 - lean));
          for (let m = 0; m <= NN; m++) {
            const tt = m / NN;
            const x = bez(ax, c1x, c2x, bx, tt), y = bez(ay, c1y, c2y, by, tt);
            const wob = (noise(p.seed * 3 + t.ph + m * 0.85 + T * 1.25) - 0.5) * hh * 0.2 * Math.sin(Math.PI * tt);
            const nx = x - sunX, ny = y - ey, nl = Math.hypot(nx, ny) || 1;
            PX[m] = x - ny / nl * wob; PY[m] = y + nx / nl * wob;
          }
          const core = Math.max(0.15, 1 - Math.abs(j) * 1.5);
          const a = env * (0.05 + 0.07 * noise(t.ph + T * 0.8)) * (1 - 0.5 * lift);
          const g = Math.round(105 + 80 * core), b = Math.round(60 + 40 * core);
          ctx.beginPath(); ctx.moveTo(PX[0], PY[0]);
          for (let m = 1; m <= NN; m++) ctx.lineTo(PX[m], PY[m]);
          /* the thread itself, faint */
          ctx.setLineDash([]);
          ctx.strokeStyle = `rgba(255, ${g}, ${b}, ${a})`;
          ctx.lineWidth = 1.4 + 2.4 * core;
          ctx.stroke();
          /* plasma streaming along it: bright runs that slide footpoint to footpoint */
          ctx.setLineDash([t.dash, t.dash * 1.6]);
          ctx.lineDashOffset = -t.dir * T * t.speed;
          ctx.strokeStyle = `rgba(255, ${Math.min(230, g + 40)}, ${Math.min(180, b + 50)}, ${a * 1.9})`;
          ctx.lineWidth = 1 + 1.8 * core;
          ctx.stroke();
          if ((k & 1) === 0) {
            gctx.setLineDash([]);
            gctx.strokeStyle = `rgba(255, 120, 70, ${a * 2.2})`;
            gctx.lineWidth = 7 * gs * 2;
            gctx.beginPath(); gctx.moveTo(PX[0] * gs, PY[0] * gs);
            for (let m = 1; m <= NN; m++) gctx.lineTo(PX[m] * gs, PY[m] * gs);
            gctx.stroke();
          }
        }
      }
      ctx.setLineDash([]);
    }

    /* ------------------------------------------------------------ solar wind */
    const NW = 900, HL = 9;
    const wx = new Float32Array(NW), wy = new Float32Array(NW), wSeed = new Float32Array(NW);
    const wh = new Float32Array(NW * HL * 2);
    const wCls = new Uint8Array(NW), wLv = new Uint8Array(NW);
    let nWind = 0, wHead = 0, wTick = 0, windReady = false;
    function windCount() {
      return Math.min(NW, Math.round(clamp((G.S * H) / (1920 * 1080), 0.45, 1.5) * 620));
    }
    /* Earth-centred coordinates (X right, Y down); the Sun's centre is at (XS, 0) */
    function spawnWind(i, anywhere) {
      const XS = G.sunX - G.ex, R = G.sunR;
      const thMax = Math.asin(clamp((H * 0.6) / R, 0, 0.99));
      for (let tries = 0; tries < 10; tries++) {
        const g = Math.random() * 2 - 1;
        const th = thMax * Math.sign(g) * Math.pow(Math.abs(g), 1.35);
        const d = anywhere ? Math.random() * (G.ex - G.limbX + W * 0.2) : Math.random() * 6;
        const X = XS + Math.cos(th) * (R + 2 + d), Y = Math.sin(th) * (R + 2 + d);
        if (!inBody(X, Y, rShock * 1.02) || !anywhere) { wx[i] = X; wy[i] = Y; break; }
      }
      wSeed[i] = Math.random() * 1000;
      for (let h = 0; h < HL; h++) { const j = (i * HL + h) * 2; wh[j] = wx[i]; wh[j + 1] = wy[i]; }
    }
    function stepWind(dt) {
      if (!windReady) {
        nWind = windCount();
        for (let i = 0; i < NW; i++) spawnWind(i, true);
        windReady = true;
      }
      const XS = G.sunX - G.ex;
      const base = G.S * 0.15 * (reduce ? 0.35 : 1);
      const b = rShield, bs = rShock;
      const rec = (wTick++ % 2) === 0;
      if (rec) wHead = (wHead + 1) % HL;
      for (let i = 0; i < nWind; i++) {
        let X = wx[i], Y = wy[i];
        /* radial outflow from the Sun, deflected by the shield (source at Earth) */
        const dxs = X - XS, ds = Math.hypot(dxs, Y) || 1;
        const r2 = X * X + Y * Y + 1;
        let u = dxs / ds + b * X / r2, v = Y / ds + b * Y / r2;
        const sheath = inBody(X, Y, bs) ? 1 : 0;
        wCls[i] = sheath;
        let spd = base;
        if (sheath) {
          spd *= 0.58;
          const n = wSeed[i] + T * 1.5;
          u += (noise(n) - 0.5) * 0.4; v += (noise(n + 57.3) - 0.5) * 0.5;
        }
        X += u * spd * dt; Y += v * spd * dt;
        if (inBody(X, Y, b)) {                    // never inside the shield: slide along it
          const phi = Math.atan2(Math.abs(Y), X), rr = bodyR(b, phi) * 1.012;
          const m = rr / Math.hypot(X, Y); X *= m; Y *= m;
        }
        wx[i] = X; wy[i] = Y;
        if (rec) { const j = (i * HL + wHead) * 2; wh[j] = X; wh[j + 1] = Y; }
        if (X > W - G.ex + 40 || Math.abs(Y) > H * 0.66) spawnWind(i, false);
      }
    }
    /* gusts: denser fronts riding out from the Sun every few seconds */
    function gust(X, Y) {
      if (reduce) return 1;
      const d = Math.hypot(X - (G.sunX - G.ex), Y) - G.sunR;
      const ph = (d / (G.S * 0.42) - T * 0.36) % 1;
      const f = ph < 0 ? ph + 1 : ph;
      return 0.8 + 0.6 * Math.pow(Math.max(0, Math.cos((f - 0.5) * Math.PI)), 6);
    }
    function drawWind() {
      const ex = G.ex, ey = G.ey;
      const seg = [[8, 6], [6, 4], [4, 2], [2, 0]];
      const alpha = [0.04, 0.1, 0.18, 0.3];
      const LV = [0.38, 0.72, 1.05];
      for (let i = 0; i < nWind; i++) {
        const k = fadeY(ey + wy[i]) * gust(wx[i], wy[i]);
        wLv[i] = k < 0.06 ? 3 : k < 0.45 ? 0 : k < 0.85 ? 1 : 2;
      }
      ctx.save();
      ctx.lineCap = 'round';
      ctx.globalCompositeOperation = 'lighter';
      for (let cls = 0; cls < 2; cls++) {
        const rgb = cls ? '255, 232, 214' : '255, 196, 140';
        ctx.lineWidth = cls ? 1.2 : 1.05;
        for (let lv = 0; lv < 3; lv++) {
          for (let s = 0; s < 5; s++) {
            ctx.strokeStyle = `rgba(${rgb}, ${(s < 4 ? alpha[s] : 0.5) * LV[lv] * (cls ? 1.2 : 1)})`;
            ctx.beginPath();
            for (let i = 0; i < nWind; i++) {
              if (wCls[i] !== cls || wLv[i] !== lv) continue;
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
      }
      ctx.restore();
    }

    /* ----------------------------------------------- the shield and the shock */
    function phiOff(b) {
      const lim = Math.max(W, H) * 1.1;
      let lo = 0.05, hi = Math.PI - 0.01;
      for (let k = 0; k < 24; k++) { const m = (lo + hi) / 2; if (bodyR(b, m) > lim) lo = m; else hi = m; }
      return lo;
    }
    /* a body outline in short segments, each faded by its height on screen */
    function fadedBody(c, b, scale, rgb, a0, width, flankFade) {
      const { ex, ey } = G;
      const pm = phiOff(b), n = 70;
      c.lineWidth = width;
      for (let side = -1; side <= 1; side += 2) {
        for (let k = 0; k < n; k++) {
          const p0 = lerp(Math.PI, pm, k / n), p1 = lerp(Math.PI, pm, (k + 1) / n);
          const r0 = bodyR(b, p0), r1 = bodyR(b, p1);
          const x0 = ex + r0 * Math.cos(p0), y0 = ey + side * r0 * Math.sin(p0);
          const x1 = ex + r1 * Math.cos(p1), y1 = ey + side * r1 * Math.sin(p1);
          const f = flankFade ? Math.pow(1 - k / n, 1.6) : 1;
          const a = a0 * f * fadeY((y0 + y1) / 2);
          if (a < 0.004) continue;
          c.strokeStyle = `rgba(${rgb}, ${a})`;
          c.beginPath(); c.moveTo(x0 * scale, y0 * scale); c.lineTo(x1 * scale, y1 * scale); c.stroke();
        }
      }
    }
    function bodyPath(c, b, scale) {
      const { ex, ey } = G;
      const pm = phiOff(b), n = 90;
      c.beginPath();
      for (let k = 0; k <= n; k++) {
        const phi = lerp(pm, Math.PI, k / n), r = bodyR(b, phi);
        c.lineTo((ex + r * Math.cos(phi)) * scale, (ey - r * Math.sin(phi)) * scale);
      }
      for (let k = n; k >= 0; k--) {
        const phi = lerp(pm, Math.PI, k / n), r = bodyR(b, phi);
        c.lineTo((ex + r * Math.cos(phi)) * scale, (ey + r * Math.sin(phi)) * scale);
      }
    }
    function drawShields() {
      const { ex, ey } = G;
      /* inside the shield: a faint cool wash, strongest along the boundary */
      bodyPath(ctx, rShield, 1);
      /* kept close to the nose, so it never reaches the type */
      const wash = ctx.createRadialGradient(ex, ey, rShield * 0.35, ex, ey, rShield * 1.45);
      wash.addColorStop(0, 'rgba(70, 130, 255, 0)');
      wash.addColorStop(0.45, 'rgba(70, 130, 255, 0.022)');
      wash.addColorStop(0.59, 'rgba(90, 160, 255, 0.07)');
      wash.addColorStop(0.74, 'rgba(80, 145, 255, 0.03)');
      wash.addColorStop(0.88, 'rgba(70, 130, 255, 0.008)');
      wash.addColorStop(1, 'rgba(70, 130, 255, 0)');
      ctx.fillStyle = wash;
      ctx.fill();
      /* bow shock: heated plasma, warm and faint */
      ctx.save();
      ctx.lineCap = 'round';
      fadedBody(ctx, rShock, 1, '255, 214, 178', 0.36, 1.1, true);
      fadedBody(gctx, rShock, gs, '255, 190, 140', 0.42, 5 * gs * 2, true);
      /* magnetopause: crisp line, soft halo, a shimmer travelling along it */
      fadedBody(ctx, rShield, 1, '132, 192, 255', 0.11, 7, false);
      fadedBody(ctx, rShield, 1, '150, 205, 255', 0.66, 1.3, false);
      ctx.save();
      bodyPath(ctx, rShield, 1);
      ctx.setLineDash([H * 0.07, H * 0.42]);
      ctx.lineDashOffset = -T * H * 0.06;
      const sh = ctx.createLinearGradient(0, G.ey - G.fade1, 0, G.ey + G.fade1);
      const f0 = 0.5 - G.fade0 / (2 * G.fade1);
      sh.addColorStop(0, 'rgba(205, 232, 255, 0)');
      sh.addColorStop(clamp(f0, 0, 0.5), 'rgba(205, 232, 255, 0.42)');
      sh.addColorStop(clamp(1 - f0, 0.5, 1), 'rgba(205, 232, 255, 0.42)');
      sh.addColorStop(1, 'rgba(205, 232, 255, 0)');
      ctx.strokeStyle = sh;
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.restore();
      fadedBody(gctx, rShield, gs, '110, 175, 255', 0.5, 6 * gs * 2, false);
      ctx.restore();
    }

    /* ------------------------------------------------------------ field lines */
    const CLOSED_L = [1.55, 2.1, 2.8, 3.7, 4.9, 6.3, 8.1];
    let lines = [], linesKey = '', precipPaths = [];
    function buildLines(psi) {
      /* rebuilt whenever the dipole's lean changes at all: a coarse key made
         the loops snap in small steps as Earth turned */
      const key = `${rShield.toFixed(2)}|${G.R.toFixed(2)}|${psi.toFixed(5)}|${G.ex.toFixed(1)},${G.ey.toFixed(1)}|${W}`;
      if (key === linesKey) return;
      linesKey = key;
      lines = []; precipPaths = [];
      const cP = Math.cos(psi), sP = Math.sin(psi);
      const R = G.R, b = rShield;
      for (const fam of [{ phi: 0, a: 1 }, { phi: 0.85, a: 0.42 }, { phi: -0.85, a: 0.3 }]) {
        for (const L of (fam.phi === 0 ? CLOSED_L : [2.8, 4.9, 8.1])) {
          for (const side of [-1, 1]) {
            const lam0 = Math.acos(Math.sqrt(1 / L));
            const pts = [];
            const n = 64;
            for (let k = 0; k <= n; k++) {
              const lam = lerp(-lam0, lam0, k / n);
              const rr = L * Math.cos(lam) * Math.cos(lam);
              const x = side * rr * Math.cos(lam) * Math.cos(fam.phi);
              const z = rr * Math.sin(lam);
              const X0 = x * cP - z * sP, Z0 = x * sP + z * cP;
              const r = Math.hypot(X0, Z0) || 1e-6;
              const rd = R * Math.pow(r, KAPPA);
              let X = X0 / r * rd, Y = -Z0 / r * rd;
              if (X < 0) {
                const lim = 0.93 * bodyR(b, Math.atan2(Math.abs(Y), X));
                const q = Math.hypot(X, Y);
                const qq = q / Math.pow(1 + Math.pow(q / lim, 5), 1 / 5);
                X *= qq / q; Y *= qq / q;
              } else {
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
      for (const hemi of [-1, 1]) {                    // open lobes: polar cap -> the tail
        for (let k = 0; k < 4; k++) {
          const colat = (8 + k * 3.2) * DEG;
          const fx = Math.sin(colat), fz = Math.cos(colat);
          const ax0 = fx * cP - hemi * fz * sP, az0 = fx * sP + hemi * fz * cP;
          const p0x = ax0 * R, p0y = -az0 * R;
          const lobe = R * (1.55 + k * 0.42);
          const P = [[p0x, p0y], [p0x + R * (0.35 + 0.2 * k), p0y - hemi * R * (1.6 + 0.4 * k)],
            [R * (3.2 + k * 0.8), -hemi * lobe], [W, -hemi * lobe * (1 + 0.04 * k)]];
          const pts = [];
          for (let s = 0; s <= 48; s++) {
            const t = s / 48, u = 1 - t;
            pts.push(u * u * u * P[0][0] + 3 * u * u * t * P[1][0] + 3 * u * t * t * P[2][0] + t * t * t * P[3][0],
              u * u * u * P[0][1] + 3 * u * u * t * P[1][1] + 3 * u * t * t * P[2][1] + t * t * t * P[3][1]);
          }
          lines.push({ pts, a: 0.5 - k * 0.07, front: false, plane: true, open: true });
        }
      }
      for (const ln of lines) if (ln.plane && !ln.open && ln.L >= 3.7 && ln.side === 1) precipPaths.push(ln);
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

    /* precipitation: bright beads sliding down the outer loops into the ovals */
    const NP = 44;
    const pS = new Float32Array(NP), pLine = new Uint8Array(NP), pHemi = new Int8Array(NP), pSpd = new Float32Array(NP);
    let pAlive = 0;
    function stepPrecip(dt) {
      const want = Math.round(NP * clamp((STORM.kp - 4) / 5, 0, 1) * 0.7);
      while (pAlive < want && precipPaths.length) {
        const i = pAlive++;
        pS[i] = Math.random();
        pLine[i] = Math.floor(Math.random() * precipPaths.length);
        pHemi[i] = Math.random() < 0.5 ? -1 : 1;
        pSpd[i] = 0.16 + Math.random() * 0.22;
      }
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

    /* ------------------------------------------------------------------ Earth */
    /* the shared textured Earth (js/deckglobe.js). View frame: x right, y up,
       z toward the viewer; north leans toward us (the northern oval shows)
       and a little toward the Sun. */
    const globe = KDeckGlobe.sphere({ tau: 0.38, omega: 0.1, sun: [-1, 0.04, 0.32],
      spin0: -95 * DEG, period: 200, reduced: reduce });   // the Americas in daylight on arrival
    const { E1, E2, N3 } = globe.basis;
    const SUN = globe.sun;
    const earthDir = (lat, lon, out) => globe.dir(lat, lon, T, out);
    function drawEarth() { globe.draw(ctx, G.ex, G.ey, T); }

    /* the air: a thin blue limb on the day side, and the storm-heated
       thermosphere, swollen and violet, brightest where the energy pours in
       at the poles */
    const tmp = [0, 0, 0];
    function poleScreen(hemi) {
      earthDir(MAG_POLE[0], MAG_POLE[1], tmp);
      return { x: G.ex + hemi * tmp[0] * G.R, y: G.ey - hemi * tmp[1] * G.R, z: hemi * tmp[2] };
    }
    function drawAir(under) {
      const { ex, ey, R } = G;
      const breathe = reduce ? 0 : Math.sin(T * 0.8) * 0.5 + 0.5;
      if (under) {
        const th = 1.13 + 0.025 * breathe;
        const g = ctx.createRadialGradient(ex, ey, R * 0.98, ex, ey, R * th);
        g.addColorStop(0, `rgba(196, 150, 255, ${0.5 + 0.08 * breathe})`);
        g.addColorStop(0.35, 'rgba(177, 140, 255, 0.2)');
        g.addColorStop(1, 'rgba(177, 140, 255, 0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(ex, ey, R * th, 0, TWO_PI); ctx.fill();
        for (const hemi of [1, -1]) {                  // heating at the poles
          const pp = poleScreen(hemi);
          const cx = ex + (pp.x - ex) * 1.02, cy = ey + (pp.y - ey) * 1.02;
          const pg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.75);
          pg.addColorStop(0, `rgba(200, 150, 255, ${0.28 + 0.1 * breathe})`);
          pg.addColorStop(1, 'rgba(200, 150, 255, 0)');
          ctx.fillStyle = pg;
          ctx.beginPath(); ctx.arc(cx, cy, R * 0.75, 0, TWO_PI); ctx.fill();
        }
        gctx.strokeStyle = 'rgba(177, 140, 255, 0.5)';
        gctx.lineWidth = Math.max(2, R * 0.12 * gs * 2);
        gctx.beginPath(); gctx.arc(ex * gs, ey * gs, R * 1.05 * gs, 0, TWO_PI); gctx.stroke();
      } else {
        /* day-side limb: scattered blue light, toward the Sun */
        ctx.save();
        ctx.lineCap = 'butt';
        ctx.lineWidth = Math.max(1.2, R * 0.045);
        const n = 48;
        for (let k = 0; k < n; k++) {
          const a0 = (k / n) * TWO_PI, a1 = ((k + 1) / n) * TWO_PI;
          const mid = (a0 + a1) / 2;
          const toSun = Math.max(0, -Math.cos(mid));
          ctx.strokeStyle = `rgba(130, 185, 255, ${0.06 + 0.5 * Math.pow(toSun, 0.8)})`;
          ctx.beginPath(); ctx.arc(ex, ey, R * 1.004, a0, a1 + 0.002); ctx.stroke();
        }
        ctx.restore();
        const g = ctx.createRadialGradient(ex, ey, R * 0.86, ex, ey, R);
        g.addColorStop(0, 'rgba(177, 140, 255, 0)');
        g.addColorStop(1, `rgba(190, 150, 255, ${0.2 + 0.06 * breathe})`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(ex, ey, R, 0, TWO_PI); ctx.fill();
      }
    }

    /* auroral ovals around the geomagnetic poles: the storm pushes them
       toward the equator; brighter and wider on the night side, a red
       upper fringe in a strong storm */
    const pj = { x: 0, y: 0, d: 0 };
    function project(v, rr) { pj.x = G.ex + v[0] * rr; pj.y = G.ey - v[1] * rr; pj.d = v[2]; return pj; }
    const Dm = [0, 0, 0];
    function drawAurora() {
      earthDir(MAG_POLE[0], MAG_POLE[1], Dm);
      const D = Dm, S = SUN;
      const dot = D[0] * S[0] + D[1] * S[1] + D[2] * S[2];
      let X = [S[0] - dot * D[0], S[1] - dot * D[1], S[2] - dot * D[2]];
      const xl = Math.hypot(X[0], X[1], X[2]) || 1; X = X.map((v) => v / xl);
      const Y = [D[1] * X[2] - D[2] * X[1], D[2] * X[0] - D[0] * X[2], D[0] * X[1] - D[1] * X[0]];
      const th0 = (90 - clamp(67.5 - 2.4 * STORM.kp, 44, 68)) * DEG;
      const inten = clamp(0.25 + STORM.kp / 9 * 0.75, 0, 1);
      const red = clamp((STORM.kp - 6) / 3, 0, 1);
      const n = 160, rr = G.R * 1.025;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const hemi of [1, -1]) {
        for (let k = 0; k < n; k++) {
          const phi = (k / n) * TWO_PI;
          const night = (1 - Math.cos(phi)) / 2;
          const th = th0 + (2.8 * night - 1.4) * DEG;
          const wid = (2.4 + 3.6 * night) * DEG;
          const shimmer = reduce ? 0.8 : 0.5 + 0.5 * noise(k * 0.37 + T * 0.9 + (hemi > 0 ? 0 : 50));
          const a = inten * (0.35 + 0.65 * night) * shimmer;
          for (let layer = -1; layer <= 1; layer++) {
            const t = th + layer * wid * 0.45;
            const st = Math.sin(t), ct = Math.cos(t) * hemi;
            const v = [ct * D[0] + st * (Math.cos(phi) * X[0] + Math.sin(phi) * Y[0]),
              ct * D[1] + st * (Math.cos(phi) * X[1] + Math.sin(phi) * Y[1]),
              ct * D[2] + st * (Math.cos(phi) * X[2] + Math.sin(phi) * Y[2])];
            project(v, rr);
            if (pj.d < 0) {
              const dx = pj.x - G.ex, dy = pj.y - G.ey;
              if (dx * dx + dy * dy < G.R * G.R) continue;
            }
            const aa = a * (layer === 0 ? 1 : 0.5) * (pj.d < 0 ? 0.45 : 1);
            const rgb = layer === -1 && red > 0 ? `255, ${Math.round(110 + 90 * (1 - red))}, 140` : '96, 255, 170';
            ctx.fillStyle = `rgba(${rgb}, ${0.15 * aa})`;
            ctx.beginPath(); ctx.arc(pj.x, pj.y, Math.max(1, G.R * 0.034), 0, TWO_PI); ctx.fill();
            if (layer === 0 && (k & 1) === 0) {
              gctx.fillStyle = `rgba(70, 255, 150, ${0.22 * aa})`;
              gctx.beginPath(); gctx.arc(pj.x * gs, pj.y * gs, G.R * 0.07 * gs, 0, TWO_PI); gctx.fill();
            }
          }
        }
      }
      ctx.restore();
    }
    /* the dipole's lean on screen, for the field lines */
    function dipolePsi() {
      earthDir(MAG_POLE[0], MAG_POLE[1], tmp);
      return Math.atan2(-tmp[0], tmp[1]);
    }

    /* two LEO satellites (altitude exaggerated) with fading trails */
    const SATS = [
      { r: 1.21, inc: 53 * DEG, node: 0.6, per: 9.5, ph: 0 },
      { r: 1.31, inc: 97.6 * DEG, node: 2.2, per: 12, ph: 2.6 },
    ];
    const satV = [0, 0, 0];
    function satPos(sat, t, out) {
      const u = sat.ph + (t / sat.per) * TWO_PI;
      const c = Math.cos(u), s = Math.sin(u);
      /* in the equatorial frame (not spinning with Earth) */
      const xo = c, yo = s * Math.cos(sat.inc), zo = s * Math.sin(sat.inc);
      const xb = xo * Math.cos(sat.node) - yo * Math.sin(sat.node);
      const yb = xo * Math.sin(sat.node) + yo * Math.cos(sat.node);
      for (let k = 0; k < 3; k++) out[k] = E1[k] * xb + E2[k] * yb + N3[k] * zo;
      return out;
    }
    function drawSats(front) {
      ctx.save();
      for (const sat of SATS) {
        const rd = G.R * sat.r;
        for (let k = 24; k >= 0; k--) {
          satPos(sat, T - k * 0.06, satV);
          project(satV, rd);
          if ((pj.d >= 0) !== front) continue;
          if (pj.d < 0 && (pj.x - G.ex) ** 2 + (pj.y - G.ey) ** 2 < G.R * G.R) continue;
          const a = 1 - k / 25;
          ctx.fillStyle = `rgba(225, 238, 255, ${0.5 * a * a})`;
          ctx.fillRect(pj.x - 0.9, pj.y - 0.9, 1.8, 1.8);
        }
        satPos(sat, T, satV);
        project(satV, rd);
        const vis = (pj.d >= 0) === front && !(pj.d < 0 && (pj.x - G.ex) ** 2 + (pj.y - G.ey) ** 2 < G.R * G.R);
        if (vis) {
          ctx.fillStyle = '#ffffff';
          ctx.beginPath(); ctx.arc(pj.x, pj.y, 2.3, 0, TWO_PI); ctx.fill();
          gctx.fillStyle = 'rgba(220, 235, 255, 0.7)';
          gctx.beginPath(); gctx.arc(pj.x * gs, pj.y * gs, 3 * gs * 2, 0, TWO_PI); gctx.fill();
        }
      }
      ctx.restore();
    }

    /* ------------------------------------------------------------------ frame */
    function step(dt) {
      if (!G) layout();
      T += dt;
      if (sunReady && sunAt < 0) sunAt = T;
      buildLines(dipolePsi());
      if (dt > 0) { stepWind(dt); stepPrecip(dt); stepProms(dt); }
    }
    function render() {
      if (!G) layout();
      if (!windReady) stepWind(0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      gctx.setTransform(1, 0, 0, 1, 0, 0);
      gctx.clearRect(0, 0, glow.width, glow.height);
      drawSun();
      drawWind();
      drawShields();
      drawLines(false);
      drawSats(false);
      drawAir(true);
      drawEarth();
      drawAir(false);
      drawAurora();
      drawPrecip();
      drawLines(true);
      drawSats(true);
      let src = glow;
      if (canBlur) {
        g2ctx.setTransform(1, 0, 0, 1, 0, 0);
        g2ctx.clearRect(0, 0, glow2.width, glow2.height);
        g2ctx.filter = `blur(${Math.max(2, G.R * 0.18 * gs).toFixed(1)}px)`;
        g2ctx.drawImage(glow, 0, 0);
        g2ctx.filter = 'none';
        src = glow2;
      }
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = canBlur ? 0.95 : 0.35;
      ctx.drawImage(src, 0, 0, W, H);
      ctx.restore();
      if (o.leaders !== false) {                       // off where no labels sit beside the scene
        const an = anchors();
        drawLeaders([an.shield, an.aurora, an.air]);
      }
    }

    /* where the labels go: each an anchor on the scene and a text position */
    function anchors() {
      if (!G) layout();
      const { ex, ey, R, limbX } = G;
      const phi = Math.PI - 0.62, rs = bodyR(rShield, phi);
      const sx = ex + rs * Math.cos(phi), sy = ey - rs * Math.sin(phi);
      return {
        limbX,                                         // the type is centred right of the Sun
        wind: { x: lerp(limbX, ex - rShock, 0.4), y: ey - H * 0.15 },
        /* the shield's label sits out in the wind, past the bow shock */
        shield: { ax: sx, ay: sy, x: sx - R * 1.25, y: sy - R * 0.8, side: 'left' },
        aurora: { ax: ex + R * 0.32, ay: ey - R * 0.98, x: ex + R * 1.5, y: ey - R * 2.2, side: 'right' },
        air: { ax: ex + R * 0.62, ay: ey + R * 0.84, x: ex + R * 1.55, y: ey + R * 2.2, side: 'right' },
      };
    }
    /* label leaders: hairlines from each feature to its label */
    function drawLeaders(list) {
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.lineWidth = 1;
      for (const l of list) {
        ctx.strokeStyle = `rgba(220, 230, 245, ${0.34 * (l.alpha == null ? 1 : l.alpha)})`;
        ctx.beginPath(); ctx.moveTo(l.ax, l.ay); ctx.lineTo(l.x, l.y); ctx.stroke();
        ctx.fillStyle = `rgba(230, 238, 250, ${0.85 * (l.alpha == null ? 1 : l.alpha)})`;
        ctx.beginPath(); ctx.arc(l.ax, l.ay, 1.8, 0, TWO_PI); ctx.fill();
      }
      ctx.restore();
    }

    /* idempotent: builds what this size still lacks; a resize mid-build
       starts over at the new size */
    let warmed = false, sunWant = '', sunDone = Promise.resolve(true);
    function prepare(sync) {
      if (!G) layout();
      if (!warmed) { warmed = true; for (let k = 0; k < 240; k++) step(1 / 60); }
      const key = `${W}x${H}@${dpr.toFixed(3)}`;
      if (!sunReady && (sunWant !== key || sync)) { sunWant = key; sunDone = buildSun(sync); }
      return Promise.all([globe.load(), sunDone]);
    }

    return {
      layout, prepare, step, render, anchors, drawLeaders,
      ready: () => sunReady,
      /* stills for checks (a hidden tab gets no animation frames, and its
         timers crawl, so the surface is built in one go) */
      simulate(seconds) {
        if (!sunReady) { prepare(true); sunAt = T - 1; }
        const n = Math.max(1, Math.round(seconds * 60));
        for (let k = 0; k < n; k++) step(1 / 60);
        render();
      },
    };
  }

  window.KDeckSun = { mount, standoffRe };
})();
