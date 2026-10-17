/* Kessler deck - a small textured Earth, shared by the deck's scenes.
   Global: KDeckGlobe.sphere(opts) -> globe
     globe.size(R, dpr)           build the per-pixel lookup for a disc of radius R (CSS px)
     globe.draw(ctx, cx, cy, T)   draw the lit Earth at (cx, cy) for scene time T (seconds)
     globe.dir(lat, lon, T, out)  an Earth-fixed direction (radians) in view coordinates
     globe.basis                  {E1, E2, N3}: the inertial equatorial frame, in view coordinates
     globe.sun                    the Sun's direction, in view coordinates
     globe.load()                 decode the imagery (js/deck-earth.js): a promise, once per page
   KDeckGlobe.paint(opts) -> canvas  a large, still Earth (horizons, close-ups), one
                                pass over the pixels it covers; full-resolution
                                imagery when the app serves the page (loadHires())
   View frame: x right, y up, z toward the viewer. opts: tau (the north pole's
   lean toward the viewer), omega (its lean toward screen-left), sun, spin0
   (radians), period (seconds per turn), reduced (no spin). Imagery: NASA
   Blue Marble / Black Marble, treated like the app's globe, a touch brighter. */
(function () {
  'use strict';

  const TWO_PI = Math.PI * 2;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  const unit = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

  let tex = null, job = null;               // decoded once, shared by every globe
  function load() {
    const src = window.KDeckEarth;
    if (!src) return Promise.resolve(false);
    if (!job) {
      const decode = (url) => new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im); im.onerror = rej;
        im.src = url;
      });
      job = Promise.all([decode(src.day), decode(src.night)]).then(([d, n]) => {
        tex = treat(d, n);                 /* data URIs never taint a canvas */
        return true;
      }).catch(() => false);
    }
    return job;
  }
  const decode = (url) => new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im); im.onerror = rej;
    im.src = url;
  });
  function treat(d, n) {
    const grab = (im) => {
      const c = document.createElement('canvas');
      c.width = im.naturalWidth; c.height = im.naturalHeight;
      const cc = c.getContext('2d');
      cc.drawImage(im, 0, 0);
      return cc.getImageData(0, 0, c.width, c.height);
    };
    const di = grab(d), ni = grab(n);
    /* keep 82% of the chroma, a slight cool tint; city lights stay warm */
    const dd = di.data;
    for (let p = 0; p < dd.length; p += 4) {
      const l = 0.299 * dd[p] + 0.587 * dd[p + 1] + 0.114 * dd[p + 2];
      dd[p] = (l + 0.82 * (dd[p] - l)) * 0.96 * 0.95;
      dd[p + 1] = (l + 0.82 * (dd[p + 1] - l)) * 0.96;
      dd[p + 2] = Math.min(255, (l + 0.82 * (dd[p + 2] - l)) * 0.96 * 1.06);
    }
    /* night: only the lights glow. Black Marble shows snow, ice and land as
       pale grey-blue, which reads as daylight if it is lifted, so everything
       below city brightness is pressed down to a faint ground */
    const nd = ni.data;
    for (let p = 0; p < nd.length; p += 4) {
      const l = (0.299 * nd[p] + 0.587 * nd[p + 1] + 0.114 * nd[p + 2]) / 255;
      const gain = 0.14 + 1.05 * smooth(0.36, 0.82, l);
      nd[p] = Math.min(190, nd[p] * gain); nd[p + 1] = Math.min(178, nd[p + 1] * gain);
      nd[p + 2] = Math.min(150, nd[p + 2] * gain);
    }
    return { dw: di.width, dh: di.height, day: dd, nw: ni.width, nh: ni.height, night: nd };
  }
  /* the app's full-resolution imagery (4096 and 3600 px wide), for Earths far
     bigger than the frame. Only when served: a file:// page can't read pixels
     from files, so it keeps the embedded imagery. */
  let texHi = null, jobHi = null;
  function loadHires() {
    if (!/^https?:$/.test(location.protocol)) return load();
    if (!jobHi) {
      jobHi = Promise.all([decode('textures/earth_day.jpg'), decode('textures/earth_night.jpg')])
        .then(([d, n]) => { texHi = treat(d, n); return true; })
        .catch(() => load());
    }
    return jobHi;
  }

  /* a large, still Earth, painted once: o = {w, h (CSS px), scale (px per CSS px),
     cx, cy, R (the disc, in the output's CSS px), tau, omega, spin, sun, night
     (city-light gain), hires}. Returns a canvas, or null before any imagery. */
  function paint(o) {
    const T = (o.hires && texHi) || (o.hiresOnly && /^https?:$/.test(location.protocol) ? null : tex);
    if (!T) return null;
    const sc = o.scale || 1;
    const Wd = Math.max(1, Math.round(o.w * sc)), Hd = Math.max(1, Math.round(o.h * sc));
    const N3 = [-Math.sin(o.omega || 0) * Math.cos(o.tau || 0), Math.cos(o.omega || 0) * Math.cos(o.tau || 0), Math.sin(o.tau || 0)];
    const E1 = unit([1 - N3[0] * N3[0], -N3[0] * N3[1], -N3[0] * N3[2]]);
    const E2 = [N3[1] * E1[2] - N3[2] * E1[1], N3[2] * E1[0] - N3[0] * E1[2], N3[0] * E1[1] - N3[1] * E1[0]];
    const SUN = unit(o.sun || [-1, 0, 0.3]);
    const spin = o.spin || 0, cityGain = o.night == null ? 1.1 : o.night;
    const c = document.createElement('canvas');
    c.width = Wd; c.height = Hd;
    const g = c.getContext('2d');
    const img = g.createImageData(Wd, Hd), out = img.data;
    const Rd = o.R * sc, cxd = o.cx * sc, cyd = o.cy * sc;
    const { dw, dh, day, nw, nh, night } = T;
    for (let j = 0; j < Hd; j++) {
      const y = -(j + 0.5 - cyd) / Rd;
      if (Math.abs(y) > 1 + 1 / Rd) continue;
      for (let i = 0; i < Wd; i++) {
        const x = (i + 0.5 - cxd) / Rd;
        const r = Math.hypot(x, y);
        if (r >= 1 + 1 / Rd) continue;
        const rr = Math.min(r, 0.99999);
        const z = Math.sqrt(1 - rr * rr);
        const nx = r ? x / r * rr : 0, ny = r ? y / r * rr : 0;
        const bx = nx * E1[0] + ny * E1[1] + z * E1[2];
        const by = nx * E2[0] + ny * E2[1] + z * E2[2];
        const bz = nx * N3[0] + ny * N3[1] + z * N3[2];
        let u = (Math.atan2(by, bx) - spin) / TWO_PI + 0.5;
        u -= Math.floor(u);
        const v = clamp(0.5 - Math.asin(clamp(bz, -1, 1)) / Math.PI, 0, 0.99999);
        const fx = u * dw - 0.5, fy = v * (dh - 1);
        let x0 = Math.floor(fx); const tx = fx - x0;
        const y0 = Math.floor(fy), ty = fy - y0;
        if (x0 < 0) x0 += dw;
        const x1 = (x0 + 1) % dw, y1 = Math.min(dh - 1, y0 + 1);
        const a = (y0 * dw + x0) * 4, b = (y0 * dw + x1) * 4, cc = (y1 * dw + x0) * 4, d = (y1 * dw + x1) * 4;
        const sd = nx * SUN[0] + ny * SUN[1] + z * SUN[2];
        const dayK = smooth(-0.1, 0.16, sd);
        const lam = 0.24 + 1.0 * Math.pow(Math.max(0, sd), 0.7);
        const haze = Math.pow(1 - z, 3) * 0.55 * dayK;
        /* night: bilinear too, a close Earth would show blocky lights */
        const nfx = u * nw - 0.5, nfy = v * (nh - 1);
        let n0 = Math.floor(nfx); const ntx = nfx - n0;
        const m0 = Math.floor(nfy), nty = nfy - m0;
        if (n0 < 0) n0 += nw;
        const n1 = (n0 + 1) % nw, m1 = Math.min(nh - 1, m0 + 1);
        const na = (m0 * nw + n0) * 4, nb = (m0 * nw + n1) * 4, nc = (m1 * nw + n0) * 4, nd = (m1 * nw + n1) * 4;
        const p = (j * Wd + i) * 4;
        for (let ch = 0; ch < 3; ch++) {
          const top = day[a + ch] + (day[b + ch] - day[a + ch]) * tx;
          const bot = day[cc + ch] + (day[d + ch] - day[cc + ch]) * tx;
          const dv = (top + (bot - top) * ty) * lam;
          const hz = ch === 0 ? 70 : ch === 1 ? 125 : 215;
          const lit = dv + (hz - dv) * haze;
          const nt0 = night[na + ch] + (night[nb + ch] - night[na + ch]) * ntx;
          const nt1 = night[nc + ch] + (night[nd + ch] - night[nc + ch]) * ntx;
          const nt = (nt0 + (nt1 - nt0) * nty) * cityGain + (ch === 2 ? 14 : ch === 1 ? 8 : 5);
          out[p + ch] = lit * dayK + nt * (1 - dayK);
        }
        out[p + 3] = 255 * clamp((1 - r) * Rd + 0.5, 0, 1);
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }

  function sphere(opts) {
    const o = Object.assign({ tau: 0.38, omega: 0.1, sun: [-1, 0.04, 0.32], spin0: 0, period: 200, reduced: false, hires: false }, opts || {});
    const N3 = [-Math.sin(o.omega) * Math.cos(o.tau), Math.cos(o.omega) * Math.cos(o.tau), Math.sin(o.tau)];
    const E1 = unit([1 - N3[0] * N3[0], -N3[0] * N3[1], -N3[0] * N3[2]]);
    const E2 = [N3[1] * E1[2] - N3[2] * E1[1], N3[2] * E1[0] - N3[0] * E1[2], N3[0] * E1[1] - N3[1] * E1[0]];
    const SUN = unit(o.sun);
    const cvs = document.createElement('canvas');
    const c2 = cvs.getContext('2d');
    let lut = null, img = null, n = 0, R = 0, dpr = 1;
    const spinAt = (T) => o.spin0 + (o.reduced ? 0 : TWO_PI * T / o.period);

    function size(Rcss, d) {
      if (lut && Rcss === R && d === dpr) return;
      R = Rcss; dpr = d;
      const Rd = R * dpr;
      n = 2 * Math.ceil(Rd) + 4;
      cvs.width = n; cvs.height = n;
      img = c2.createImageData(n, n);
      const idx = [], lon = [], lat = [], sun = [], edge = [], mu = [];
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const x = (i + 0.5 - n / 2) / Rd, y = -(j + 0.5 - n / 2) / Rd;
          const r = Math.hypot(x, y);
          if (r >= 1 + 1 / Rd) continue;
          const rr = Math.min(r, 0.99999);
          const z = Math.sqrt(1 - rr * rr);
          const nx = x / (r || 1) * rr, ny = y / (r || 1) * rr;
          idx.push((j * n + i) * 4);
          lon.push(Math.atan2(nx * E2[0] + ny * E2[1] + z * E2[2], nx * E1[0] + ny * E1[1] + z * E1[2]));
          lat.push(Math.asin(clamp(nx * N3[0] + ny * N3[1] + z * N3[2], -1, 1)));
          sun.push(nx * SUN[0] + ny * SUN[1] + z * SUN[2]);
          edge.push(clamp((1 - r) * Rd + 0.5, 0, 1));
          mu.push(z);
        }
      }
      lut = { n: idx.length, idx: Int32Array.from(idx), lon: Float32Array.from(lon), lat: Float32Array.from(lat),
        sun: Float32Array.from(sun), edge: Float32Array.from(edge), mu: Float32Array.from(mu) };
    }

    function dir(lat, lon, T, out) {
      const s = spinAt(T);
      const xb = Math.cos(lat) * Math.cos(lon), yb = Math.cos(lat) * Math.sin(lon), zb = Math.sin(lat);
      const x1 = xb * Math.cos(s) - yb * Math.sin(s), y1 = xb * Math.sin(s) + yb * Math.cos(s);
      for (let c = 0; c < 3; c++) out[c] = E1[c] * x1 + E2[c] * y1 + N3[c] * zb;
      return out;
    }

    function draw(ctx, cx, cy, T) {
      if (!lut) return;
      const out = img.data;
      out.fill(0);
      const L = lut, s = spinAt(T);
      const TX = (o.hires && texHi) || tex;          // the full-resolution imagery once it's in
      if (TX) {
        const { dw, dh, day, nw, nh, night } = TX;
        for (let k = 0; k < L.n; k++) {
          let u = (L.lon[k] - s) / TWO_PI + 0.5;
          u -= Math.floor(u);
          const v = clamp(0.5 - L.lat[k] / Math.PI, 0, 0.99999);
          const fx = u * dw - 0.5, fy = v * (dh - 1);
          let x0 = Math.floor(fx); const tx = fx - x0;
          const y0 = Math.floor(fy), ty = fy - y0;
          if (x0 < 0) x0 += dw;
          const x1 = (x0 + 1) % dw, y1 = Math.min(dh - 1, y0 + 1);
          const a = (y0 * dw + x0) * 4, b = (y0 * dw + x1) * 4, c = (y1 * dw + x0) * 4, d = (y1 * dw + x1) * 4;
          const sd = L.sun[k];
          const dayK = smooth(-0.1, 0.16, sd);
          const lam = 0.24 + 1.0 * Math.pow(Math.max(0, sd), 0.7);
          const haze = Math.pow(1 - L.mu[k], 3) * 0.55 * dayK;
          const nk = (Math.floor(v * nh) * nw + Math.floor(u * nw) % nw) * 4;
          const p = L.idx[k];
          for (let ch = 0; ch < 3; ch++) {
            const top = day[a + ch] + (day[b + ch] - day[a + ch]) * tx;
            const bot = day[c + ch] + (day[d + ch] - day[c + ch]) * tx;
            const dv = (top + (bot - top) * ty) * lam;
            const hz = ch === 0 ? 70 : ch === 1 ? 125 : 215;
            const lit = dv + (hz - dv) * haze;
            const nt = night[nk + ch] * 1.1 + (ch === 2 ? 14 : ch === 1 ? 8 : 5);
            out[p + ch] = lit * dayK + nt * (1 - dayK);
          }
          out[p + 3] = 255 * L.edge[k];
        }
      } else {                                  // imagery not decoded yet: a shaded sphere
        for (let k = 0; k < L.n; k++) {
          const sd = L.sun[k], dayK = smooth(-0.1, 0.16, sd);
          const lam = 0.25 + 0.8 * Math.max(0, sd);
          const p = L.idx[k];
          out[p] = (34 * lam) * dayK + 4; out[p + 1] = (82 * lam) * dayK + 8; out[p + 2] = (150 * lam) * dayK + 16;
          out[p + 3] = 255 * L.edge[k];
        }
      }
      c2.putImageData(img, 0, 0);
      const sz = n / dpr;
      ctx.drawImage(cvs, cx - sz / 2, cy - sz / 2, sz, sz);
    }

    return { size, draw, dir, load, basis: { E1, E2, N3 }, sun: SUN, ready: () => !!tex };
  }

  window.KDeckGlobe = { sphere, load, loadHires, paint, get ready() { return !!(texHi || tex); } };
})();
