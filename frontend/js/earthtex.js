/* Kessler - treated Earth textures for the globe engine (LUT renderer).
   Global: KEarthTexture - load / ready / onReady / drawHemisphere.
   Contract: docs/CONTRACTS.md "v7 additions - Textured Earth" (the
   reduced-quality drag mode is deleted - every render is full quality).

   load() fetches textures/earth_day.jpg (4096x2048) + textures/earth_night.jpg
   (3600x1800) once (idempotent, single-flight), decodes them to ImageData, and
   applies the palette treatment in an offscreen pass so the raw imagery never
   reaches the screen. Retuned coefficients (previous look read muddy):

   - DAY   (Blue Marble): desaturate 35% toward luma (keep 65% of chroma -
     was 40%), darken x0.72 (was x0.55), slight cool tint (R x0.94, B x1.06).
   - NIGHT (Black Marble): luma-weighted gain 0.45 + 0.55*(luma/255)^2 - the
     dark base sits at ~x0.45 while bright city cores ramp toward x1.0 with
     the channel ratios preserved (lights stay warm) - then every channel is
     hard-clamped at NIGHT_MAX_CH.

   AMBER-RESERVATION HEURISTIC (design rule: amber/red are risk-only):
   --k-elevated resolves to #e0a63c - peak channel 224. NIGHT_MAX_CH = 112
   (half that peak) is the documented ceiling for any treated night channel,
   so city-light glow can never approach the elevated-amber intensity and
   risk dots stay unambiguous over the night side. With the city gain
   reaching x1.0 the clamp is now load-bearing (bright cores hit it), not
   just a guarantee behind a darker multiplier.

   drawHemisphere(ctx, {cx, cy, radiusPx, spin, tilt, gmstRad, sunDirTEME})
   renders the front hemisphere at FULL resolution from a precomputed
   per-disk-pixel LUT (opts.quality is accepted for byte-compatibility with
   callers and ignored - there is no reduced-quality mode):

   - LUT (rebuilt only when disk geometry - clip bbox, center, radius, dpr -
     or TILT or the textures change): per visible disk pixel, the day base
     column at G=0 packed with its 8-bit bilinear x-weight (col << 9 | w),
     the day row start (element index, y-weight in a parallel array), the
     night nearest column + row, and the view-space sphere normal y & z
     components for the sun term. Rows are stored as [start, count] spans so
     the render walks the LUT linearly. The build is a typed-array pass with
     the same fast inverse trig the sampler used (polynomial atan +
     16384-entry asin LUT), mirroring the left half onto the right when the
     row span is symmetric (longitude mirrors as u' = (0.5 - u) mod 1;
     latitude/normals are even in x), with all stores inlined on hoisted
     locals. Measured (harness section 24, host realm): 2.6 ms for the
     520 px reference disk at dpr 2 (260 device px radius), 10 ms for a
     canvas-filling 1040 px disk - the reference budget (<= 8 ms) holds, so
     the contract's half-resolution staging fallback for a > 8 ms rebuild is
     not needed; even 1040 px tilt drags stay inside a 60 fps frame.
   - Per frame the render is a pure resample: spin/GMST changes shift the
     cached base columns by one integer column offset per texture
     (quantization 360/4096 = 0.088 deg - under the engine's 0.2 deg GMST
     cadence), and each row is split into day / terminator-band / night RUNS
     (sin elevation along a row is unimodal, so the two monotone halves
     around its closed-form extremum are binary-searched for the +/-6 deg
     crossings): only band pixels pay the per-pixel smoothstep sun blend,
     where n.sun uses the CACHED normal - zero trig per pixel anywhere.
     Day bilinear runs on packed RGBA lanes (Uint32 loads, R|B and G blended
     together with 8-bit integer weights - byte-exact on uniform texels,
     within 2/255 of float filtering elsewhere). Measured: 2.5 ms/frame for
     the 520 px reference disk (target <= 4 ms), 4.3 ms at the dashboard's
     340 device-px radius, 9.3 ms for the 1040 px disk - drags render at
     full quality every frame at 60 fps.
   - DAY is sampled BILINEAR; the treated day texture carries a duplicated
     wrap column and a duplicated bottom row so neither x+1 nor y+1 ever
     branches. NIGHT stays NEAREST by decision: city lights are point-like -
     bilinear would smear single-texel lights and dim their peaks, and the
     terminator blend already softens the transition band.

   Treated textures are flat row-major Uint8 (RGBA) arrays built once at
   load, with parallel Uint32 views for the packed fetches. Two LUT slots
   are kept so the dashboard globe and the simulator context view can
   interleave without thrashing rebuilds; a tilt change on a known geometry
   rebuilds its own slot in place, and slot arrays are grow-only, so tilt
   drags rebuild with zero allocation.

   Returns false (draws nothing) until both textures are decoded - callers
   keep their wireframe until ready() flips. Sampling is clipped to the
   intersection of the disk with the destination canvas. The destination ctx
   is assumed to carry a uniform scale transform (the engine's DPR
   transform); its .a component sets the device sampling density.

   _loadImages is an injectable hook (harness: feed synthetic ImageData),
   _treated exposes the treated textures ({w, h, stride, data} - day carries
   the wrap column/row via stride and one extra row) for design-rule
   assertions, and _stats exposes LUT build counts / timings / cached-array
   bytes for the perf harness. */
'use strict';

const KEarthTexture = (() => {
  const DAY_URL = 'textures/earth_day.jpg';
  const NIGHT_URL = 'textures/earth_night.jpg';

  /* treatment constants - documented in the header block */
  const DAY_SAT_KEEP = 0.65;           // keep 65% of chroma = desaturate 35%
  const DAY_DARKEN = 0.72;
  const DAY_TINT_R = 0.94;             // slight cool tint
  const DAY_TINT_B = 1.06;
  const NIGHT_GAIN_BASE = 0.45;        // gain at luma 0 (dark terrain)
  const NIGHT_GAIN_CITY = 1.0;         // gain at luma 255 (city cores)
  const NIGHT_MAX_CH = 112;            // = round(0.5 * 224), peak of #e0a63c

  const TWO_PI = Math.PI * 2;
  const INV_2PI = 1 / TWO_PI;
  const HALF_PI = Math.PI / 2;
  const SIN6 = Math.sin(6 * Math.PI / 180);   // day/night blend half-band
  const INV_BAND = 1 / (2 * SIN6);

  const perfNow = (typeof performance !== 'undefined' && performance
    && typeof performance.now === 'function')
    ? () => performance.now() : () => Date.now();

  /* Little-endian RGBA lane layout (R bits 0-7, G 8-15, B 16-23, A 24-31)
     powers the packed bilinear; a byte-accurate scalar path covers the
     (practically nonexistent) big-endian case. */
  const IS_LE = (() => {
    const b = new ArrayBuffer(4);
    new Uint32Array(b)[0] = 0x0a0b0c0d;
    return new Uint8Array(b)[0] === 0x0d;
  })();
  const R_SH = IS_LE ? 0 : 24, G_SH = IS_LE ? 8 : 16, B_SH = IS_LE ? 16 : 8;
  const A_MASK = IS_LE ? 0xff000000 : 0x000000ff;

  /* Fast inverse trig for the LUT build (per-pixel only on rebuild, never in
     the frame render):
     - atan: odd minimax polynomial on [-1,1] + octant fixup, max error
       ~2e-5 rad = 0.0011 deg (a texel is 0.088 deg of longitude at 4096).
     - asin -> equirect v: 16384-entry LUT over z. dv error stays under
       ~1.5 texel rows only in the last couple of pole rows (asin slope
       blows up there); the poles are single-color ice/dark in both maps. */
  const AT_C1 = 0.99997726, AT_C3 = -0.33262347, AT_C5 = 0.19354346;
  const AT_C7 = -0.11643287, AT_C9 = 0.05265332, AT_C11 = -0.01172120;
  function fastAtan(t) {               // |t| <= 1
    const s = t * t;
    return t * (AT_C1 + s * (AT_C3 + s * (AT_C5 + s * (AT_C7 + s * (AT_C9 + s * AT_C11)))));
  }
  const VLUT_N = 16384;
  const VLUT_SCALE = VLUT_N / 2;
  const VLUT = new Float64Array(VLUT_N);
  for (let k = 0; k < VLUT_N; k++) {
    VLUT[k] = 0.5 - Math.asin(-1 + (2 * (k + 0.5)) / VLUT_N) / Math.PI;
  }

  /* treated textures: {w, h, stride, data:Uint8ClampedArray, d32:Uint32Array}
     - the day map carries one duplicated wrap column (stride = w + 1) and
     one duplicated bottom row so bilinear x+1 / y+1 never branch. */
  let dayTex = null, nightTex = null;
  let isReady = false;
  let loadPromise = null;
  const readyCbs = [];
  let texGen = 0;                      // bumped when the treated textures change

  /* shared scratch canvas (grow-only); per-slot ImageData buffers */
  let scratch = null, scratchCtx = null;

  /* --- LUT slots ------------------------------------------------------------ */

  /* Two slots so two live views with different disk geometry (dashboard globe
     + simulator context view) never thrash each other's LUT; a tilt change on
     a known geometry rebuilds its own slot in place. */
  function newSlot() {
    return {
      gen: -1, x0: 0, y0: 0, x1: 0, y1: 0, cxD: 0, cyD: 0, rD: 0, tilt: NaN,
      bw: 0, bh: 0, npx: 0, cap: 0, invR: 0, stamp: 0,
      rowI0: null, rowN: null,                       // Int32Array(bh)
      dayXF: null,                                   // Int32: base day col << 9 | fx (0..256)
      dayR0: null,                                   // Int32: day row start (u32 element index)
      dayFyI: null,                                  // Int32: bilinear y weight (0..256)
      nightX: null, nightR: null,                    // Int32: night col / row start (elements)
      nyA: null, nzA: null,                          // Float32: view normal y, z
      img: null, out32: null,                        // per-slot pixel buffer
    };
  }
  const slots = [newSlot(), newSlot()];
  let stampCounter = 0;
  let lutBuilds = 0, lastLutMs = 0, lastRenderMs = 0;

  /* --- treatment (runs once at load) ---------------------------------------- */

  function treatDay(img) {
    const w = img.width, h = img.height, src = img.data;
    const stride = w + 1;
    const out = new Uint8ClampedArray(stride * (h + 1) * 4);
    for (let y = 0; y < h; y++) {
      let sp = y * w * 4, dp = y * stride * 4;
      const rp = dp;                                 // row start (wrap source)
      for (let x = 0; x < w; x++, sp += 4, dp += 4) {
        const r = src[sp], g = src[sp + 1], b = src[sp + 2];
        const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        out[dp] = (l + DAY_SAT_KEEP * (r - l)) * DAY_DARKEN * DAY_TINT_R;
        out[dp + 1] = (l + DAY_SAT_KEEP * (g - l)) * DAY_DARKEN;
        out[dp + 2] = (l + DAY_SAT_KEEP * (b - l)) * DAY_DARKEN * DAY_TINT_B;
        out[dp + 3] = 255;
      }
      out[dp] = out[rp];                             // wrap column = column 0
      out[dp + 1] = out[rp + 1];
      out[dp + 2] = out[rp + 2];
      out[dp + 3] = 255;
    }
    out.copyWithin(h * stride * 4, (h - 1) * stride * 4, h * stride * 4);
    return { w, h, stride, data: out, d32: new Uint32Array(out.buffer) };
  }

  function treatNight(img) {
    const d = img.data;
    for (let p = 0; p < d.length; p += 4) {
      const r = d[p], g = d[p + 1], b = d[p + 2];
      const t = (0.2126 * r + 0.7152 * g + 0.0722 * b) * (1 / 255);
      const gain = NIGHT_GAIN_BASE + (NIGHT_GAIN_CITY - NIGHT_GAIN_BASE) * t * t;
      const rr = r * gain, gg = g * gain, bb = b * gain;
      d[p] = rr > NIGHT_MAX_CH ? NIGHT_MAX_CH : rr;
      d[p + 1] = gg > NIGHT_MAX_CH ? NIGHT_MAX_CH : gg;
      d[p + 2] = bb > NIGHT_MAX_CH ? NIGHT_MAX_CH : bb;
      d[p + 3] = 255;
    }
    return {
      w: img.width, h: img.height, stride: img.width, data: d,
      d32: new Uint32Array(d.buffer, d.byteOffset, img.width * img.height),
    };
  }

  /* --- loading -------------------------------------------------------------- */

  /* Default loader: decode both JPEGs via <img> + canvas -> [day, night]
     ImageData. Overridable (api._loadImages) so headless harnesses can feed
     synthetic textures without an image decoder. */
  function _loadImages() {
    const decode = (url) => new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const c = document.createElement('canvas');
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          const cc = c.getContext('2d', { willReadFrequently: true });
          cc.drawImage(img, 0, 0);
          resolve(cc.getImageData(0, 0, c.width, c.height));
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = () => reject(new Error(`earth texture failed: ${url}`));
      img.src = url;
    });
    return Promise.all([decode(DAY_URL), decode(NIGHT_URL)]);
  }

  function load() {
    if (loadPromise) return loadPromise;
    loadPromise = Promise.resolve()
      .then(() => api._loadImages())
      .then(([day, night]) => {
        dayTex = treatDay(day);
        nightTex = treatNight(night);
        texGen++;                      // LUTs baked against old textures are stale
        isReady = true;
        while (readyCbs.length) readyCbs.shift()();
      })
      .catch((err) => {
        loadPromise = null;            // allow a retry; callers stay wireframe
        throw err;
      });
    return loadPromise;
  }

  function ready() {
    return isReady;
  }

  function onReady(cb) {
    if (isReady) cb();
    else readyCbs.push(cb);
  }

  /* --- LUT build ------------------------------------------------------------ */

  function ensureScratch(bw, bh) {
    if (!scratch) {
      scratch = document.createElement('canvas');
      scratchCtx = scratch.getContext('2d');
    }
    if (scratch.width < bw) scratch.width = bw;      // grow-only, no churn
    if (scratch.height < bh) scratch.height = bh;
  }

  /* Precompute the per-disk-pixel LUT for this geometry + tilt. Typed-array
     pass on hoisted locals, grow-only allocations - a tilt drag rebuilds
     with zero GC churn. */
  function buildLut(L, x0, y0, x1, y1, cxD, cyD, rD, tilt) {
    const t0 = perfNow();
    const bw = x1 - x0, bh = y1 - y0;
    const invR = 1 / rD;
    const cT = Math.cos(tilt), sT = Math.sin(tilt);
    const dw = dayTex.w, dh = dayTex.h, dStrideEl = dayTex.stride;
    const nw = nightTex.w, nh = nightTex.h;

    if (!L.rowI0 || L.rowI0.length < bh) {
      L.rowI0 = new Int32Array(bh);
      L.rowN = new Int32Array(bh);
    }
    const rowI0 = L.rowI0, rowN = L.rowN;

    /* pass 1: row spans + pixel count (bh sqrt calls - negligible) */
    let npx = 0;
    for (let j = 0; j < bh; j++) {
      const zs = (cyD - (y0 + j + 0.5)) * invR;
      const rr = 1 - zs * zs;
      rowN[j] = 0;
      if (rr <= 0) continue;
      const half = Math.sqrt(rr) * rD;
      let i0 = Math.ceil(cxD - half - x0 - 0.5);
      let i1 = Math.floor(cxD + half - x0 - 0.5);
      if (i0 < 0) i0 = 0;
      if (i1 >= bw) i1 = bw - 1;
      if (i1 < i0) continue;
      rowI0[j] = i0;
      rowN[j] = i1 - i0 + 1;
      npx += rowN[j];
    }
    if (npx > L.cap) {
      L.cap = npx;
      L.dayXF = new Int32Array(npx);
      L.dayR0 = new Int32Array(npx);
      L.dayFyI = new Int32Array(npx);
      L.nightX = new Int32Array(npx);
      L.nightR = new Int32Array(npx);
      L.nyA = new Float32Array(npx);
      L.nzA = new Float32Array(npx);
    }
    const dayXF = L.dayXF, dayR0 = L.dayR0, dayFyI = L.dayFyI;
    const nightX = L.nightX, nightR = L.nightR, nyA = L.nyA, nzA = L.nzA;

    /* mirror pairing: pixels i and mSum - i sit at +/-xs when mSum is (near)
       integer; longitude mirrors algebraically (u' = (0.5 - u) mod 1) and
       everything else is even in xs, so symmetric rows run the inverse trig
       for the left half only. */
    const mSumF = 2 * (cxD - x0) - 1;
    const mSum = Math.round(mSumF);
    const mirrorable = Math.abs(mSumF - mSum) < 1e-9;

    let k = 0;
    for (let j = 0; j < bh; j++) {
      const n = rowN[j];
      if (!n) continue;
      const i0 = rowI0[j], i1 = i0 + n - 1;
      const zs = (cyD - (y0 + j + 0.5)) * invR;
      const rr = 1 - zs * zs;
      const zsT = zs * sT, zcT = zs * cT;
      const sym = mirrorable && (i0 + i1 === mSum);
      const nCompute = sym ? ((n + 1) >> 1) : n;
      let xs = (x0 + i0 + 0.5 - cxD) * invR;
      const base = k;
      for (let c = 0; c < nCompute; c++, xs += invR) {
        const kl = base + c;
        let d2 = rr - xs * xs;
        if (d2 < 0) d2 = 0;
        const ys = -Math.sqrt(d2);
        const y1v = ys * cT + zsT;
        const zv = zcT - ys * sT;
        /* equirect u at G = 0 via polynomial atan2 */
        const ax = xs < 0 ? -xs : xs, ay = y1v < 0 ? -y1v : y1v;
        let lon;
        if (ay <= ax) {
          lon = ax < 1e-12 ? 0 : fastAtan(y1v / xs);
          if (xs < 0) lon += y1v >= 0 ? Math.PI : -Math.PI;
        } else {
          lon = (y1v >= 0 ? HALF_PI : -HALF_PI) - fastAtan(xs / y1v);
        }
        const u = lon * INV_2PI + 0.5;
        /* equirect v (+90 lat at top) via the asin LUT */
        let vi = ((zv + 1) * VLUT_SCALE) | 0;
        if (vi >= VLUT_N) vi = VLUT_N - 1; else if (vi < 0) vi = 0;
        const vv = VLUT[vi];
        /* day: packed base column + 8-bit x weight; row start + y weight
           (the texture's pad column/row absorb x+1 / y+1) */
        let xf = u * dw - 0.5;
        if (xf < 0) xf += dw;
        let xd = xf | 0;
        if (xd >= dw) xd = dw - 1;                   // u == 1 edge guard
        dayXF[kl] = (xd << 9) | (((xf - xd) * 256 + 0.5) | 0);
        const yf = vv * dh - 0.5;
        let yd = yf | 0;
        let fyI = ((yf - yd) * 256 + 0.5) | 0;
        if (yf < 0) { yd = 0; fyI = 0; }             // top pole clamp
        dayR0[kl] = yd * dStrideEl;
        dayFyI[kl] = fyI;
        /* night: nearest */
        let xn = (u * nw) | 0;
        if (xn >= nw) xn = nw - 1;
        let yn = (vv * nh) | 0;
        if (yn >= nh) yn = nh - 1;
        nightX[kl] = xn;
        nightR[kl] = yn * nw;
        nyA[kl] = ys;
        nzA[kl] = zs;
        if (sym) {
          const kr = base + n - 1 - c;
          if (kr !== kl) {
            let ur = (y1v >= 0 ? 1.5 : 0.5) - u;     // lon' = sgn(y1v)*PI - lon
            if (ur >= 1) ur -= 1;
            let xfr = ur * dw - 0.5;
            if (xfr < 0) xfr += dw;
            let xdr = xfr | 0;
            if (xdr >= dw) xdr = dw - 1;
            dayXF[kr] = (xdr << 9) | (((xfr - xdr) * 256 + 0.5) | 0);
            dayR0[kr] = dayR0[kl];
            dayFyI[kr] = fyI;
            let xnr = (ur * nw) | 0;
            if (xnr >= nw) xnr = nw - 1;
            nightX[kr] = xnr;
            nightR[kr] = nightR[kl];
            nyA[kr] = ys;
            nzA[kr] = zs;
          }
        }
      }
      k = base + n;
    }
    L.npx = k;
    L.gen = texGen;
    L.x0 = x0; L.y0 = y0; L.x1 = x1; L.y1 = y1;
    L.cxD = cxD; L.cyD = cyD; L.rD = rD; L.tilt = tilt;
    L.bw = bw; L.bh = bh; L.invR = invR;

    /* per-slot pixel buffer: outside-disk bytes stay transparent; every
       in-span pixel is fully rewritten each render, so the clear runs only
       here (rebuild), never per frame. */
    if (!L.img || L.img.width !== bw || L.img.height !== bh) {
      L.img = scratchCtx.createImageData(bw, bh);
      L.out32 = new Uint32Array(L.img.data.buffer, L.img.data.byteOffset, bw * bh);
    } else {
      L.img.data.fill(0);
    }
    lutBuilds++;
    lastLutMs = perfNow() - t0;
  }

  /* --- per-frame render ----------------------------------------------------- */

  /* Branch-free day run: packed-lane bilinear (R|B and G lanes blended
     together with the 8-bit integer weights), one Uint32 load per tap and
     one store per pixel. Pixel indices a..b of the LUT row starting at kb+a. */
  function dayRunLE(out, D, dayXF, dayR0, dayFyI, dw, dStrideEl, duSh, kb, pRow, a, b) {
    for (let i = a; i <= b; i++) {
      const k = kb + i;
      const tpk = dayXF[k] + duSh;
      let x = tpk >>> 9;
      if (x >= dw) x -= dw;
      const fx = tpk & 511, gx = 256 - fx;
      const o0 = dayR0[k] + x, o1 = o0 + dStrideEl;
      const c00 = D[o0], c01 = D[o0 + 1], c10 = D[o1], c11 = D[o1 + 1];
      const fy = dayFyI[k], gy = 256 - fy;
      const rb0 = (((c00 & 0xff00ff) * gx + (c01 & 0xff00ff) * fx) >>> 8) & 0xff00ff;
      const gg0 = (((c00 & 0xff00) * gx + (c01 & 0xff00) * fx) >>> 8) & 0xff00;
      const rb1 = (((c10 & 0xff00ff) * gx + (c11 & 0xff00ff) * fx) >>> 8) & 0xff00ff;
      const gg1 = (((c10 & 0xff00) * gx + (c11 & 0xff00) * fx) >>> 8) & 0xff00;
      out[pRow + i] = ((((rb0 * gy + rb1 * fy) >>> 8) & 0xff00ff)
        | (((gg0 * gy + gg1 * fy) >>> 8) & 0xff00)) | 0xff000000;
    }
  }

  /* branch-free night run: nearest fetch (treated alpha is already 255) */
  function nightRunLE(out, N, nightX, nightR, nw, duN, kb, pRow, a, b) {
    for (let i = a; i <= b; i++) {
      const k = kb + i;
      let xn = nightX[k] + duN;
      if (xn >= nw) xn -= nw;
      out[pRow + i] = N[nightR[k] + xn];
    }
  }

  /* sin(sun elevation) at pixel i of a row - the exact expression the band
     loop and the run bisection share, so run boundaries land on the same
     pixels a per-pixel evaluation would pick. */
  function seVal(nyA, kb, i, x0, cxD, invR, s2x, s2y, rowC) {
    return ((x0 + i + 0.5 - cxD) * invR) * s2x + nyA[kb + i] * s2y + rowC;
  }

  /* First index in [lo, hi+1] where the mode predicate on se holds (hi+1 when
     none) - se is monotone over the searched half of the row.
     mode 0: se < thr   1: se <= thr   2: se > thr   3: se >= thr */
  function seFirst(mode, thr, nyA, kb, lo, hi, x0, cxD, invR, s2x, s2y, rowC) {
    let a = lo, b = hi + 1;
    while (a < b) {
      const m = (a + b) >> 1;
      const se = seVal(nyA, kb, m, x0, cxD, invR, s2x, s2y, rowC);
      const p = mode === 0 ? se < thr : mode === 1 ? se <= thr
        : mode === 2 ? se > thr : se >= thr;
      if (p) b = m; else a = m + 1;
    }
    return a;
  }

  /* Pure resample from the LUT: integer column shifts carry spin+GMST, the
     cached normals carry the sun term - no trig, no allocation. Each row is
     split into day / terminator-band / night RUNS: se(x) along a row is
     unimodal (its derivative s2x + s2y*x/sqrt(rr-x^2) is monotone), so the
     two monotone halves around the closed-form extremum are binary-searched
     for the +/-SIN6 crossings and only the band pixels (a few percent) pay
     the per-pixel sun + blend math. */
  function renderLutPacked(L, duD, duN, sunOk, s2x, s2y, s2z) {
    const D = dayTex.d32, dw = dayTex.w, dStrideEl = dayTex.stride;
    const N = nightTex.d32, nw = nightTex.w;
    const rowI0 = L.rowI0, rowN = L.rowN;
    const dayXF = L.dayXF, dayR0 = L.dayR0, dayFyI = L.dayFyI;
    const nightX = L.nightX, nightR = L.nightR;
    const nyA = L.nyA, nzA = L.nzA;
    const out = L.out32;
    const bw = L.bw, bh = L.bh, x0 = L.x0, cxD = L.cxD, invR = L.invR;
    const rD = L.rD, duSh = duD << 9;

    let base = 0;
    for (let j = 0; j < bh; j++) {
      const n = rowN[j];
      if (!n) continue;
      const i0 = rowI0[j], i1 = i0 + n - 1;
      const kb = base - i0;                          // nyA[kb + i]
      const pRow = j * bw;
      base += n;
      if (!sunOk) {                                  // absent sun -> full day
        dayRunLE(out, D, dayXF, dayR0, dayFyI, dw, dStrideEl, duSh, kb, pRow, i0, i1);
        continue;
      }
      const zs = nzA[kb + i0];
      const rowC = zs * s2z;

      /* monotone halves around the se extremum (closed form: xs/sq = -s2x/s2y) */
      let iM;                                        // last index of the first half
      let d1, d2;                                    // half directions: 1 = decreasing
      if (s2y > 0 || s2y < 0) {
        const t = -s2x / s2y;
        const rr = 1 - zs * zs;
        const xsE = t * Math.sqrt(rr > 0 ? rr / (1 + t * t) : 0);
        iM = Math.round(xsE * rD + (cxD - x0) - 0.5);
        if (iM < i0) iM = i0; else if (iM > i1) iM = i1;
        d1 = s2y > 0 ? 1 : 0;                        // convex: min at iM (falls, rises)
        d2 = 1 - d1;
      } else if (s2x !== 0) {
        iM = i1;                                     // linear: one monotone segment
        d1 = s2x < 0 ? 1 : 0;
        d2 = d1;
      } else {                                       // constant se = rowC
        if (rowC >= SIN6) {
          dayRunLE(out, D, dayXF, dayR0, dayFyI, dw, dStrideEl, duSh, kb, pRow, i0, i1);
        } else if (rowC <= -SIN6) {
          nightRunLE(out, N, nightX, nightR, nw, duN, kb, pRow, i0, i1);
        } else {
          bandRun(L, duD, duN, s2x, s2y, rowC, kb, pRow, i0, i1);
        }
        continue;
      }

      for (let seg = 0; seg < 2; seg++) {
        const a = seg === 0 ? i0 : iM + 1;
        const b = seg === 0 ? iM : i1;
        if (a > b) continue;
        const dec = seg === 0 ? d1 : d2;
        let e1, e2;                                  // run boundaries within [a, b]
        if (dec) {
          /* falling: day [a, e1-1], band [e1, e2-1], night [e2, b] */
          e1 = seFirst(0, SIN6, nyA, kb, a, b, x0, cxD, invR, s2x, s2y, rowC);
          e2 = seFirst(1, -SIN6, nyA, kb, e1, b, x0, cxD, invR, s2x, s2y, rowC);
          if (e1 > a) dayRunLE(out, D, dayXF, dayR0, dayFyI, dw, dStrideEl, duSh, kb, pRow, a, e1 - 1);
          if (e2 > e1) bandRun(L, duD, duN, s2x, s2y, rowC, kb, pRow, e1, e2 - 1);
          if (b >= e2) nightRunLE(out, N, nightX, nightR, nw, duN, kb, pRow, e2, b);
        } else {
          /* rising: night [a, e1-1], band [e1, e2-1], day [e2, b] */
          e1 = seFirst(2, -SIN6, nyA, kb, a, b, x0, cxD, invR, s2x, s2y, rowC);
          e2 = seFirst(3, SIN6, nyA, kb, e1, b, x0, cxD, invR, s2x, s2y, rowC);
          if (e1 > a) nightRunLE(out, N, nightX, nightR, nw, duN, kb, pRow, a, e1 - 1);
          if (e2 > e1) bandRun(L, duD, duN, s2x, s2y, rowC, kb, pRow, e1, e2 - 1);
          if (b >= e2) dayRunLE(out, D, dayXF, dayR0, dayFyI, dw, dStrideEl, duSh, kb, pRow, e2, b);
        }
      }
    }
  }

  /* Terminator band: per-pixel sun weight (smoothstep on sin elevation
     across +/-6 deg) blending day bilinear with night nearest. */
  function bandRun(L, duD, duN, s2x, s2y, rowC, kb, pRow, a, b) {
    const D = dayTex.d32, dw = dayTex.w, dStrideEl = dayTex.stride;
    const N = nightTex.d32, nw = nightTex.w;
    const dayXF = L.dayXF, dayR0 = L.dayR0, dayFyI = L.dayFyI;
    const nightX = L.nightX, nightR = L.nightR, nyA = L.nyA;
    const out = L.out32;
    const x0 = L.x0, cxD = L.cxD, invR = L.invR;
    const duSh = duD << 9;
    for (let i = a; i <= b; i++) {
      const k = kb + i;
      const se = ((x0 + i + 0.5 - cxD) * invR) * s2x + nyA[k] * s2y + rowC;
      let w = 1;
      if (se <= -SIN6) w = 0;
      else if (se < SIN6) {
        const t = (se + SIN6) * INV_BAND;
        w = t * t * (3 - 2 * t);
      }
      let xn = nightX[k] + duN;
      if (xn >= nw) xn -= nw;
      const nv = N[nightR[k] + xn];
      if (w <= 0) {
        out[pRow + i] = nv;
        continue;
      }
      const tpk = dayXF[k] + duSh;
      let x = tpk >>> 9;
      if (x >= dw) x -= dw;
      const fx = tpk & 511, gx = 256 - fx;
      const o0 = dayR0[k] + x, o1 = o0 + dStrideEl;
      const c00 = D[o0], c01 = D[o0 + 1], c10 = D[o1], c11 = D[o1 + 1];
      const fy = dayFyI[k], gy = 256 - fy;
      const rb0 = (((c00 & 0xff00ff) * gx + (c01 & 0xff00ff) * fx) >>> 8) & 0xff00ff;
      const gg0 = (((c00 & 0xff00) * gx + (c01 & 0xff00) * fx) >>> 8) & 0xff00;
      const rb1 = (((c10 & 0xff00ff) * gx + (c11 & 0xff00ff) * fx) >>> 8) & 0xff00ff;
      const gg1 = (((c10 & 0xff00) * gx + (c11 & 0xff00) * fx) >>> 8) & 0xff00;
      const rb = (((rb0 * gy + rb1 * fy) >>> 8) & 0xff00ff);
      const gg = (((gg0 * gy + gg1 * fy) >>> 8) & 0xff00);
      if (w >= 1) {
        out[pRow + i] = (rb | gg) | 0xff000000;
        continue;
      }
      const wn = 1 - w;
      const r = (rb & 255) * w + (nv & 255) * wn;
      const g = ((gg >>> 8) & 255) * w + ((nv >>> 8) & 255) * wn;
      const bl = (rb >>> 16) * w + ((nv >>> 16) & 255) * wn;
      out[pRow + i] = (r | 0) | ((g | 0) << 8) | ((bl | 0) << 16) | 0xff000000;
    }
  }

  /* Byte-accurate scalar path for big-endian hosts (no packed-lane tricks). */
  function renderLutScalar(L, duD, duN, sunOk, s2x, s2y, s2z) {
    const day = dayTex.data, dw = dayTex.w, dStride4 = dayTex.stride << 2;
    const ngt = nightTex.data, nw = nightTex.w;
    const rowI0 = L.rowI0, rowN = L.rowN;
    const dayXF = L.dayXF, dayR0 = L.dayR0, dayFyI = L.dayFyI;
    const nightX = L.nightX, nightR = L.nightR;
    const nyA = L.nyA, nzA = L.nzA;
    const out = L.out32;
    const bw = L.bw, bh = L.bh, x0 = L.x0, cxD = L.cxD, invR = L.invR;
    const duSh = duD << 9;
    let k = 0;
    for (let j = 0; j < bh; j++) {
      let n = rowN[j];
      if (!n) continue;
      const i0 = rowI0[j];
      let p = j * bw + i0;
      let xs = (x0 + i0 + 0.5 - cxD) * invR;
      for (; n > 0; n--, k++, p++, xs += invR) {
        let w = 1;
        if (sunOk) {
          const se = xs * s2x + nyA[k] * s2y + nzA[k] * s2z;
          if (se <= -SIN6) w = 0;
          else if (se < SIN6) {
            const t = (se + SIN6) * INV_BAND;
            w = t * t * (3 - 2 * t);
          }
        }
        let r, g, b;
        if (w > 0) {
          const tpk = dayXF[k] + duSh;
          let x = tpk >>> 9;
          if (x >= dw) x -= dw;
          const fx = (tpk & 511) * (1 / 256), fy = dayFyI[k] * (1 / 256);
          const a0 = (dayR0[k] + x) << 2, a1 = a0 + dStride4;
          const r0 = day[a0], g0 = day[a0 + 1], b0 = day[a0 + 2];
          const tr0 = r0 + fx * (day[a0 + 4] - r0);
          const tg0 = g0 + fx * (day[a0 + 5] - g0);
          const tb0 = b0 + fx * (day[a0 + 6] - b0);
          const r1 = day[a1], g1 = day[a1 + 1], b1 = day[a1 + 2];
          const tr1 = r1 + fx * (day[a1 + 4] - r1);
          const tg1 = g1 + fx * (day[a1 + 5] - g1);
          const tb1 = b1 + fx * (day[a1 + 6] - b1);
          r = tr0 + fy * (tr1 - tr0);
          g = tg0 + fy * (tg1 - tg0);
          b = tb0 + fy * (tb1 - tb0);
          if (w < 1) {
            let xn = nightX[k] + duN;
            if (xn >= nw) xn -= nw;
            const an = (nightR[k] + xn) << 2;
            const wn = 1 - w;
            r = r * w + ngt[an] * wn;
            g = g * w + ngt[an + 1] * wn;
            b = b * w + ngt[an + 2] * wn;
          }
        } else {
          let xn = nightX[k] + duN;
          if (xn >= nw) xn -= nw;
          const an = (nightR[k] + xn) << 2;
          r = ngt[an]; g = ngt[an + 1]; b = ngt[an + 2];
        }
        out[p] = ((r | 0) << R_SH) | ((g | 0) << G_SH) | ((b | 0) << B_SH) | A_MASK;
      }
    }
  }

  /* --- public draw ----------------------------------------------------------- */

  /* Full-quality hemisphere render into the clipped disk bbox, then one
     drawImage onto ctx. Returns true when it handled the draw (even if the
     disk is fully off-canvas), false before the textures are ready or on
     unusable inputs. */
  function drawHemisphere(ctx, opts) {
    if (!isReady || !ctx) return false;
    const o = opts || {};
    /* destination device scale from the ctx transform (uniform scale assumed
       - the globe layers only ever carry the DPR scale) */
    let m = 1;
    if (typeof ctx.getTransform === 'function') {
      const t = ctx.getTransform();
      if (t && Number.isFinite(t.a) && t.a > 0) m = t.a;
    }
    const cxD = Number(o.cx) * m, cyD = Number(o.cy) * m;
    const rD = Number(o.radiusPx) * m;
    if (!Number.isFinite(cxD) || !Number.isFinite(cyD) || !(rD > 0)) return false;

    /* clip the disk bbox to the destination canvas (device px) */
    let cw = Infinity, ch = Infinity;
    const cv = ctx.canvas;
    if (cv && typeof cv.width === 'number' && cv.width > 0) {
      cw = cv.width;
      ch = cv.height;
    }
    let x0 = Math.floor(cxD - rD), y0 = Math.floor(cyD - rD);
    let x1 = Math.ceil(cxD + rD), y1 = Math.ceil(cyD + rD);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > cw) x1 = cw;
    if (y1 > ch) y1 = ch;
    if (x1 <= x0 || y1 <= y0) return true;           // fully off-canvas

    ensureScratch(x1 - x0, y1 - y0);

    /* LUT slot: exact match resamples; a same-geometry slot (tilt drag)
       rebuilds in place; otherwise the LRU slot is rebuilt. */
    const tilt = Number(o.tilt) || 0;
    let L = null, geo = null;
    for (const s of slots) {
      if (s.gen === texGen && s.x0 === x0 && s.y0 === y0 && s.x1 === x1
          && s.y1 === y1 && s.cxD === cxD && s.cyD === cyD && s.rD === rD) {
        if (s.tilt === tilt) { L = s; break; }
        geo = s;
      }
    }
    if (!L) {
      L = geo || (slots[0].stamp <= slots[1].stamp ? slots[0] : slots[1]);
      buildLut(L, x0, y0, x1, y1, cxD, cyD, rD, tilt);
    }
    L.stamp = ++stampCounter;

    /* sun in the view frame - the exact rotation the projection applies to
       ECI vectors (spin then tilt); dot(view normal, view sun) equals the
       ECEF elevation sine because rotations preserve dot products. */
    const spin = Number(o.spin) || 0;
    const G = spin + (Number(o.gmstRad) || 0);
    let sunOk = false, s2x = 0, s2y = 0, s2z = 0;
    const sv = o.sunDirTEME;
    if (sv && Number.isFinite(sv[0]) && Number.isFinite(sv[1]) && Number.isFinite(sv[2])) {
      const cT = Math.cos(tilt), sT = Math.sin(tilt);
      const cS = Math.cos(spin), sS = Math.sin(spin);
      const sx1 = sv[0] * cS - sv[1] * sS;
      const sy1 = sv[0] * sS + sv[1] * cS;
      s2x = sx1;
      s2y = sy1 * cT - sv[2] * sT;
      s2z = sy1 * sT + sv[2] * cT;
      sunOk = true;                                  // absent sun -> full day
    }

    /* ECEF lon = frame lon - G: one integer column offset per texture */
    let sh = -G * INV_2PI;
    sh -= Math.floor(sh);                            // frac in [0,1)
    let duD = (sh * dayTex.w + 0.5) | 0;
    if (duD >= dayTex.w) duD -= dayTex.w;
    let duN = (sh * nightTex.w + 0.5) | 0;
    if (duN >= nightTex.w) duN -= nightTex.w;

    const t0 = perfNow();
    if (IS_LE) renderLutPacked(L, duD, duN, sunOk, s2x, s2y, s2z);
    else renderLutScalar(L, duD, duN, sunOk, s2x, s2y, s2z);
    lastRenderMs = perfNow() - t0;

    scratchCtx.putImageData(L.img, 0, 0);
    /* dest coords are ctx-local (CSS px under the engine's DPR transform) */
    ctx.drawImage(scratch, 0, 0, L.bw, L.bh,
      x0 / m, y0 / m, (x1 - x0) / m, (y1 - y0) / m);
    return true;
  }

  /* --- debug / harness hooks ------------------------------------------------ */

  function _stats() {
    let lutBytes = 0, scratchBytes = 0;
    for (const s of slots) {
      if (s.rowI0) lutBytes += s.rowI0.byteLength + s.rowN.byteLength;
      if (s.dayXF) {
        lutBytes += s.dayXF.byteLength + s.dayR0.byteLength + s.dayFyI.byteLength
          + s.nightX.byteLength + s.nightR.byteLength
          + s.nyA.byteLength + s.nzA.byteLength;
      }
      if (s.img) scratchBytes += s.img.data.byteLength;
    }
    const texBytes = (dayTex ? dayTex.data.byteLength : 0)
      + (nightTex ? nightTex.data.byteLength : 0);
    return {
      lutBuilds, lastLutMs, lastRenderMs,
      lutBytes, texBytes, scratchBytes, vlutBytes: VLUT.byteLength,
      totalBytes: lutBytes + texBytes + scratchBytes + VLUT.byteLength,
    };
  }

  /* _treated: harness hook for design-rule assertions on the treated bytes. */
  const api = {
    load, ready, onReady, drawHemisphere,
    _loadImages,
    _treated: () => ({ day: dayTex, night: nightTex }),
    _stats,
  };
  return api;
})();
