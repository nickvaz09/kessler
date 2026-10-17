/* Kessler orbital plot - orthographic oblique view of orbit geometry.
   OrbitPlot.draw(canvas, { primary:{label, path}, secondary:{label, path}|null })
   path: [[x,y,z], ...] km TEME (one full period sampled from the TLE epoch,
   so path[0] is the position at epoch). Situational display, axis-free.
   depth-aware - the Earth is a solid disc, orbit arcs behind it are
   drawn faint and hidden where the planet covers them, so the orbit reads
   as a ring around a globe. Fixed oblique view (24° above the equator);
   drag (mouse, pen or touch) or the arrow keys rotate about the spin axis.
   DPR-aware; colors from --k-* tokens only, re-read on every draw, and the
   plot re-renders on KTheme.onChange and window resize. */
'use strict';

const OrbitPlot = (() => {
  const RE_KM = 6378;
  const live = new Set();

  function cssVar(name) {
    if (typeof KTheme !== 'undefined' && KTheme.css) return KTheme.css(name);
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function circlePts(r, plane, n = 120, zOff = 0) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2;
      if (plane === 'lat') pts.push([r * Math.cos(a), r * Math.sin(a), zOff]);
      else pts.push([r * Math.cos(a), 0, r * Math.sin(a)]); /* x-z meridian */
    }
    return pts;
  }

  function render(canvas) {
    const st = canvas.__kOrbit;
    if (!st) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return;
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width;
    const H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const sans = cssVar('--k-sans') || 'Inter, system-ui, sans-serif';
    const cx = W / 2;
    const cy = H / 2;
    const cosY = Math.cos(st.yaw);
    const sinY = Math.sin(st.yaw);
    const cosT = Math.cos(st.tilt);
    const sinT = Math.sin(st.tilt);

    const orbits = [];
    if (st.primary && Array.isArray(st.primary.path) && st.primary.path.length > 1) {
      orbits.push({ label: st.primary.label, path: st.primary.path, color: cssVar('--k-data-1') });
    }
    if (st.secondary && Array.isArray(st.secondary.path) && st.secondary.path.length > 1) {
      orbits.push({ label: st.secondary.label, path: st.secondary.path, color: cssVar('--k-data-2') });
    }

    let maxR = RE_KM * 1.15;
    for (const o of orbits) {
      for (const p of o.path) {
        const r = Math.hypot(p[0], p[1], p[2]);
        if (r > maxR) maxR = r;
      }
    }
    /* clamp so a tiny canvas never produces a negative arc radius */
    const scale = Math.max(Math.min(W, H) / 2 - 30, 8) / maxR;
    const rEarth = RE_KM * scale;

    /* yaw about z, then view from 24° above the equator plane.
       returns [screenX, screenY, depth (toward viewer), radialScreenDist] */
    const proj = (p) => {
      const x1 = p[0] * cosY + p[1] * sinY;
      const y1 = -p[0] * sinY + p[1] * cosY;
      const up = p[2] * cosT + y1 * sinT;
      const depth = -y1 * cosT + p[2] * sinT;
      return [cx + x1 * scale, cy - up * scale, depth, Math.hypot(x1, up) * scale];
    };
    const hidden = (s) => s[2] < 0 && s[3] < rEarth - 0.5;

    /* stroke a 3-D polyline in runs: `want(front)` picks which runs draw */
    const strokeRuns = (pts, color, width, alpha, want) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.globalAlpha = alpha;
      let open = false;
      ctx.beginPath();
      for (const p of pts) {
        const s = proj(p);
        const front = s[2] >= 0;
        const ok = want(front, s);
        if (ok) {
          if (!open) { ctx.moveTo(s[0], s[1]); open = true; } else ctx.lineTo(s[0], s[1]);
        } else {
          open = false;
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    /* 1. far side of each orbit (behind the planet plane), faint, and only
       where the disc does not cover it */
    for (const o of orbits) {
      strokeRuns(o.path, o.color, 1.5, 0.32, (front, s) => !front && !hidden(s));
    }

    /* 2. the planet: solid disc + limb, a few latitude lines on the near side */
    ctx.beginPath();
    ctx.arc(cx, cy, rEarth, 0, Math.PI * 2);
    ctx.fillStyle = cssVar('--k-surface-2');
    ctx.fill();
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, rEarth, 0, Math.PI * 2);
    ctx.clip();
    const cGrid = cssVar('--k-border');
    for (const lat of [-60, -30, 30, 60]) {
      const r = RE_KM * Math.cos((lat * Math.PI) / 180);
      const z = RE_KM * Math.sin((lat * Math.PI) / 180);
      strokeRuns(circlePts(r, 'lat', 120, z), cGrid, 1, 1, (front) => front);
    }
    for (let k = 0; k < 6; k++) {
      const a = (k / 6) * Math.PI;
      const mer = circlePts(RE_KM, 'meridian').map((p) => [
        p[0] * Math.cos(a), p[0] * Math.sin(a), p[2],
      ]);
      strokeRuns(mer, cGrid, 1, 1, (front) => front);
    }
    strokeRuns(circlePts(RE_KM, 'lat'), cssVar('--k-border-strong'), 1, 1, (front) => front);
    ctx.restore();
    ctx.beginPath();
    ctx.arc(cx, cy, rEarth, 0, Math.PI * 2);
    ctx.strokeStyle = cssVar('--k-border-strong');
    ctx.lineWidth = 1;
    ctx.stroke();

    /* empty state - keep the planet as context, say why it's bare */
    if (!orbits.length) {
      ctx.font = `450 13px ${sans}`;
      ctx.fillStyle = cssVar('--k-text-3');
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('No orbit data for this object', cx, cy - rEarth - 14);
    }

    /* 3. near side of each orbit, full strength */
    for (const o of orbits) {
      strokeRuns(o.path, o.color, 2, 1, (front) => front);
    }

    /* 4. epoch position dots + direct labels near apogee */
    const cSurface = cssVar('--k-surface');
    const cLabel = cssVar('--k-text-2');
    ctx.font = `550 12px ${sans}`;
    for (const o of orbits) {
      const s0 = proj(o.path[0]);
      const dim = hidden(s0);
      ctx.globalAlpha = dim ? 0.35 : 1;
      ctx.beginPath();
      ctx.arc(s0[0], s0[1], 6.5, 0, Math.PI * 2);
      ctx.fillStyle = cSurface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(s0[0], s0[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = o.color;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (o.label) {
        let ap = o.path[0];
        let apR = 0;
        for (const p of o.path) {
          const r = Math.hypot(p[0], p[1], p[2]);
          if (r > apR) { apR = r; ap = p; }
        }
        const sa = proj(ap);
        const text = String(o.label);
        const tw = ctx.measureText(text).width;
        const tx = Math.min(Math.max(sa[0] + 9, 6), W - tw - 6);
        const ty = Math.min(Math.max(sa[1] - 9, 16), H - 30);
        ctx.fillStyle = cLabel;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(text, tx, ty);
      }
    }

    /* 5. scale bar bottom-left: rule + distance */
    const targets = [500, 1000, 2000, 5000, 10000, 20000, 50000];
    let len = targets[0];
    for (const t of targets) if (t * scale <= 140) len = t;
    const px = len * scale;
    const x0 = 2.5;
    const y0 = H - 6.5;
    const cText3 = cssVar('--k-text-3');
    ctx.strokeStyle = cText3;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + px, y0);
    ctx.moveTo(x0, y0 - 4);
    ctx.lineTo(x0, y0);
    ctx.moveTo(x0 + px, y0 - 4);
    ctx.lineTo(x0 + px, y0);
    ctx.stroke();
    ctx.fillStyle = cText3;
    ctx.font = `450 11px ${sans}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`${len.toLocaleString('en-US')} km`, x0, y0 - 8);
  }

  function bind(canvas) {
    const st = canvas.__kOrbit;
    if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
    if (!canvas.hasAttribute('role')) canvas.setAttribute('role', 'img');
    canvas.style.touchAction = 'pan-y';
    canvas.addEventListener('pointerdown', (e) => {
      st.drag = { x0: e.clientX, yaw0: st.yaw, id: e.pointerId };
      try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* old engine */ }
      canvas.classList.add('dragging');
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!st.drag || st.drag.id !== e.pointerId) return;
      st.yaw = st.drag.yaw0 - (e.clientX - st.drag.x0) * 0.008;
      render(canvas);
    });
    const end = () => {
      if (!st.drag) return;
      st.drag = null;
      canvas.classList.remove('dragging');
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      st.yaw += (e.key === 'ArrowLeft' ? 1 : -1) * 0.12;
      render(canvas);
    });
    const onResize = () => {
      if (!canvas.isConnected) {
        window.removeEventListener('resize', onResize);
        live.delete(canvas);
        return;
      }
      render(canvas);
    };
    window.addEventListener('resize', onResize);
    live.add(canvas);
  }

  if (typeof KTheme !== 'undefined' && KTheme.onChange) {
    KTheme.onChange(() => {
      for (const c of Array.from(live)) {
        if (!c.isConnected) { live.delete(c); continue; }
        render(c);
      }
    });
  }

  return {
    draw(canvas, opts) {
      let st = canvas.__kOrbit;
      if (!st) {
        st = canvas.__kOrbit = { yaw: -0.6, tilt: (24 * Math.PI) / 180, drag: null };
        bind(canvas);
      }
      st.primary = (opts && opts.primary) || null;
      st.secondary = (opts && opts.secondary) || null;
      render(canvas);
    },
  };
})();
