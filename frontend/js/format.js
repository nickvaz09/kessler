/* Shared formatting utilities. All numeric display goes through these so the
   whole UI renders values identically. Risk levels come from the API - the
   frontend never derives them from Pc thresholds.

   Humane output: numbers stay precise but read like a person wrote
   them - "412 m", "1.24 km", "10.8 km/s", countdowns as "3 h 12 min",
   ages as "15 min" / "6.3 h" / "3 days". Function names and signatures are
   stable API; the helpers after `esc` are newer additions. */
'use strict';

const FMT = (() => {
  const DASH = '—';
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  function missing(v) {
    return v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
  }

  /* Collision probability, scientific: "9.5e-5". Below 1e-10 is shown as a
     floor, never as zero. */
  function pc(p) {
    if (missing(p)) return DASH;
    if (p <= 0 || p < 1e-10) return '<1e-10';  // negligible: below display floor
    let exp = Math.floor(Math.log10(p));
    let mant = p / Math.pow(10, exp);
    if (mant >= 9.95) { mant /= 10; exp += 1; }  // 9.96e-6 & fp edge -> 1.0e-5
    return `${mant.toFixed(1)}e${exp >= 0 ? '+' : ''}${exp}`;
  }

  /* distance in metres -> "412 m", "1.24 km", "12.3 km", "1,204 km" */
  function dist(m) {
    if (missing(m)) return DASH;
    const v = Number(m);
    if (Math.abs(v) < 1000) return `${Math.round(v)} m`;
    const k = v / 1000;
    const a = Math.abs(k);
    const digits = a < 10 ? 2 : a < 100 ? 1 : 0;
    return `${k.toLocaleString('en-US', {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    })} km`;
  }

  function km(v, digits = 1) {
    if (missing(v)) return DASH;
    return `${num(Number(v).toFixed(digits))} km`;
  }

  /* relative velocity m/s -> "10.8 km/s" (slow approaches keep m/s) */
  function vel(ms) {
    if (missing(ms)) return DASH;
    const v = Number(ms);
    if (Math.abs(v) < 100) return `${Math.round(v)} m/s`;
    return `${(v / 1000).toFixed(1)} km/s`;
  }

  function num(n) {
    if (missing(n)) return DASH;
    return Number(n).toLocaleString('en-US');
  }

  function deg(v, digits = 2) {
    if (missing(v)) return DASH;
    return `${Number(v).toFixed(digits)}°`;
  }

  /* "2026-07-21T14:32:05Z" -> "2026-07-21 14:32:05Z" (precise, machine form;
     see `when` for the friendly form) */
  function utc(iso) {
    if (!iso) return DASH;
    return String(iso).replace('T', ' ').replace(/\.\d+Z?$/, 'Z');
  }

  /* span in ms -> "2 d 4 h", "3 h 12 min", "42 min 05 s", "18 s" */
  function span(ms) {
    ms = Math.abs(ms);
    const d = Math.floor(ms / 86400000);
    const h = Math.floor((ms % 86400000) / 3600000);
    const m = Math.floor((ms % 3600000) / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    /* one ladder for every relative time in the app: 1 d 15 h · 2 h 8 min */
    if (ms >= 86400000) return h ? `${d} d ${h} h` : `${d} d`;
    if (ms >= 3600000) return m ? `${Math.floor(ms / 3600000)} h ${m} min` : `${Math.floor(ms / 3600000)} h`;
    if (ms >= 60000) return `${m} min ${String(s).padStart(2, '0')} s`;
    return `${s} s`;
  }

  /* Countdown to a UTC instant: "3 h 12 min" before the event,
     "3 h 12 min ago" after it. */
  function countdown(iso) {
    if (!iso) return DASH;
    const t = new Date(iso).getTime();
    if (!Number.isFinite(t)) return DASH;
    const ms = t - Date.now();
    return ms < 0 ? `${span(ms)} ago` : span(ms);
  }

  /* Age of an instant: "15 min", "6.3 h", "14 h", "3 days". */
  function ageHours(iso) {
    if (!iso) return DASH;
    const t = new Date(iso).getTime();
    if (!Number.isFinite(t)) return DASH;
    const h = Math.max(0, (Date.now() - t) / 3600000);
    if (h < 1) return `${Math.round(h * 60)} min`;
    if (h < 10) return `${h.toFixed(1)} h`;
    if (h < 48) return `${Math.round(h)} h`;
    const d = h / 24;
    return `${d < 10 ? d.toFixed(1) : Math.round(d)} days`;
  }

  function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  /* Ticking UTC clock, "13:37 UTC". Returns the interval id. */
  function startUtcClock(el) {
    if (!el) return null;
    function tick() {
      el.textContent = `${new Date().toISOString().slice(11, 16)} UTC`;
    }
    tick();
    return setInterval(tick, 1000);
  }

  /* --- humane formatters -------------------------------------------------- */

  /* Friendly UTC instant: "Thu 24 Sep, 04:45 UTC" (opts.seconds adds :ss,
     opts.year adds the year, opts.day=false drops the weekday). */
  function when(iso, opts) {
    if (!iso) return DASH;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return DASH;
    const o = opts || {};
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mm = String(d.getUTCMinutes()).padStart(2, '0');
    const ss = o.seconds ? `:${String(d.getUTCSeconds()).padStart(2, '0')}` : '';
    const day = o.day === false ? '' : `${DAYS[d.getUTCDay()]} `;
    const yr = o.year ? ` ${d.getUTCFullYear()}` : '';
    return `${day}${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}${yr}, ${hh}:${mm}${ss} UTC`;
  }

  /* Relative phrase: "in 3 h 12 min", "12 min ago", "just now". */
  function rel(iso) {
    if (!iso) return DASH;
    const t = new Date(iso).getTime();
    if (!Number.isFinite(t)) return DASH;
    const ms = t - Date.now();
    const a = Math.abs(ms);
    if (a < 45000) return 'just now';
    /* under an hour, whole minutes read calmer than a ticking seconds field */
    const txt = a < 3600000 ? `${Math.max(1, Math.round(a / 60000))} min` : span(a);
    return ms > 0 ? `in ${txt}` : `${txt} ago`;
  }

  /* probability as odds for plain-language secondary text: "1 in 10,500" */
  function pcOdds(p) {
    if (missing(p) || !(p > 0)) return DASH;
    if (p >= 0.5) return 'better than even';
    const n = 1 / p;
    if (n > 1e9) return 'less than 1 in a billion';
    const mag = Math.pow(10, Math.max(0, Math.floor(Math.log10(n)) - 1));
    return `1 in ${num(Math.round(n / mag) * mag)}`;
  }

  /* Duration in seconds -> "6 min 20 s". */
  function dur(s) {
    if (missing(s) || !Number.isFinite(Number(s))) return DASH;
    return span(Number(s) * 1000);
  }

  return {
    pc, dist, km, vel, num, deg, utc, countdown, ageHours, esc, startUtcClock,
    when, rel, pcOdds, dur, span,
  };
})();
