"""Space weather: what the Sun is doing, and what it does to orbits.

Live conditions come from NOAA's Space Weather Prediction Center (public
JSON products, no key), cached for CACHE_S and served stale (flagged) when a
feed fails - a feed outage never breaks the page.

The physics chain Kessler shows:

  flare (X-rays, ~8 min)   -> radio blackouts on the sunlit side (R scale)
  CME / fast solar wind    -> geomagnetic storm (Kp, G scale): energy pours
  (hours to ~3 days)          into the auroral zones around both magnetic
                              poles (Joule heating + particle precipitation)
  heated thermosphere      -> the upper atmosphere swells: air density at a
                              given LEO altitude rises (NASA/NRL MSIS 2.1)
  more drag                -> orbits sink faster, and positions predicted
                              without the storm drift off along-track

Orbit effects are for a circular orbit with a constant ballistic
coefficient B = Cd*A/m:

  da/dt   = -rho * B * sqrt(mu * a)                 (decay rate)
  ds(t)  ~=  (3/4) * n * |d(da/dt)| * t^2           (along-track error after
                                                     t when the extra drag
                                                     is left out of the
                                                     prediction)

Storm drag on the real catalog (the risk layer): every TLE carries the drag
its object actually felt while the TLE was fitted, the first derivative of
mean motion (ndot). The prediction extrapolates that drag; the storm's
contribution is how far the air density at the object's altitude moves away
from what it was during the fit:

  R(Kp, h)      NRLMSIS density ratio vs a quiet day (Kp 1), per Kp level
  R_fit         mean R over the 36 h before the TLE epoch (the fit window)
  extra ndot(t) = (R(Kp(t), h) / R_fit - 1) * ndot
  drift(T)      = a * | integral_{t0}^{T} (T - t) * extra ndot(t) dt |

over the prediction's span (TLE epoch t0 to the time T asked about), with
Kp(t) NOAA's 3-hourly planetary Kp: observed for the past, NOAA's forecast
for the next three days, then held at the last forecast value. A TLE fitted
inside a steady storm already carries it (no extra); a storm that starts -
or ends - after the fit moves the object off its prediction. The drift is
reported as an extra 1-sigma along-track uncertainty (capped at 1,000 km,
where a public TLE says little anyway).

Access: peek() never waits on NOAA (stale-while-revalidate: a stale cache
starts one background refresh and the caller gets what is there). The status
line, the conjunction list and the object pages use it; only a cold
GET /spaceweather blocks on a fetch.

Sources: NOAA SWPC products (services.swpc.noaa.gov); Emmert et al. 2021,
NRLMSIS 2.0 (pymsis, MSIS 2.1); Shue et al. 1998 (magnetopause standoff);
the Bartels Kp -> ap table.
"""
from __future__ import annotations

import functools
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx

from backend.app import config

SWPC = "https://services.swpc.noaa.gov"
FEEDS = {
    "kp": "/products/noaa-planetary-k-index.json",
    "kp_forecast": "/products/noaa-planetary-k-index-forecast.json",
    "scales": "/products/noaa-scales.json",
    "xray": "/json/goes/primary/xrays-6-hour.json",
    "wind_speed": "/products/summary/solar-wind-speed.json",
    "wind_mag": "/products/summary/solar-wind-mag-field.json",
    "wind_plasma": "/json/rtsw/rtsw_wind_1m.json",
    "f107": "/json/f107_cm_flux.json",
    "aurora": "/json/ovation_aurora_latest.json",
}
CACHE_S = 300
RETRY_S = 60                    # after a failed fetch, wait before the next try
TIMEOUT_S = 10.0
MU = 3.986004418e14            # m^3/s^2
RE_KM = 6378.137
ALTS_KM = list(range(250, 901, 25))
QUIET_AP = 4                    # Kp ~1: a quiet day

# Bartels: Kp (in thirds) -> ap (nT)
_KP_AP = [(0, 0), (1 / 3, 2), (2 / 3, 3), (1, 4), (4 / 3, 5), (5 / 3, 6), (2, 7), (7 / 3, 9),
          (8 / 3, 12), (3, 15), (10 / 3, 18), (11 / 3, 22), (4, 27), (13 / 3, 32), (14 / 3, 39),
          (5, 48), (16 / 3, 56), (17 / 3, 67), (6, 80), (19 / 3, 94), (20 / 3, 111), (7, 132),
          (22 / 3, 154), (23 / 3, 179), (8, 207), (25 / 3, 236), (26 / 3, 300), (9, 400)]

_cache: dict = {"t": 0.0, "data": None}
_lock = threading.Lock()                 # guards _cache and _bg
_fetch_lock = threading.Lock()           # one NOAA fetch at a time
_bg = {"running": False, "fail_t": 0.0}
_raw_last: dict = {}                     # feed -> (fetched_at, body): the last good copy
RAW_KEEP_S = 2 * 3600                    # a failed feed falls back to a copy this fresh


# --- small physics -----------------------------------------------------------------

def kp_to_ap(kp: float) -> float:
    kp = max(0.0, min(9.0, float(kp)))
    for (k0, a0), (k1, a1) in zip(_KP_AP, _KP_AP[1:]):
        if kp <= k1:
            return a0 + (a1 - a0) * (kp - k0) / (k1 - k0)
    return 400.0


def flare_class(flux_wm2: float | None) -> str | None:
    """GOES 0.1-0.8 nm flux -> 'C1.5' style class."""
    if flux_wm2 is None or not flux_wm2 > 0:
        return None
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if flux_wm2 >= base:
            return f"{letter}{flux_wm2 / base:.1f}"
    return "A0.0"


def magnetopause_re(density_cc: float | None, speed_kms: float | None, bz_nt: float | None) -> dict:
    """Dayside magnetopause standoff (Shue et al. 1998) from the solar wind."""
    if not density_cc or not speed_kms:
        return {"pdyn_npa": None, "standoff_re": None}
    pdyn = 1.6726e-6 * density_cc * speed_kms ** 2
    bz = bz_nt if bz_nt is not None else 0.0
    r0 = (10.22 + 1.29 * math.tanh(0.184 * (bz + 8.14))) * pdyn ** (-1 / 6.6)
    return {"pdyn_npa": round(pdyn, 2), "standoff_re": round(r0, 2)}


def decay_rate_m_day(rho: float, b_m2kg: float, alt_km: float) -> float:
    a = (RE_KM + alt_km) * 1e3
    return rho * b_m2kg * math.sqrt(MU * a) * 86400.0


def along_track_error_m(extra_decay_m_day: float, alt_km: float, hours: float) -> float:
    a = (RE_KM + alt_km) * 1e3
    n = math.sqrt(MU / a ** 3)
    adot = abs(extra_decay_m_day) / 86400.0
    t = hours * 3600.0
    return 0.75 * n * adot * t * t


# --- the atmosphere ------------------------------------------------------------------

def densities(alts_km, f107: float, f107a: float, ap: float, when: datetime | None = None):
    """Orbit-averaged mass density (kg/m^3) per altitude, NRLMSIS 2.1 averaged
    over latitudes -60..60 and all longitudes at `when` (so day and night
    sides both count). None when the model is unavailable."""
    try:
        import numpy as np
        import pymsis
    except Exception:          # the page then says the model is unavailable
        return None
    when = when or datetime.now(timezone.utc)
    date = np.array([when.replace(tzinfo=None)], dtype="datetime64[m]")
    lats = np.arange(-60, 61, 15)
    lons = np.arange(0, 360, 30)
    out = pymsis.calculate(date, lons, lats, list(alts_km), f107s=[f107], f107as=[f107a],
                           aps=[[ap] * 7])
    rho = out[..., pymsis.Variable.MASS_DENSITY]          # (1, lon, lat, alt)
    w = np.cos(np.radians(lats))[None, None, :, None]     # area weights
    mean = (rho * w).sum(axis=(0, 1, 2)) / (w.sum() * len(lons))
    return [float(v) for v in mean]


def thermosphere(f107: float, f107a: float, ap_now: float, ap_peak: float | None = None) -> dict:
    alts = ALTS_KM
    now = densities(alts, f107, f107a, ap_now)
    quiet = densities(alts, f107, f107a, QUIET_AP)
    peak = densities(alts, f107, f107a, ap_peak) if ap_peak else None
    if now is None or quiet is None:
        return {"model": None, "alts_km": alts}
    ratio = lambda xs: [round(x / q, 3) for x, q in zip(xs, quiet)]  # noqa: E731
    return {
        "model": "NASA/NRL MSIS 2.1 (pymsis), orbit-averaged over latitudes ±60°",
        "alts_km": alts, "ap_now": round(ap_now, 1), "ap_quiet": QUIET_AP,
        "f107": f107, "f107a": f107a,
        "density_now": now, "density_quiet": quiet, "ratio_now": ratio(now),
        "ap_peak_forecast": round(ap_peak, 1) if ap_peak else None,
        "ratio_peak_forecast": ratio(peak) if peak else None,
    }


def orbit_effects(alt_km: float, b_m2kg: float, kp: float, f107: float, f107a: float) -> dict:
    alt_km = max(200.0, min(1000.0, float(alt_km)))
    ap = kp_to_ap(kp)
    now = densities([alt_km], f107, f107a, ap)
    quiet = densities([alt_km], f107, f107a, QUIET_AP)
    if now is None or quiet is None:
        return {"model": None}
    rho, rho_q = now[0], quiet[0]
    d_now = decay_rate_m_day(rho, b_m2kg, alt_km)
    d_q = decay_rate_m_day(rho_q, b_m2kg, alt_km)
    extra = d_now - d_q
    return {
        "model": "NASA/NRL MSIS 2.1 (pymsis)",
        "alt_km": alt_km, "b_m2kg": b_m2kg, "kp": kp, "ap": round(ap, 1),
        "f107": f107, "f107a": f107a,
        "density_kg_m3": rho, "density_quiet_kg_m3": rho_q,
        "density_ratio": round(rho / rho_q, 3),
        "decay_m_day": round(d_now, 2), "decay_quiet_m_day": round(d_q, 2),
        "along_track_error_m": {str(h): round(along_track_error_m(extra, alt_km, h), 1)
                                for h in (24, 72)},
    }


# --- NOAA feeds ----------------------------------------------------------------------

def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fetch_all() -> tuple[dict, dict]:
    """Every feed in parallel (worst case one TIMEOUT_S, not one per feed)."""
    raw, status = {}, {}
    with httpx.Client(base_url=SWPC, timeout=TIMEOUT_S, follow_redirects=True,
                      headers={"User-Agent": "Kessler space-weather view"}) as c:
        def get(item):
            k, path = item
            try:
                r = c.get(path)
                r.raise_for_status()
                return k, r.json(), "ok"
            except Exception as e:     # one dead feed must not take the others down
                return k, None, f"unavailable ({type(e).__name__})"
        with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
            for k, body, st in pool.map(get, FEEDS.items()):
                status[k] = st
                if body is not None:
                    raw[k] = body
    return raw, status


def _aurora_boundaries(av: dict) -> dict:
    """Equatorward edge of the auroral oval (probability >= 10%) per 2 degrees
    of longitude, each hemisphere, from the OVATION nowcast grid."""
    grid = av.get("coordinates") or []
    north, south = {}, {}
    for lon, lat, p in grid:
        if p is None or p < 10 or abs(lat) < 35:
            continue
        key = int(lon) // 2 * 2
        if lat > 0:
            north[key] = min(north.get(key, 90), lat)
        else:
            south[key] = max(south.get(key, -90), lat)
    nb = sorted(north.items())
    sb = sorted(south.items())
    mean = lambda xs: round(sum(v for _, v in xs) / len(xs), 1) if xs else None  # noqa: E731
    # the aurora itself, thinned for drawing: (lon, lat, probability) >= 5%
    pts = [[int(lon), int(lat), int(p)] for lon, lat, p in grid
           if p is not None and p >= 5 and abs(lat) >= 35 and int(lon) % 2 == 0]
    return {"observation": av.get("Observation Time"), "forecast": av.get("Forecast Time"),
            "north_edge": nb, "south_edge": sb,
            "north_mean_lat": mean(nb), "south_mean_lat": mean(sb), "points": pts}


def _build(raw: dict, status: dict) -> dict:
    out: dict = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "source": "NOAA Space Weather Prediction Center", "feeds": status}
    sc = raw.get("scales") or {}
    now = sc.get("0") or {}
    out["scales"] = {k: {"scale": _num((now.get(k) or {}).get("Scale")),
                         "text": (now.get(k) or {}).get("Text")} for k in ("R", "S", "G")}
    out["scales_outlook"] = [{"date": sc[d].get("DateStamp"),
                              "R_minor_prob": _num((sc[d].get("R") or {}).get("MinorProb")),
                              "R_major_prob": _num((sc[d].get("R") or {}).get("MajorProb")),
                              "S_prob": _num((sc[d].get("S") or {}).get("Prob")),
                              "G": _num((sc[d].get("G") or {}).get("Scale"))}
                             for d in ("1", "2", "3") if d in sc]
    xr = [r for r in (raw.get("xray") or []) if r.get("energy") == "0.1-0.8nm"]
    if xr:
        last = xr[-1]
        out["flare"] = {"flux_wm2": _num(last.get("flux")), "class": flare_class(_num(last.get("flux"))),
                        "time": last.get("time_tag")}
        step = max(1, len(xr) // 90)
        out["xray"] = [[r.get("time_tag"), _num(r.get("flux"))] for r in xr[::step]]
        peak = max(xr, key=lambda r: _num(r.get("flux")) or 0)
        out["flare_6h_peak"] = {"class": flare_class(_num(peak.get("flux"))), "time": peak.get("time_tag")}
    kp = raw.get("kp") or []
    obs = [(r.get("time_tag"), _num(r.get("Kp"))) for r in kp if isinstance(r, dict)]
    fc = raw.get("kp_forecast") or []
    fc_rows = [(r.get("time_tag"), _num(r.get("kp")), r.get("observed")) for r in fc if isinstance(r, dict)]
    obs = [(t, k) for t, k in obs if t and k is not None]
    if not obs:                               # the k-index feed failed: NOAA's forecast
        now_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")   # product
        obs = [(t, k) for t, k, kind in fc_rows   # carries observed/estimated rows too
               if t and k is not None and kind in ("observed", "estimated") and t[:19] <= now_tag]
    if obs:
        out["kp"] = {"now": obs[-1][1], "time": obs[-1][0], "recent": obs[-24:]}
    pred = [r for r in fc_rows if r[2] == "predicted"]
    out["kp_forecast"] = [[t, k] for t, k, _ in pred]
    out["kp_series"] = _kp_series(obs, fc_rows)
    out["kp_forecast_max"] = max((k for _, k, _ in pred if k is not None), default=None)
    ws = (raw.get("wind_speed") or [{}])[-1]
    wm = (raw.get("wind_mag") or [{}])[-1]
    plasma = [r for r in (raw.get("wind_plasma") or []) if r.get("active") and r.get("proton_density")]
    dens = _num(plasma[0].get("proton_density")) if plasma else None   # newest first in this feed
    speed = _num(ws.get("proton_speed"))
    bz = _num(wm.get("bz_gsm"))
    out["solar_wind"] = {"speed_kms": speed, "density_cc": dens, "bt_nt": _num(wm.get("bt")),
                         "bz_nt": bz, "time": ws.get("time_tag"),
                         **magnetopause_re(dens, speed, bz)}
    f = [r for r in (raw.get("f107") or []) if _num(r.get("flux")) and r.get("time_tag")]
    f107 = _num(f[0]["flux"]) if f else None
    # MSIS wants the 81-day mean: one value per day (the feed reports up to
    # three times a day), over the 81 days before the newest report
    daily: dict = {}
    for r in f:
        daily.setdefault(r["time_tag"][:10], _num(r["flux"]))
    days = sorted(daily)[-81:]
    f107a = round(sum(daily[d] for d in days) / len(days), 1) if days else None
    out["f107"] = {"now": f107, "mean": f107a, "time": f[0].get("time_tag") if f else None}
    if raw.get("aurora"):
        out["aurora"] = _aurora_boundaries(raw["aurora"])
    kp_now = (out.get("kp") or {}).get("now")
    if f107 and kp_now is not None:
        peak = out.get("kp_forecast_max")
        out["thermosphere"] = thermosphere(f107, f107a or f107, kp_to_ap(kp_now),
                                           kp_to_ap(peak) if peak is not None and peak > kp_now else None)
    if f107:
        out["storm"] = storm_table(f107, f107a or f107)
    return out


def _parse_tag(tag) -> datetime | None:
    """NOAA time tags ('2026-10-10T15:00:00', with or without Z) as UTC."""
    if not isinstance(tag, str) or len(tag) < 16:
        return None
    try:
        dt = datetime.fromisoformat(tag.replace("Z", "").replace(" ", "T")[:19])
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _kp_series(obs, fc_rows) -> list:
    """3-hourly planetary Kp as [iso, kp, kind]: observed (the k-index feed,
    then the forecast feed's own observed/estimated rows) followed by NOAA's
    prediction - one row per 3-hour interval start, oldest first."""
    rows: dict = {}
    for tag, kp in obs:
        t = _parse_tag(tag)
        if t is not None and kp is not None:
            rows[t] = [t, kp, "observed"]
    for tag, kp, kind in fc_rows:
        t = _parse_tag(tag)
        if t is not None and kp is not None and t not in rows:
            rows[t] = [t, kp, kind if kind in ("observed", "estimated", "predicted") else "predicted"]
    return [[t.strftime("%Y-%m-%dT%H:%M:%SZ"), kp, kind] for t, kp, kind in
            (rows[t] for t in sorted(rows))]


def _fresh(now: float) -> bool:
    return _cache["data"] is not None and now - _cache["t"] < CACHE_S


def conditions(force: bool = False) -> dict:
    """Current conditions, fetching when the cache is cold or stale (blocks
    on NOAA; GET /spaceweather uses it only on a cold cache). On a total feed
    failure: the last good payload flagged stale, else an error payload."""
    if not config.spaceweather_enabled():
        with _lock:
            if _cache["data"] is not None:
                return _cache["data"]
        return {"updated": None, "feeds": {}, "stale": True,
                "error": "live space weather is switched off on this server"}
    with _lock:
        if _fresh(time.time()) and not force:
            return _cache["data"]
    with _fetch_lock:
        with _lock:                  # a concurrent caller may have just refreshed
            if _fresh(time.time()) and not force:
                return _cache["data"]
        raw, status = _fetch_all()
        now = time.time()
        for k, st in list(status.items()):       # one flaky feed must not blank its panel
            if st == "ok":
                _raw_last[k] = (now, raw[k])
            elif k in _raw_last and now - _raw_last[k][0] < RAW_KEEP_S:
                raw[k] = _raw_last[k][1]
                status[k] = (f"{st}; kept the copy from "
                             f"{datetime.fromtimestamp(_raw_last[k][0], timezone.utc):%H:%M} UTC")
        if not any(v == "ok" for v in status.values()):
            with _lock:
                _bg["fail_t"] = time.time()
                if _cache["data"] is not None:
                    stale = dict(_cache["data"])
                    stale["stale"] = True
                    stale["feeds"] = status
                    return stale
            return {"updated": None, "feeds": status, "stale": True,
                    "error": "NOAA space weather feeds are unreachable"}
        data = _build(raw, status)
        with _lock:
            _cache.update(t=time.time(), data=data)
        return data


def _bg_refresh() -> None:
    try:
        conditions()
    except Exception as exc:          # never let the refresher die loudly
        with _lock:
            _bg["fail_t"] = time.time()
        print(f"[spaceweather] refresh failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    finally:
        with _lock:
            _bg["running"] = False


def peek() -> dict | None:
    """The cached conditions now, never waiting on NOAA. A cold or stale
    cache starts one background refresh (at most every RETRY_S after a
    failure); the caller gets what is there - None before the first fetch."""
    if not config.spaceweather_enabled():
        return _cache["data"]
    now = time.time()
    with _lock:
        data = _cache["data"]
        due = (not _fresh(now) and not _bg["running"]
               and now - _bg["fail_t"] >= RETRY_S)
        if due:
            _bg["running"] = True
    if due:
        threading.Thread(target=_bg_refresh, name="kessler-spaceweather",
                         daemon=True).start()
    return data


warm = peek            # app startup: start the first fetch in the background


def age_s(data: dict | None) -> float | None:
    """Seconds since `data` was built (None when it never was)."""
    t = _parse_tag((data or {}).get("updated"))
    return None if t is None else max(0.0, (datetime.now(timezone.utc) - t).total_seconds())


G_TEXT = {1: "minor", 2: "moderate", 3: "strong", 4: "severe", 5: "extreme"}


def g_from_kp(kp: float | None) -> int:
    """NOAA G scale from planetary Kp (G1 = Kp 5- and up ... G5 = Kp 9)."""
    if kp is None:
        return 0
    return max(0, min(5, int(math.floor(kp + 0.34)) - 4))


def _interp(x: float, xs, ys) -> float | None:
    if not xs or not ys:
        return None
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def summary(data: dict | None = None) -> dict | None:
    """The one-line view every page shows (header pill, Overview card, AI):
    storm level, the NOAA scales, the flare, the wind, and what the storm
    is doing to the air at two typical LEO altitudes. None when no data."""
    d = peek() if data is None else data
    if not d or not d.get("updated"):
        return None
    if (d.get("kp") or {}).get("now") is None:
        return None                        # no storm level: say nothing, never "quiet"
    sc = d.get("scales") or {}
    scale = lambda k: int((sc.get(k) or {}).get("scale") or 0)  # noqa: E731
    kp = (d.get("kp") or {}).get("now")
    g = max(g_from_kp(kp), 0)
    level = "storm" if g >= 1 else "active" if (kp or 0) >= 4 else "quiet"
    th = d.get("thermosphere") or {}
    ratio = lambda h: (round(_interp(h, th.get("alts_km"), th.get("ratio_now")), 2)  # noqa: E731
                       if th.get("ratio_now") else None)
    r_now = scale("R")
    flare = d.get("flare") or {}
    if g >= 1:
        label = f"G{g} {G_TEXT[g]} storm"
    elif r_now >= 1:
        label = f"R{r_now} radio blackout"
    elif level == "active":
        label = "Unsettled"
    else:
        label = "Quiet"
    age = age_s(d)
    wind = d.get("solar_wind") or {}
    aurora = d.get("aurora") or {}
    return {
        "level": level, "label": label, "g": g, "r": r_now, "s": scale("S"),
        "g_24h": scale("G"),
        "kp": kp, "kp_forecast_max": d.get("kp_forecast_max"),
        "flare": flare.get("class"),
        "flare_6h_peak": (d.get("flare_6h_peak") or {}).get("class"),
        "wind_kms": wind.get("speed_kms"), "standoff_re": wind.get("standoff_re"),
        "density_ratio": {"400": ratio(400), "550": ratio(550), "800": ratio(800)},
        "aurora_lat": aurora.get("north_mean_lat"),
        "updated": d.get("updated"), "age_s": None if age is None else round(age),
        "stale": bool(d.get("stale")) or (age is not None and age > 4 * CACHE_S),
    }


# --- storm drag on the real catalog ------------------------------------------------

KP_LEVELS = [round(i / 3, 4) for i in range(28)]      # 0, 0.33, ... 9 (Kp in thirds)
_REF_KP_INDEX = 3                                     # Kp 1 -> ap 4: the quiet day
_STEP_S = 3 * 3600.0
DRAG_ALT_MAX_KM = 1000.0                              # above this, drag is negligible


def storm_table(f107: float, f107a: float, when: datetime | None = None) -> dict | None:
    """Density ratio vs a quiet day (Kp 1) for every Kp level x altitude,
    NRLMSIS 2.1 orbit-averaged over latitudes -60..60 and all longitudes,
    driven by a daily Ap equal to each level's 3-hourly ap (sustained
    conditions). One vectorised model call; None when pymsis is missing."""
    try:
        import numpy as np
        import pymsis
    except Exception:
        return None
    when = (when or datetime.now(timezone.utc)).replace(tzinfo=None)
    n = len(KP_LEVELS)
    dates = np.array([when] * n, dtype="datetime64[m]")
    lats = np.arange(-60, 61, 20)
    lons = np.arange(0, 360, 45)
    out = pymsis.calculate(dates, lons, lats, ALTS_KM, f107s=[f107] * n, f107as=[f107a] * n,
                           aps=[[kp_to_ap(k)] * 7 for k in KP_LEVELS])
    rho = out[..., pymsis.Variable.MASS_DENSITY]                   # (kp, lon, lat, alt)
    w = np.cos(np.radians(lats))[None, None, :, None]
    mean = (rho * w).sum(axis=(1, 2)) / (w.sum() * len(lons))      # (kp, alt)
    ratio = mean / mean[_REF_KP_INDEX][None, :]
    return {"kp_levels": KP_LEVELS, "alts_km": ALTS_KM, "ratio": np.round(ratio, 4).tolist(),
            "f107": f107, "f107a": f107a, "ref_kp": KP_LEVELS[_REF_KP_INDEX]}


@functools.lru_cache(maxsize=16384)
def drag_profile(line1: str, line2: str) -> tuple | None:
    """(a_m, drag_alt_km, ndot_rad_s2, epoch_s, perigee_km, apogee_km) from a
    TLE, or None. ndot is the TLE's measured mean-motion derivative (python-
    sgp4 stores ndot/2 in rad/min^2). The drag altitude is the mean altitude,
    capped 60 km above perigee for eccentric orbits (drag acts at perigee)."""
    try:
        from sgp4.api import Satrec
        sat = Satrec.twoline2rv(line1, line2)
    except Exception:
        return None
    n = sat.no_kozai / 60.0                               # rad/s
    if not n > 0:
        return None
    a = (MU / (n * n)) ** (1.0 / 3.0)
    e = min(max(sat.ecco, 0.0), 0.99)
    perigee = a * (1 - e) / 1e3 - RE_KM
    apogee = a * (1 + e) / 1e3 - RE_KM
    alt = min((perigee + apogee) / 2.0, perigee + 60.0)
    ndot = 2.0 * sat.ndot / 3600.0                        # rad/min^2 (halved) -> rad/s^2
    epoch_s = (sat.jdsatepoch - 2440587.5 + sat.jdsatepochF) * 86400.0
    return (a, alt, ndot, epoch_s, perigee, apogee)


DRIFT_CAP_M = 1.0e6
FIT_WINDOW_S = 36 * 3600.0


class StormContext:
    """One conditions payload, ready for fast per-object drift sums: for
    every altitude column, prefix sums of R(t) and t*R(t) over the 3-hourly
    Kp record, so each drift is a handful of lookups (O(log n))."""

    def __init__(self, data: dict):
        import bisect
        self._bisect = bisect.bisect_right
        st = data["storm"]
        self.updated = data.get("updated")
        self.kp_now = (data.get("kp") or {}).get("now")
        self.alts = list(st["alts_km"])
        self.table = [list(row) for row in st["ratio"]]  # [kp][alt]
        self.levels = list(st["kp_levels"])
        iv = []
        for tag, kp, kind in data.get("kp_series") or []:
            t = _parse_tag(tag)
            if t is not None and kp is not None:
                iv.append((t.timestamp(), float(kp)))
        iv.sort()
        if not iv:
            raise ValueError("no Kp record")
        self.t_ref = iv[0][0]
        starts = [t - self.t_ref for t, _ in iv]
        ends = starts[1:] + [starts[-1] + _STEP_S]
        self.starts, self.ends = starts, ends
        self.kps = [kp for _, kp in iv]
        nalt = len(self.alts)
        self.R, self.A, self.B = [], [], []        # per altitude column
        for j in range(nalt):
            col = [row[j] for row in self.table]
            r = [self._kp_interp(col, kp) for kp in self.kps]
            a_cum, b_cum = [0.0], [0.0]
            for k, rk in enumerate(r):
                s0, s1 = starts[k], ends[k]
                a_cum.append(a_cum[-1] + rk * (s1 - s0))
                b_cum.append(b_cum[-1] + rk * (s1 * s1 - s0 * s0) * 0.5)
            self.R.append(r)
            self.A.append(a_cum)
            self.B.append(b_cum)

    @staticmethod
    def _kp_interp(col, kp):
        x = max(0.0, min(9.0, kp)) * 3.0
        i = min(int(x), len(col) - 2)
        return col[i] + (col[i + 1] - col[i]) * (x - i)

    def _alt(self, alt_km):
        alts = self.alts
        h = max(alts[0], min(alts[-1], float(alt_km)))
        j = min(int((h - alts[0]) // (alts[1] - alts[0])), len(alts) - 2)
        return j, (h - alts[j]) / (alts[j + 1] - alts[j])

    def ratio(self, kp: float, alt_km: float) -> float:
        """Density ratio vs a quiet day at one Kp and altitude."""
        j, f = self._alt(alt_km)
        lo = self._kp_interp([row[j] for row in self.table], kp)
        hi = self._kp_interp([row[j + 1] for row in self.table], kp)
        return lo + (hi - lo) * f

    def ratio_at(self, t_s: float, alt_km: float) -> float:
        """R at an absolute time (the Kp record's value then)."""
        j, f = self._alt(alt_km)
        k = max(0, min(len(self.starts) - 1, self._bisect(self.starts, t_s - self.t_ref) - 1))
        return self.R[j][k] * (1 - f) + self.R[j + 1][k] * f

    def _cum(self, j: int, t: float) -> tuple:
        """(integral R dt, integral t*R dt) from the record start to t
        (relative seconds); the first value extends back, the last forward."""
        st, en, R, A, B = self.starts, self.ends, self.R[j], self.A[j], self.B[j]
        if t <= st[0]:
            r0 = R[0]
            return -r0 * (st[0] - t), -r0 * (st[0] * st[0] - t * t) * 0.5
        if t >= en[-1]:
            rl = R[-1]
            return A[-1] + rl * (t - en[-1]), B[-1] + rl * (t * t - en[-1] * en[-1]) * 0.5
        k = self._bisect(st, t) - 1
        rk = R[k]
        return A[k] + rk * (t - st[k]), B[k] + rk * (t * t - st[k] * st[k]) * 0.5

    def _terms(self, j: int, t0: float, t1: float, e: float) -> tuple:
        a0, b0 = self._cum(j, t0)
        a1, b1 = self._cum(j, t1)
        f0, _ = self._cum(j, e - FIT_WINDOW_S)
        f1, _ = self._cum(j, e)
        return a1 - a0, b1 - b0, (f1 - f0) / FIT_WINDOW_S

    def fit_ratio(self, prof: tuple) -> float:
        """Mean R over the 36 h before the TLE epoch (the drag ndot holds)."""
        j, f = self._alt(prof[1])
        e = prof[3] - self.t_ref
        lo = self._terms(j, e, e, e)[2]
        hi = self._terms(j + 1, e, e, e)[2]
        return max(1e-6, lo + (hi - lo) * f)

    def drift_m(self, prof: tuple | None, t0_s: float, t1_s: float) -> float:
        """Storm along-track drift [m] at t1 for a prediction started at t0
        (see the module docstring), capped at DRIFT_CAP_M."""
        if prof is None or t1_s <= t0_s:
            return 0.0
        a, alt, ndot, epoch_s = prof[0], prof[1], prof[2], prof[3]
        if not ndot > 0 or alt > DRAG_ALT_MAX_KM:
            return 0.0
        j, f = self._alt(alt)
        t0, t1, e = t0_s - self.t_ref, t1_s - self.t_ref, epoch_s - self.t_ref
        i0a, i1a, fa = self._terms(j, t0, t1, e)
        i0b, i1b, fb = self._terms(j + 1, t0, t1, e)
        i0, i1, rfit = i0a + (i0b - i0a) * f, i1a + (i1b - i1a) * f, max(1e-6, fa + (fb - fa) * f)
        span = t1 - t0
        integral = (t1 * i0 - i1) / rfit - 0.5 * span * span   # int (T - t)(R/R_fit - 1) dt
        return min(DRIFT_CAP_M, abs(a * ndot * integral))


_ctx_cache: dict = {"key": None, "ctx": None}


def storm_context(data: dict | None = None) -> StormContext | None:
    """The StormContext for the current (peeked) payload, or None when there
    is no live data or no atmosphere model."""
    d = peek() if data is None else data
    if not d or not d.get("storm") or not d.get("kp_series"):
        return None
    key = (id(d), d.get("updated"))
    if _ctx_cache["key"] != key:
        try:
            _ctx_cache.update(key=key, ctx=StormContext(d))
        except Exception:
            return None
    return _ctx_cache["ctx"]


def object_effects(line1: str | None, line2: str | None, ctx: StormContext | None = None,
                   now: datetime | None = None) -> dict | None:
    """What the current storm does to one object, from its own TLE: decay
    now vs a quiet day, and how far its TLE prediction drifts from storm
    drag by now, in 24 h and in 72 h (forecast Kp). None without data or a
    usable TLE."""
    ctx = ctx or storm_context()
    if ctx is None or not line1 or not line2:
        return None
    prof = drag_profile(line1, line2)
    if prof is None:
        return None
    a, alt, ndot, epoch_s, perigee, apogee = prof
    now_s = (now or datetime.now(timezone.utc)).timestamp()
    applies = alt <= DRAG_ALT_MAX_KM and ndot > 0
    n = math.sqrt(MU / a ** 3)
    kp = ctx.kp_now if ctx.kp_now is not None else 0.0
    r_now = ctx.ratio(kp, alt) if applies else 1.0
    r_fit = ctx.fit_ratio(prof) if applies else 1.0
    # the TLE's measured decay holds the fit window's air; scale to a quiet
    # day (R = 1) and to now
    fitted = (2.0 / 3.0) * (a / n) * ndot * 86400.0 if ndot > 0 else 0.0   # m/day
    quiet = fitted / r_fit
    return {
        "applies": bool(applies),
        "drag_alt_km": round(alt, 1), "perigee_km": round(perigee, 1), "apogee_km": round(apogee, 1),
        "kp": ctx.kp_now, "density_ratio": round(r_now, 2), "fit_density_ratio": round(r_fit, 2),
        "decay_quiet_m_day": round(quiet, 1) if applies else None,
        "decay_now_m_day": round(quiet * r_now, 1) if applies else None,
        "drift_24h_m": round(ctx.drift_m(prof, epoch_s, now_s + 86400.0)) if applies else 0,
        "drift_72h_m": round(ctx.drift_m(prof, epoch_s, now_s + 3 * 86400.0)) if applies else 0,
        "drift_since_epoch_m": round(ctx.drift_m(prof, epoch_s, now_s)) if applies else 0,
        "tle_age_h": round((now_s - epoch_s) / 3600.0, 1),
        "updated": ctx.updated,
    }


def pair_drift_m(line_p: tuple | None, line_s: tuple | None, tca: datetime,
                 ctx: StormContext) -> float:
    """Combined (RSS) storm drift of two objects at TCA, each over its own
    prediction span (TLE epoch to TCA) - the same span the assumed sigma
    uses."""
    t1 = tca.timestamp()
    total = 0.0
    for lines in (line_p, line_s):
        if not lines:
            continue
        prof = drag_profile(*lines)
        if prof is not None:
            total += ctx.drift_m(prof, prof[3], t1) ** 2
    return math.sqrt(total)
