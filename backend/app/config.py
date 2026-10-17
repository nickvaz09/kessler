"""Central configuration. Paths, data sources, screening and risk parameters.

Values here are the single source of truth - modules must import from config,
never hardcode thresholds or URLs.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "kessler.db"
MODEL_DIR = DATA_DIR / "models"
KELVINS_DIR = DATA_DIR / "kelvins"
FRONTEND_DIR = ROOT / "frontend"

# --- CelesTrak ---------------------------------------------------------------
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
# Groups ingested into the catalog. Primaries are the operator-relevant objects
# we screen conjunctions for; every catalog object is a screening secondary.
PRIMARY_GROUPS = ["stations", "cubesat"]
CATALOG_GROUPS = [
    "active",          # every operational payload (~11.5k, incl. Starlink).
                       # CelesTrak 403s this group for scripted clients as of
                       # 2026-08; fetch_catalog skips it and the Space-Track
                       # full catalog (KESSLER_FULL_CATALOG=1) covers actives.
    "stations",
    "cubesat",
    "cosmos-2251-debris",
    "iridium-33-debris",
    "fengyun-1c-debris",
    # 'last-30-days' removed 2026-08: CelesTrak dropped the group (hard 404)
]
HTTP_TIMEOUT_S = 60.0

# --- Conjunction screening ---------------------------------------------------
SCREENING_WINDOW_HOURS = 72        # look-ahead window from "now"
SCREENING_COARSE_STEP_S = 60.0     # coarse propagation step
SCREENING_THRESHOLD_KM = 15.0      # report close approaches under this range
SCREENING_MAX_LEO_ALT_KM = 3000.0  # skip non-LEO secondaries (apogee above this)
# Pairs closing slower than this are co-moving, not conjuncting: docked station
# modules, ejecta riding the same TLE state, formation flyers. Real screening
# excludes them - a hazard needs an actual encounter velocity.
SCREENING_MIN_REL_VEL_MS = 10.0

# --- Collision probability / covariance assumptions --------------------------
# TLE-derived states carry no covariance. We assume a TLE-class positional
# uncertainty that grows with propagation age; documented in models/pc.py.
HARD_BODY_RADIUS_M = 10.0
SIGMA_BASE_M = 300.0        # 1-sigma cross-track/radial position error at epoch
SIGMA_GROWTH_M_PER_DAY = 400.0  # linear growth of 1-sigma with propagation age

# --- Risk levels -------------------------------------------------------------
# Operational thresholds (NASA CARA-style): Pc >= 1e-4 is maneuver-consideration
# territory; 1e-6..1e-4 warrants monitoring.
PC_CRITICAL = 1e-4
PC_ELEVATED = 1e-6

RISK_MODEL_FILE = MODEL_DIR / "risk_model.json"
RISK_MODEL_META_FILE = MODEL_DIR / "risk_model.meta.json"
HEURISTIC_MODEL_VERSION = "heuristic-0.1"

# --- Automation switches -----------------------------------------------------
# Read from the environment AT CALL TIME (not import time) so the scheduler
# and the pipeline honor the environment of the process they actually run in - 
# tests set/unset these per case via monkeypatch.


def refresh_interval_h() -> float | None:
    """KESSLER_REFRESH_INTERVAL_H as float hours; None = auto-refresh off.

    Unset, empty, non-numeric or <= 0 all read as "off" - a malformed value
    must never crash app startup.
    """
    raw = os.environ.get("KESSLER_REFRESH_INTERVAL_H", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0.0 else None


def full_catalog_enabled() -> bool:
    """KESSLER_FULL_CATALOG truthiness ('1'/'true'/'yes'/'on', case-insens)."""
    raw = os.environ.get("KESSLER_FULL_CATALOG", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def spaceweather_enabled() -> bool:
    """KESSLER_SPACEWEATHER: live NOAA space-weather fetches on (default) or
    off ('0'/'false'/'no'/'off'). Off, every space-weather surface degrades
    to "unavailable" exactly as during a NOAA outage - the test suite runs
    this way so it never touches the network."""
    raw = os.environ.get("KESSLER_SPACEWEATHER", "").strip().lower()
    return raw not in ("0", "false", "no", "off")
