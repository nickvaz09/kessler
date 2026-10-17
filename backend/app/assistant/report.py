"""Standalone HTML report renderer for the Kessler AI create_report tool.

Reports combine real database tables for the subject (queried through the
same helpers the REST layer uses) with model-authored narrative sections.
Narrative is escaped first, then a minimal markdown subset is applied:
paragraphs, **bold**, `code`, and "- " lists - nothing else.

Files land in data/reports/<uuid>.html and are served by
GET /reports/{id} (api/assistant.py). Pages are fully self-contained:
inline Kessler dark tokens, @media print light theme, a PRINT / SAVE AS PDF
button, and no external requests of any kind.
"""
import html
import re
import uuid
from datetime import datetime, timezone

from backend.app import config
from backend.app.api import conjunctions as conjunctions_api
from backend.app.api import objects as objects_api
from backend.app.api import status as status_api

_e = html.escape

_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")

# Kessler dark tokens (mirrors frontend/css/main.css) + print light theme.
# Hairlines, 2px radii, no gradients/shadows; amber/red reserved for risk.
_CSS = """
:root {
  --k-bg:#0a0d11; --k-surface:#0e1319; --k-border:#1e2733;
  --k-border-strong:#2a3646; --k-text:#dbe2ea; --k-text-2:#93a1b0;
  --k-text-3:#5c6a78; --k-elevated:#e0a63c; --k-critical:#e05252;
  --k-mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,monospace;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--k-bg); color:var(--k-text); font-family:var(--k-mono);
  font-size:13px; line-height:1.55; max-width:960px; margin:0 auto; padding:32px; }
.brand { font-size:11px; letter-spacing:0.12em; color:var(--k-text-3); }
h1 { font-size:18px; letter-spacing:0.03em; margin:4px 0 2px; }
h2 { font-size:11px; letter-spacing:0.08em; text-transform:uppercase;
  color:var(--k-text-2); border-bottom:1px solid var(--k-border);
  padding-bottom:6px; margin:28px 0 10px; }
.meta { color:var(--k-text-3); font-size:11px; }
header .meta { margin-bottom:8px; }
footer.meta { margin-top:32px; border-top:1px solid var(--k-border); padding-top:10px; }
table { width:100%; border-collapse:collapse; margin:10px 0 4px; font-size:12px; }
th { text-align:left; color:var(--k-text-3); font-weight:500; font-size:10px;
  letter-spacing:0.06em; border-bottom:1px solid var(--k-border-strong);
  padding:6px 10px 6px 0; }
td { border-bottom:1px solid var(--k-border); padding:7px 10px 7px 0;
  vertical-align:top; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
table.kv td:first-child { color:var(--k-text-2); width:230px; }
.chip { display:inline-block; border:1px solid var(--k-border-strong);
  border-radius:2px; padding:1px 6px; font-size:10px; letter-spacing:0.05em;
  color:var(--k-text-2); }
.chip--elevated { color:var(--k-elevated); border-color:rgba(224,166,60,0.45); }
.chip--critical { color:var(--k-critical); border-color:rgba(224,82,82,0.45); }
.narrative p { margin:0 0 10px; }
.narrative ul { margin:0 0 10px 18px; }
.narrative li { margin:2px 0; }
.narrative code { background:var(--k-surface); border:1px solid var(--k-border);
  border-radius:2px; padding:0 4px; font-size:12px; }
.print-btn { position:fixed; top:16px; right:16px; background:var(--k-surface);
  color:var(--k-text); border:1px solid var(--k-border-strong); border-radius:2px;
  font:600 10.5px/1 var(--k-mono); letter-spacing:0.06em; padding:8px 12px;
  cursor:pointer; }
.print-btn:hover { border-color:var(--k-text-3); }
@media print {
  :root { --k-bg:#ffffff; --k-surface:#f2f3f5; --k-border:#d5d9de;
    --k-border-strong:#b8bec6; --k-text:#16191d; --k-text-2:#454c55;
    --k-text-3:#6a727c; --k-elevated:#8a6210; --k-critical:#a32626; }
  body { padding:0; }
  .print-btn { display:none; }
}
"""


def render_markdown(text: str) -> str:
    """Escape-first minimal markdown: paragraphs, **bold**, `code`, - lists."""
    esc = _e(str(text))
    esc = _CODE_RE.sub(r"<code>\1</code>", esc)
    esc = _BOLD_RE.sub(r"<strong>\1</strong>", esc)
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    def flush_list():
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for line in esc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            flush_para()
            items.append(stripped[2:].strip())
        elif not stripped:
            flush_para()
            flush_list()
        else:
            flush_list()
            para.append(stripped)
    flush_para()
    flush_list()
    return "\n".join(out)


# --- formatting helpers ------------------------------------------------------

def _fmt(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _fmt_pc(value) -> str:
    return "—" if value is None else f"{value:.2e}"


def _chip(level) -> str:
    level = level or "nominal"
    return f'<span class="chip chip--{_e(level)}">{_e(level.upper())}</span>'


def _kv_table(rows: list[tuple[str, str]]) -> str:
    """Key/value table. Values are pre-rendered HTML; keys are escaped here."""
    body = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


# --- subject data sections (REAL DB tables) ----------------------------------

def _as_int(value, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer for this subject type")


def _cdm_table(cdms) -> str:
    if not cdms:
        return '<p class="meta">NO CDM MESSAGES RECORDED.</p>'
    rows = "".join(
        f"<tr><td>{_e(c['message_time'])}</td><td>{_e(c['source'])}</td>"
        f"<td class='num'>{_fmt(c['miss_distance_m'])}</td>"
        f"<td class='num'>{_fmt(c['relative_velocity_ms'])}</td>"
        f"<td class='num'>{_fmt_pc(c['probability_of_collision'])}</td></tr>"
        for c in cdms
    )
    return (
        "<table><thead><tr><th>MESSAGE TIME (UTC)</th><th>SOURCE</th>"
        "<th class='num'>MISS DIST (M)</th><th class='num'>REL VEL (M/S)</th>"
        "<th class='num'>PC</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _prediction_table(preds) -> str:
    if not preds:
        return '<p class="meta">NO RISK PREDICTIONS RECORDED.</p>'
    rows = "".join(
        f"<tr><td>{_e(p['prediction_time'])}</td><td>{_e(p['model_version'])}</td>"
        f"<td class='num'>{_fmt_pc(p['predicted_final_risk'])}</td></tr>"
        for p in preds
    )
    return (
        "<table><thead><tr><th>PREDICTION TIME (UTC)</th><th>MODEL</th>"
        "<th class='num'>PREDICTED FINAL RISK</th></tr></thead><tbody>"
        + rows + "</tbody></table>"
    )


def _side_label(side: dict) -> str:
    return f"{_e(side['name'])} (NORAD {side['norad_id']}, {_e(side['object_type'])})"


def _conjunction_data(conn, cid: int) -> tuple[str, str]:
    items = conjunctions_api.query_summaries(conn, "c.id = ?", (cid,))
    if not items:
        raise ValueError(f"Conjunction {cid} not found")
    item = items[0]
    cdms = [
        dict(row)
        for row in conn.execute(
            "SELECT message_time, miss_distance_m, relative_velocity_ms,"
            "       probability_of_collision, source"
            "  FROM cdm_messages WHERE conjunction_id = ?"
            " ORDER BY message_time ASC, id ASC",
            (cid,),
        )
    ]
    preds = [
        dict(row)
        for row in conn.execute(
            "SELECT predicted_final_risk, prediction_time, model_version"
            "  FROM risk_predictions WHERE conjunction_id = ?"
            " ORDER BY prediction_time ASC, id ASC",
            (cid,),
        )
    ]
    facts = _kv_table([
        ("PRIMARY", _side_label(item["primary"])),
        ("SECONDARY", _side_label(item["secondary"])),
        ("TCA (UTC)", _e(item["tca"])),
        ("STATUS", _e(item["status"].upper())),
        ("RISK LEVEL", _chip(item["risk_level"])),
        ("MISS DISTANCE (M)", _fmt(item["miss_distance_m"])),
        ("RELATIVE VELOCITY (M/S)", _fmt(item["relative_velocity_ms"])),
        ("PC (LATEST CDM)", _fmt_pc(item["pc_latest"])),
        ("PREDICTED FINAL RISK", _fmt_pc(item["predicted_final_risk"])
         + (f" · {_e(item['model_version'])}" if item["model_version"] else "")),
        ("CDM COUNT", str(item["cdm_count"])),
        ("FIRST SEEN (UTC)", _e(item["first_seen_at"])),
    ])
    data_html = (
        f"<h2>PAIR FACTS</h2>{facts}"
        f"<h2>CDM HISTORY</h2>{_cdm_table(cdms)}"
        f"<h2>RISK PREDICTIONS</h2>{_prediction_table(preds)}"
    )
    label = (
        f"CONJUNCTION #{cid} · {item['primary']['name']} ↔ "
        f"{item['secondary']['name']}"
    )
    return label, data_html


def _object_data(conn, norad: int) -> tuple[str, str]:
    row = conn.execute(
        objects_api._OBJECTS_SQL + " WHERE sat.norad_id = ?", (norad,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Object {norad} not found")
    item = objects_api._item_from_row(row)
    facts = _kv_table([
        ("NAME", _e(item["name"])),
        ("NORAD ID", str(item["norad_id"])),
        ("INTL DESIGNATOR", _e(item["intl_designator"] or "—")),
        ("OBJECT TYPE", _e(item["object_type"])),
        ("TLE EPOCH (UTC)", _e(item["epoch"] or "—")),
        ("INCLINATION (DEG)", _fmt(item["inclination_deg"], 2)),
        ("RAAN (DEG)", _fmt(item["raan_deg"], 2)),
        ("ECCENTRICITY", "—" if item["eccentricity"] is None
         else f"{item['eccentricity']:.6f}"),
        ("ARG PERIGEE (DEG)", _fmt(item["arg_perigee_deg"], 2)),
        ("MEAN MOTION (REV/DAY)", _fmt(item["mean_motion_rev_day"], 4)),
        ("PERIOD (MIN)", _fmt(item["period_min"], 2)),
        ("APOGEE (KM)", _fmt(item["apogee_km"])),
        ("PERIGEE (KM)", _fmt(item["perigee_km"])),
        ("LAST UPDATED (UTC)", _e(item["last_updated"])),
    ])
    conj = conjunctions_api.query_summaries(
        conn,
        "(c.primary_object_id = ? OR c.secondary_object_id = ?)",
        (row["id"], row["id"]),
    )
    conj.sort(key=lambda i: (0 if i["status"] == "active" else 1,
                             *conjunctions_api.sort_key(i)))
    if conj:
        rows = []
        for i in conj:
            partner = (i["secondary"] if i["primary"]["norad_id"] == norad
                       else i["primary"])
            rows.append(
                f"<tr><td class='num'>{i['id']}</td>"
                f"<td>{_side_label(partner)}</td>"
                f"<td>{_e(i['tca'])}</td><td>{_e(i['status'].upper())}</td>"
                f"<td>{_chip(i['risk_level'])}</td>"
                f"<td class='num'>{_fmt(i['miss_distance_m'])}</td>"
                f"<td class='num'>{_fmt_pc(i['pc_latest'])}</td></tr>"
            )
        conj_html = (
            "<table><thead><tr><th class='num'>ID</th><th>PARTNER</th>"
            "<th>TCA (UTC)</th><th>STATUS</th><th>RISK</th>"
            "<th class='num'>MISS DIST (M)</th><th class='num'>PC</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        conj_html = '<p class="meta">NO CONJUNCTIONS RECORDED FOR THIS OBJECT.</p>'
    data_html = (
        f"<h2>ORBITAL ELEMENTS · LATEST TLE</h2>{facts}"
        f"<h2>CONJUNCTIONS</h2>{conj_html}"
    )
    return f"OBJECT {item['name']} · NORAD {norad}", data_html


def _overview_data(conn) -> tuple[str, str]:
    s = status_api.get_status(conn)
    facts = _kv_table([
        ("CATALOG SIZE", f"{s['catalog_size']:,}"),
        ("ACTIVE CONJUNCTIONS", str(s["active_conjunctions"])),
        ("ELEVATED", str(s["elevated_count"])),
        ("CRITICAL", str(s["critical_count"])),
        ("LAST INGEST (UTC)", _e(s["last_ingest_at"] or "—")),
        ("MODEL VERSION", _e(s["model_version"])),
        ("SCREENING WINDOW (H)", str(s["window_hours"])),
    ])
    top = conjunctions_api.query_summaries(conn, "c.status = 'active'")
    top.sort(key=conjunctions_api.sort_key)
    top = top[:10]
    if top:
        rows = "".join(
            f"<tr><td class='num'>{i['id']}</td>"
            f"<td>{_e(i['primary']['name'])} ↔ {_e(i['secondary']['name'])}</td>"
            f"<td>{_e(i['tca'])}</td><td>{_chip(i['risk_level'])}</td>"
            f"<td class='num'>{_fmt_pc(i['predicted_final_risk'])}</td>"
            f"<td class='num'>{_fmt(i['miss_distance_m'])}</td></tr>"
            for i in top
        )
        top_html = (
            "<table><thead><tr><th class='num'>ID</th><th>PAIR</th>"
            "<th>TCA (UTC)</th><th>RISK</th>"
            "<th class='num'>PC (PRED)</th><th class='num'>MISS DIST (M)</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )
    else:
        top_html = '<p class="meta">NO ACTIVE CONJUNCTIONS.</p>'
    data_html = (
        f"<h2>SYSTEM STATUS</h2>{facts}"
        f"<h2>TOP RISKS · ACTIVE</h2>{top_html}"
    )
    return "SYSTEM OVERVIEW", data_html


# Retention cap for data/reports: only the newest files by mtime survive.
REPORTS_KEEP = 100


def _prune_reports(reports_dir) -> None:
    """Delete all but the newest REPORTS_KEEP report files; never raises."""
    try:
        pages = sorted(reports_dir.glob("*.html"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in pages[REPORTS_KEEP:]:
            stale.unlink(missing_ok=True)
    except Exception:
        pass


# --- page assembly -----------------------------------------------------------

def _render_page(title: str, subject_label: str, data_html: str,
                 sections: list[dict]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    narrative = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = _e(str(section.get("heading") or "SECTION"))
        body = render_markdown(str(section.get("body") or ""))
        narrative.append(
            f'<section class="narrative"><h2>{heading}</h2>\n{body}</section>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)} — KESSLER REPORT</title>
<link rel="icon" href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g fill="none"><ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(30 12 12)" stroke="%239aa8b8" stroke-width="1.4"/><ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(-30 12 12)" stroke="%235c6875" stroke-width="1.4"/></g><circle cx="12" cy="12" r="1.3" fill="%239aa8b8"/><circle cx="19.2" cy="13.8" r="0.95" fill="%23dbe2ea"/></svg>'>
<style>{_CSS}</style>
</head>
<body>
<button class="print-btn" type="button" onclick="window.print()">PRINT / SAVE AS PDF</button>
<header>
  <div class="brand">KESSLER · AI ORBITAL RISK · REPORT</div>
  <h1>{_e(title)}</h1>
  <div class="meta">{_e(subject_label)} · GENERATED {generated}</div>
</header>
{data_html}
{"".join(narrative)}
<footer class="meta">KESSLER · AI ORBITAL RISK · DATA TABLES RENDERED
SERVER-SIDE FROM THE LIVE DATABASE · NARRATIVE SECTIONS AUTHORED BY
KESSLER AI</footer>
</body>
</html>
"""


def create_report(conn, title: str, subject: dict, sections: list[dict]) -> str:
    """Render a report to data/reports/<uuid>.html; return its /reports URL.

    Raises ValueError for an unknown subject type, a missing/invalid id, or
    a subject entity that does not exist - the tool executor surfaces those
    to the model as is_error tool results. After the write, data/reports is
    pruned to the newest REPORTS_KEEP files (retention, never fatal).
    """
    stype = (subject or {}).get("type")
    sid = (subject or {}).get("id")
    if stype == "conjunction":
        label, data_html = _conjunction_data(conn, _as_int(sid, "subject.id"))
    elif stype == "object":
        label, data_html = _object_data(conn, _as_int(sid, "subject.id"))
    elif stype == "overview":
        label, data_html = _overview_data(conn)
    else:
        raise ValueError("subject.type must be one of conjunction|object|overview")

    uid = str(uuid.uuid4())
    reports_dir = config.DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    page = _render_page(str(title or "Kessler report"), label, data_html,
                        sections or [])
    (reports_dir / f"{uid}.html").write_text(page, encoding="utf-8")
    _prune_reports(reports_dir)
    return f"/reports/{uid}"
