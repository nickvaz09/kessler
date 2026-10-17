/* Kessler AI - command deck, loaded on every page.
   Launcher button (bottom-right) opens a FULL-HEIGHT right drawer (440px,
   max 92vw) that overlays content from the topbar's bottom edge to the
   viewport bottom - no layout reflow. The launcher hides while the drawer
   is open. Shortcuts: Cmd/Ctrl+K toggles, '/' opens when focus is not in an
   input, ESC closes (the trajectory modal's ESC takes precedence).

   Transcript + open state persist in sessionStorage under schema v2
   ('kesslerAI.v2', migrated from the v1 key on first load). Each assistant
   turn may carry report_url, an applied-actions list (rendered as chips)
   and a stopped flag (' - stopped' marker for aborted streams).

   Streaming: POST /assistant/chat/stream (SSE over fetch + ReadableStream).
   data: lines are parsed incrementally by createSseParser (chunk splits,
   CRLF, multiple events per chunk all handled); text deltas re-render the
   live turn through the same safe markdown as completed turns (a CSS
   ::after caret marks the writing point; an empty live body shows a
   THINKING shimmer). Tool events build a step timeline in the live turn
   (▸ running / ✓ done - a step completes when text resumes or the next
   tool starts). action events EXECUTE IMMEDIATELY as they arrive
   (applyActionLive) - the screen changes while the reply is still
   streaming; only navigations (and dashboard-only actions when not on the
   dashboard) defer to end-of-turn via runActions. STOP (replaces SEND
   while streaming) aborts via AbortController keeping the partial text -
   actions already applied stay applied, deferred ones are dropped. If the
   stream fails before the first byte, the client retries ONCE against the
   non-streaming POST /assistant/chat.

   Assistant replies render through minimal safe markdown - FMT.esc FIRST,
   then **bold**, `code`, "- " lists and internal [text](/relative-url)
   links only; inlineMd strips NULs first so assistant text can never forge
   the \u0000-based code-span placeholders (audit-hardened - preserve).

   Server UI actions apply in response order with navigate last:
   - select_on_globe -> KGlobe.select on the dashboard, else folds into /?sel=
   - set_table_filter / set_globe_filter / set_globe_speed -> window.KesslerUI
     hooks (dashboard.js) on the dashboard, else fold into a navigation to /
     with ?risk=&search= params (dashboard.js applies them on load)
   - show_encounter -> KGlobe.showEncounter when the engine provides it
     (typeof-guarded), else navigate to /?conj=<id> (dashboard.js applies it)
   - open_simulator -> KSim.open on ANY page (typeof-guarded; the encounter
     simulator module js/simulator.js is loaded on all three pages; absent
     module is a silent no-op - there is no URL fallback). The chat drawer
     sits ABOVE the simulator overlay (z 740 > ksim 700) so the assistant
     stays visible while the simulation runs.
   - set_field -> types into any whitelisted-shape field by DOM id (INPUT/
     TEXTAREA/SELECT only, id re-validated client-side) and fires input +
     change so the page reacts as if the user typed
   - show_trajectory -> modal overlay reusing OrbitPlot (js/orbit_plot.js)
     and drawGroundTrack (js/groundtrack.js); canvases are drawn only after
     the modal is visible so CSS-size * DPR is correct. ESC/backdrop close;
     the modal (z 600) sits above the chat drawer (z 500).
   Failure modes: HTTP errors keep the user's message in the transcript and
   render a click-to-retry error line; a 503 (no API key) renders its detail
   string as a normal assistant-side message so the feature degrades
   gracefully. Mid-stream errors keep any partial text (marked ' - stopped')
   and surface the error line.

   push-to-talk mic (SpeechRecognition, button hidden when the engine
   is unavailable) dictates into the textarea - interim results replace only
   what the mic wrote (text typed before listening is preserved); a final
   result auto-sends; listening stops on send, close and drawer toggle.
   SHIFT BRIEF quick action (rendered first): GET /watchlog?limit=1 -> the
   entry appends as a PLAIN-STRING assistant turn (sanitizeMessages-safe,
   prefixed by a 'WATCHLOG · <utc>' line) and entry.spoken plays via POST
   /assistant/tts (one clip, stopped on drawer close; tts failure is silent
   - captions stand). No entries -> POST /watchlog/run with a 'RUNNING WATCH
   PASS' activity line; a 503 detail renders as a normal assistant message
   (the keyless degrade idiom). No Claude chat call on this path.

   "Ask Kessler": a floating glass sheet (bottom sheet on phones) with a
   pill launcher, greeting + suggestion chips, bubble-less assistant text,
   a quiet tool-step timeline and sentence-case action chips. A missing
   Anthropic key (503, or a stream error naming the key) is a NOTICE, not an
   error: state.notice = 'no_key' renders a friendly setup card and is never
   sent to the model. open_simulator may carry apply_plan -> KSim.open(
   {conjunctionId, applyPlan:true}) loads the one-click avoidance plan. */
'use strict';

(() => {
  const STORE_KEY = 'kesslerAI.v2';
  const LEGACY_STORE_KEY = 'kesslerAI.v1';

  /* ==== pure helpers (no DOM - exercised by the node test harness) ========= */

  const TOOL_LABELS = {
    get_status: 'Checking status',
    list_conjunctions: 'Looking at close approaches',
    get_conjunction: 'Reading the event',
    get_object: 'Reading the object',
    search_objects: 'Searching the catalog',
    get_risk: 'Evaluating risk',
    explain_risk: 'Explaining the risk',
    run_pipeline_refresh: 'Refreshing the catalog',
    get_changes: 'Checking what changed',
    get_passes: 'Predicting passes',
    get_anomalies: 'Scanning for maneuvers',
    simulate_maneuver: 'Simulating a burn',
    optimize_maneuver: 'Sizing the burn',
    plan_avoidance: 'Planning avoidance',
    list_user_objects: 'Reading your fleet',
    screen_user_object: 'Screening your satellite',
    create_report: 'Writing the report',
    navigate_user: 'Navigating',
    select_on_globe: 'Selecting on the globe',
    set_globe_filter: 'Filtering the globe',
    set_globe_speed: 'Setting the clock speed',
    set_table_filter: 'Filtering the table',
    fill_field: 'Filling in a field',
    show_trajectory: 'Plotting the trajectory',
    show_encounter: 'Staging the encounter',
    open_simulator: 'Opening the simulator',
  };

  function toolLabel(name) {
    if (typeof name !== 'string' || !name) return 'Working';
    if (TOOL_LABELS[name]) return TOOL_LABELS[name];
    const words = name.replace(/[^A-Za-z0-9]+/g, ' ').trim().toLowerCase();
    return words ? words.charAt(0).toUpperCase() + words.slice(1) : 'Working';
  }

  /* Same-origin paths only: "//x" and "/\\x" are both protocol-relative in
     browsers, so the resolved origin is checked as well. */
  function safeRelUrl(u) {
    if (typeof u !== 'string' || !u.startsWith('/') || u[1] === '/' || u[1] === '\\') {
      return null;
    }
    try {
      if (new URL(u, window.location.origin).origin !== window.location.origin) return null;
    } catch (_) { return null; }
    return u;
  }

  /* Keep only known action types with typed fields - stored in sessionStorage
     (chips) and fed to the executor, so everything is validated up front. */
  function sanitizeAction(a) {
    if (!a || typeof a !== 'object' || typeof a.type !== 'string') return null;
    const num = (v) => {
      const n = Number(v);
      return (v !== null && v !== undefined && v !== '' && Number.isFinite(n)) ? n : null;
    };
    switch (a.type) {
      case 'navigate': {
        const url = safeRelUrl(a.url);
        return url ? { type: 'navigate', url } : null;
      }
      case 'select_on_globe': {
        const n = num(a.norad);
        return n !== null ? { type: 'select_on_globe', norad: n } : null;
      }
      case 'set_table_filter': {
        const out = { type: 'set_table_filter' };
        if (typeof a.risk === 'string') out.risk = a.risk;
        if (typeof a.search === 'string') out.search = a.search;
        return out;
      }
      case 'set_globe_filter': {
        const out = { type: 'set_globe_filter' };
        if (Array.isArray(a.types)) out.types = a.types.slice(0, 8).map(String);
        const ro = a.risk_only !== undefined ? a.risk_only : a.riskOnly;
        if (ro !== undefined) out.risk_only = !!ro;
        return out;
      }
      case 'set_globe_speed': {
        const m = num(a.mult);
        return m !== null ? { type: 'set_globe_speed', mult: m } : null;
      }
      case 'show_trajectory': {
        const n = num(a.norad);
        if (n === null) return null;
        const out = { type: 'show_trajectory', norad: n };
        const p = num(a.partner_norad);
        if (p !== null) out.partner_norad = p;
        return out;
      }
      case 'show_encounter': {
        const out = { type: 'show_encounter' };
        const id = num(a.conjunction_id);
        const p = num(a.primary_norad);
        const s = num(a.secondary_norad);
        if (id !== null) out.conjunction_id = id;
        if (p !== null) out.primary_norad = p;
        if (s !== null) out.secondary_norad = s;
        if (typeof a.tca === 'string') out.tca = a.tca;
        return (id !== null || (p !== null && s !== null)) ? out : null;
      }
      case 'open_simulator': {
        const id = num(a.conjunction_id);
        if (id === null) return null;
        const out = { type: 'open_simulator', conjunction_id: id };
        if (a.apply_plan === true) out.apply_plan = true;
        return out;
      }
      case 'set_field': {
        if (typeof a.field_id !== 'string' ||
            !/^[A-Za-z0-9_-]{1,64}$/.test(a.field_id)) return null;
        if (typeof a.value !== 'string' || a.value.length > 500) return null;
        return { type: 'set_field', field_id: a.field_id, value: a.value };
      }
      default:
        return null; /* unknown action types dropped (forward-compatible) */
    }
  }

  function sanitizeMessages(list) {
    const out = [];
    if (!Array.isArray(list)) return out;
    for (const m of list) {
      if (!m || (m.role !== 'user' && m.role !== 'assistant') ||
          typeof m.content !== 'string') continue;
      const keep = { role: m.role, content: m.content };
      if (safeRelUrl(m.report_url)) keep.report_url = m.report_url;
      if (m.stopped === true) keep.stopped = true;
      if (m.role === 'user' && m.unanswered === true) keep.unanswered = true;
      if (Array.isArray(m.actions)) {
        const acts = m.actions.map(sanitizeAction).filter(Boolean);
        if (acts.length) keep.actions = acts;
      }
      out.push(keep);
    }
    return out;
  }

  /* Schema v2 load with migration from the v1 key. Both inputs are raw
     sessionStorage strings (or null). Returns {state, migrated}; `migrated`
     means the state came from the legacy key and should be re-saved under
     the v2 key. Corrupt JSON in either slot falls through to a fresh state. */
  function migrateStore(rawV2, rawV1) {
    const normalize = (src) => ({
      v: 2,
      open: !!(src && src.open),
      messages: sanitizeMessages(src && src.messages),
      pendingError: (src && typeof src.pendingError === 'string')
        ? src.pendingError : null,
      notice: (src && src.notice === 'no_key') ? 'no_key' : null,
    });
    if (typeof rawV2 === 'string' && rawV2) {
      try {
        const s = JSON.parse(rawV2);
        if (s && typeof s === 'object') return { state: normalize(s), migrated: false };
      } catch (_) { /* corrupted v2 - try legacy, else fresh */ }
    }
    if (typeof rawV1 === 'string' && rawV1) {
      try {
        const s = JSON.parse(rawV1);
        if (s && typeof s === 'object') return { state: normalize(s), migrated: true };
      } catch (_) { /* corrupted v1 - fresh */ }
    }
    return { state: normalize(null), migrated: false };
  }

  /* A missing Anthropic key is a setup state, not a failure. */
  function isNoKey(status, detail) {
    return status === 503 ||
      (typeof detail === 'string' && /api key|ANTHROPIC_API_KEY|ant auth login/i.test(detail));
  }

  /* Incremental SSE parser. push() accepts arbitrary chunk boundaries (events
     may split anywhere, including between CR and LF); end() flushes a final
     unterminated event. onData receives each event's data payload (multiple
     data: lines joined with '\n'); comment/field lines are ignored. */
  function createSseParser(onData) {
    let buf = '';
    let dataLines = [];
    const emit = () => {
      if (dataLines.length) {
        const d = dataLines.join('\n');
        dataLines = [];
        onData(d);
      }
    };
    const processLine = (line) => {
      if (line === '') { emit(); return; }        /* blank line ends the event */
      if (line.charCodeAt(0) === 58) return;      /* ':' comment */
      if (line.startsWith('data:')) {
        let v = line.slice(5);
        if (v.startsWith(' ')) v = v.slice(1);
        dataLines.push(v);
      }
      /* event:/id:/retry: fields carry nothing for this protocol - ignored */
    };
    return {
      push(text) {
        buf += text;
        /* hold back a trailing CR - its LF may arrive in the next chunk */
        const limit = buf.endsWith('\r') ? buf.length - 1 : buf.length;
        let i = 0;
        let start = 0;
        while (i < limit) {
          const c = buf.charCodeAt(i);
          if (c === 10) {                    /* \n */
            processLine(buf.slice(start, i));
            i += 1;
            start = i;
          } else if (c === 13) {             /* \r or \r\n */
            processLine(buf.slice(start, i));
            i += (buf.charCodeAt(i + 1) === 10) ? 2 : 1;
            start = i;
          } else {
            i += 1;
          }
        }
        buf = buf.slice(start);
      },
      end() {
        if (buf !== '') {
          let line = buf;
          buf = '';
          if (line.endsWith('\r')) line = line.slice(0, -1);
          processLine(line);
        }
        emit();
      },
    };
  }

  /* ==== node test harness hook (no DOM available) ========================== */

  if (typeof window === 'undefined' || typeof document === 'undefined') {
    globalThis.__KAI_TEST__ = {
      createSseParser, migrateStore, sanitizeAction, sanitizeMessages,
      toolLabel, isNoKey, STORE_KEY, LEGACY_STORE_KEY,
    };
    return;
  }

  /* ==== browser wiring ===================================================== */

  const PAGE = /object\.html$/.test(window.location.pathname) ? 'object'
    : /alerts\.html$/.test(window.location.pathname) ? 'alerts'
    : /fleet\.html$/.test(window.location.pathname) ? 'fleet'
    : /wall\.html$/.test(window.location.pathname) ? 'wall'
    : /model\.html$/.test(window.location.pathname) ? 'model'
    : 'dashboard';

  /* --- persisted state (schema v2; migrated from the v1 key) --------------- */

  let state;
  {
    let rawV2 = null;
    let rawV1 = null;
    try {
      rawV2 = sessionStorage.getItem(STORE_KEY);
      rawV1 = sessionStorage.getItem(LEGACY_STORE_KEY);
    } catch (_) { /* storage blocked - in-memory only */ }
    const m = migrateStore(rawV2, rawV1);
    state = m.state;
    try {
      if (rawV1 !== null) sessionStorage.removeItem(LEGACY_STORE_KEY);
      if (m.migrated) sessionStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch (_) { /* storage blocked/full - chat still works in-memory */ }
  }

  function save() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify(state)); }
    catch (_) { /* storage blocked/full - chat still works in-memory */ }
  }

  /* "Kessler AI isn't set up" - one user-facing sentence everywhere (the
     simulator's Narrate/Voice read the same flag and wording). The flag is
     learned from the first no-key reply and shared through sessionStorage
     (NOKEY_KEY) so every page and the simulator agree; any successful reply
     clears it. Operator detail lives only in a tooltip. */
  const NOKEY_KEY = 'kessler.ai.nokey';
  const NOKEY_TEXT = 'Kessler AI isn’t set up on this server yet.';
  const NOKEY_OPS = 'For operators: set ANTHROPIC_API_KEY (or run `ant auth login`) on the server and restart Kessler.';
  let aiNoKey = false;
  try { aiNoKey = sessionStorage.getItem(NOKEY_KEY) === '1'; } catch (_) { /* blocked */ }
  if (state.notice === 'no_key') aiNoKey = true;
  function setNoKey(on) {
    aiNoKey = !!on;
    try {
      if (aiNoKey) sessionStorage.setItem(NOKEY_KEY, '1');
      else sessionStorage.removeItem(NOKEY_KEY);
    } catch (_) { /* blocked - in-memory only */ }
    document.dispatchEvent(new CustomEvent('kessler:ai-availability', { detail: { available: !aiNoKey } }));
  }
  /* no-key reply: show the setup card and mark the question it left open */
  function noteNoKey() {
    state.notice = 'no_key';
    const last = state.messages[state.messages.length - 1];
    if (last && last.role === 'user') last.unanswered = true;
    setNoKey(true);
  }
  function noteAnswered() {
    for (const m of state.messages) delete m.unanswered;
    if (aiNoKey) setNoKey(false);
  }

  let inflight = false;   /* non-streaming (fallback) request in flight */
  let stream = null;      /* active stream turn state or null */
  let mic = null;         /* {rec, base} while push-to-talk listens */
  let briefBusy = false;  /* SHIFT BRIEF request in flight - blocks sends */
  let briefActivity = null; /* transcript activity label while the watch pass runs */
  let briefAudio = null;  /* {el, url} spoken watchlog clip - one at a time */

  /* --- minimal safe markdown (escape first, then transform) ---------------- */

  function inlineMd(escaped) {
    /* protect code spans so bold/link markers inside them stay literal;
       strip NULs first so assistant text can never forge the \u0000-based
       placeholder markers used by the restore pass below */
    const codes = [];
    let s = escaped.replace(/\u0000/g, '').replace(/`([^`\n]+)`/g, (m, c) => {
      codes.push(c);
      return `\u0000${codes.length - 1}\u0000`;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    /* internal links only: target must start with a single '/' */
    s = s.replace(/\[([^\]\n]+)\]\((\/(?!\/)[^)\s]*)\)/g, '<a href="$2">$1</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${codes[Number(i)]}</code>`);
  }

  function renderMarkdown(raw) {
    const s = inlineMd(FMT.esc(raw));
    const out = [];
    let list = null;   /* {tag:'ul'|'ol', items:[]} */
    let para = [];
    const flushPara = () => {
      if (para.length) { out.push(`<p>${para.join('<br>')}</p>`); para = []; }
    };
    const flushList = () => {
      if (list) {
        out.push(`<${list.tag}>${list.items.join('')}</${list.tag}>`);
        list = null;
      }
    };
    const pushItem = (tag, item) => {
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(`<li>${item}</li>`);
    };
    for (const line of s.split('\n')) {
      const t = line.trim();
      const ol = /^(\d+)\. (.*)$/.exec(t);
      const h = /^#{1,3} (.*)$/.exec(t);
      if (/^- /.test(t)) {
        flushPara();
        pushItem('ul', t.slice(2));
      } else if (ol) {
        flushPara();
        pushItem('ol', ol[2]);
      } else if (h) {
        flushPara();
        flushList();
        out.push(`<div class="kai-h">${h[1]}</div>`);
      } else if (/^---+$/.test(t)) {
        flushPara();
        flushList();
        out.push('<hr class="kai-rule">');
      } else if (t === '') {
        flushPara();
        flushList();
      } else {
        flushList();
        para.push(t);
      }
    }
    flushPara();
    flushList();
    return out.join('');
  }

  /* --- widget DOM ---------------------------------------------------------- */

  const MARK_SVG =
    '<svg class="kai-mark" viewBox="0 0 24 24" aria-hidden="true">' +
      '<g fill="none" stroke="currentColor" stroke-width="1.6">' +
        '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(30 12 12)"/>' +
        '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-30 12 12)" opacity="0.55"/>' +
      '</g>' +
      '<circle cx="12" cy="12" r="1.6" fill="currentColor"/>' +
    '</svg>';
  /* Thinking: the mark comes alive. Two satellites ride its rings in
     opposite directions at different periods, each trailing a short comet
     tail, around a breathing core. SMIL motion follows the exact ellipse, so
     no id/JS is needed and any number of copies can run at once. Reduced
     motion gets the still mark. */
  const REDUCE_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const RING_PATH = 'M21 12A9 3.6 0 1 1 3 12A9 3.6 0 1 1 21 12';
  const comet = (dur, reverse, step, cls) => {
    const kp = reverse ? ' keyPoints="1;0" keyTimes="0;1" calcMode="linear"' : '';
    return [[1.25, 1], [1.0, 0.55], [0.8, 0.3], [0.6, 0.14]].map(([r, o], i) =>
      `<circle class="${cls}" r="${r}" opacity="${o}">` +
        `<animateMotion dur="${dur}s" repeatCount="indefinite" begin="${(-(3 - i) * step).toFixed(2)}s"` +
        ` path="${RING_PATH}"${kp}/></circle>`).join('');
  };
  const ORBITER_SVG = REDUCE_MOTION ? MARK_SVG.replace('class="kai-mark"', 'class="kai-mark kai-orbiter"') :
    '<svg class="kai-mark kai-orbiter" viewBox="0 0 24 24" aria-hidden="true">' +
      '<g fill="none" stroke="currentColor" stroke-width="1.3">' +
        '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(30 12 12)" opacity="0.5"/>' +
        '<ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-30 12 12)" opacity="0.3"/>' +
      '</g>' +
      '<circle class="ko-core" cx="12" cy="12" r="1.5" fill="currentColor">' +
        '<animate attributeName="r" values="1.5;1.9;1.5" dur="1.8s" repeatCount="indefinite"/>' +
        '<animate attributeName="opacity" values="1;0.6;1" dur="1.8s" repeatCount="indefinite"/></circle>' +
      `<g transform="rotate(30 12 12)">${comet(1.8, false, 0.055, 'ko-a')}</g>` +
      `<g transform="rotate(-30 12 12)">${comet(2.7, true, 0.08, 'ko-b')}</g>` +
    '</svg>';
  const thinkingHtml = (label) =>
    `<div class="kai-thinking" role="status">${ORBITER_SVG}<span>${FMT.esc(label)}</span></div>`;
  const ICON_SEND = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5.5 11.5L12 5l6.5 6.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const ICON_STOP = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>';
  const ICON_MIC = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3.5" width="6" height="11" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v2.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
  const ICON_NEW = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
  const ICON_CLOSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

  /* launcher: "Ask Kessler" pill (icon-only circle on phones) */
  const fab = document.createElement('button');
  fab.type = 'button';
  fab.className = 'kai-fab';
  fab.setAttribute('aria-label', 'Ask Kessler, the AI assistant');
  fab.title = 'Ask Kessler (⌘K)';
  fab.innerHTML = `${MARK_SVG}<span class="kai-fab__tag">Ask Kessler</span>`;

  const panel = document.createElement('section');
  panel.className = 'kai-panel';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Ask Kessler assistant');
  panel.innerHTML =
    '<div class="kai-resize" aria-hidden="true"></div>' +
    '<div class="kai-hd">' +
      `<span class="kai-avatar">${MARK_SVG}${ORBITER_SVG}</span>` +
      '<div class="kai-hd__txt">' +
        '<span class="kai-title">Ask Kessler</span>' +
        '<span class="kai-meta">Ready</span>' +
      '</div>' +
      `<button type="button" class="kai-hd-btn" data-act="clear" aria-label="New conversation" title="New conversation">${ICON_NEW}</button>` +
      `<button type="button" class="kai-hd-btn" data-act="close" aria-label="Close assistant" title="Close (Esc)">${ICON_CLOSE}</button>` +
    '</div>' +
    '<div class="kai-log" role="log" aria-live="polite"></div>' +
    '<div class="kai-input">' +
      '<div class="kai-composer">' +
        '<textarea class="kai-text" rows="1" placeholder="Ask Kessler anything" ' +
          'aria-label="Message Kessler"></textarea>' +
        `<button type="button" class="kai-mic" title="Push to talk" aria-pressed="false" aria-label="Push to talk">${ICON_MIC}</button>` +
        `<button type="button" class="kai-send" aria-label="Send">${ICON_SEND}</button>` +
      '</div>' +
      '<div class="kai-foot">Kessler can make mistakes. Numbers come from the live database; maneuver figures are screening-grade.</div>' +
    '</div>';
  document.body.appendChild(fab);
  document.body.appendChild(panel);

  const logEl = panel.querySelector('.kai-log');
  const metaEl = panel.querySelector('.kai-meta');
  const textEl = panel.querySelector('.kai-text');
  const sendEl = panel.querySelector('.kai-send');
  const micEl = panel.querySelector('.kai-mic');

  /* push-to-talk requires a speech engine - the button vanishes without one */
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  micEl.hidden = !SpeechRec;

  /* --- drawer geometry (topbar bottom -> viewport bottom) ------------------ */

  function positionDrawer() {
    if (panel.hidden) return;
    const tb = document.querySelector('.topbar');
    const top = tb ? Math.max(0, tb.getBoundingClientRect().bottom) : 0;
    panel.style.top = `${Math.round(top) + (window.innerWidth > 640 ? 12 : 8)}px`;
  }
  window.addEventListener('resize', positionDrawer);
  window.addEventListener('scroll', positionDrawer, { passive: true });

  /* --- launcher vs page docks (QA A4) ----------------------------------------
     The launcher must never cover page controls pinned to the bottom edge
     (the Overview control dock and its time scrubber on phones). Any element
     matching DOCK_SEL that overlaps the launcher's resting spot lifts it to
     sit just above the dock's top edge (--kai-fab-lift); body.has-dock marks
     pages that have one. The simulator hides the launcher entirely (css). */
  const DOCK_SEL = '#stage-dock, .stage__dock, .ov-dock, [data-kai-dock]';
  let dockRaf = 0;
  function placeFab() {
    dockRaf = 0;
    const docks = Array.from(document.querySelectorAll(DOCK_SEL));
    document.body.classList.toggle('has-dock', docks.length > 0);
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const phone = vw <= 640;
    const size = phone ? 52 : 48;
    const inset = phone ? 14 : 20;
    /* full pill width, measured without the compact class */
    const pillW = phone ? size : (fab.classList.contains('is-compact')
      ? (fab.dataset.pillW ? Number(fab.dataset.pillW) : 150) : (fab.offsetWidth || 150));
    if (!phone && fab.offsetWidth && !fab.classList.contains('is-compact')) fab.dataset.pillW = String(fab.offsetWidth);
    const restFor = (w) => ({ l: vw - inset - w, r: vw - inset, t: vh - inset - size, b: vh - inset });
    const hits = (rest) => docks.map((d) => d.getBoundingClientRect()).filter((r) =>
      r.width && r.height && r.bottom > 0 && r.top < vh &&
      r.left < rest.r + 6 && r.right > rest.l - 6 && r.top < rest.b + 6 && r.bottom > rest.t - 6);
    let lift = 0;
    let compact = false;
    const pillHits = hits(restFor(pillW));
    if (pillHits.length) {
      /* a dock that only clips the pill's label: collapse to the icon */
      if (!phone && !hits(restFor(size)).length) compact = true;
      else for (const r of pillHits) lift = Math.max(lift, vh - r.top + 10 - inset);
    }
    lift = Math.min(Math.max(0, Math.round(lift)), Math.round(vh * 0.7));
    fab.style.setProperty('--kai-fab-lift', `${lift}px`);
    fab.classList.toggle('is-lifted', lift > 0);
    fab.classList.toggle('is-compact', compact);
  }
  function queueFab() { if (!dockRaf) dockRaf = requestAnimationFrame(placeFab); }
  window.addEventListener('resize', queueFab);
  window.addEventListener('scroll', queueFab, { passive: true, capture: true });
  /* docks animate in / pages render late: re-check a few times after load */
  [0, 300, 900, 2000].forEach((ms) => setTimeout(queueFab, ms));

  /* --- adjustable width (drag the left edge; persists per browser) ---------- */

  const WIDTH_LS_KEY = 'kesslerAI.width';
  const WIDTH_MIN = 340;

  function widthMax() {
    return Math.max(WIDTH_MIN, Math.floor(window.innerWidth * 0.92));
  }

  function applyWidth(w) {
    const clamped = Math.max(WIDTH_MIN, Math.min(widthMax(), Math.round(w)));
    panel.style.width = `${clamped}px`;
    return clamped;
  }

  try {
    const saved = Number(localStorage.getItem(WIDTH_LS_KEY));
    if (Number.isFinite(saved) && saved > 0) applyWidth(saved);
  } catch (_) { /* storage blocked - default width */ }

  {
    const grip = panel.querySelector('.kai-resize');
    let drag = null;   /* {startX, startW} while resizing */
    grip.addEventListener('pointerdown', (ev) => {
      drag = { startX: ev.clientX, startW: panel.getBoundingClientRect().width };
      try { grip.setPointerCapture(ev.pointerId); } catch (_) { /* older engines */ }
      ev.preventDefault();
    });
    grip.addEventListener('pointermove', (ev) => {
      if (!drag) return;
      applyWidth(drag.startW + (drag.startX - ev.clientX));
    });
    const endDrag = () => {
      if (!drag) return;
      drag = null;
      try {
        localStorage.setItem(WIDTH_LS_KEY,
          String(Math.round(panel.getBoundingClientRect().width)));
      } catch (_) { /* storage blocked - width lives for this page only */ }
    };
    grip.addEventListener('pointerup', endDrag);
    grip.addEventListener('pointercancel', endDrag);
  }

  /* --- auto-growing textarea (1 -> 4 rows; CSS max-height caps it) --------- */

  function autosize() {
    textEl.style.height = 'auto';
    textEl.style.height = `${textEl.scrollHeight + 2}px`; /* +2 borders */
  }
  textEl.addEventListener('input', autosize);

  /* --- push-to-talk --------------------------------------------------------- */

  /* Text typed before the mic started is never overwritten - dictation is
     appended after it (space-separated) and only the mic-written tail is
     replaced as interim results firm up. */
  function micJoin(base, txt) {
    if (!txt) return base;
    if (!base) return txt;
    return /\s$/.test(base) ? base + txt : `${base} ${txt}`;
  }

  function micStop() {
    if (!mic) return;
    micEl.setAttribute('aria-pressed', 'false');
    const rec = mic.rec;
    mic = null;
    micEl.classList.remove('is-active');
    updateMeta();
    try { rec.abort(); } catch (_) { /* engine already ended */ }
  }

  function micStart() {
    if (!SpeechRec || mic || inflight || stream || briefBusy) return;
    let rec;
    try { rec = new SpeechRec(); } catch (_) { return; } /* engine refused */
    rec.lang = 'en-US';
    rec.interimResults = true;
    const my = { rec, base: textEl.value, micText: '' };
    rec.onresult = (ev) => {
      if (mic !== my) return;
      /* the user may TYPE while dictating: re-derive the base as whatever
         the field holds minus the tail the mic wrote last, so keystrokes
         between interim results are never clobbered */
      const cur = textEl.value;
      const joined = micJoin(my.base, my.micText);
      if (cur !== joined) {
        my.base = my.micText && cur.endsWith(my.micText)
          ? cur.slice(0, cur.length - my.micText.length).replace(/\s$/, '')
          : cur;
      }
      let txt = '';
      let done = false;
      for (let i = 0; i < ev.results.length; i++) {
        const r = ev.results[i];
        txt += (r[0] && r[0].transcript) || '';
        if (r.isFinal) done = true;
      }
      txt = txt.trim();
      my.micText = txt;
      textEl.value = micJoin(my.base, txt);
      autosize();
      if (done) {           /* final result - fill and auto-send */
        micStop();
        if (textEl.value.trim()) send(textEl.value);
      }
    };
    /* errors (no-speech, denied, aborted) degrade silently - state resets */
    rec.onerror = () => { if (mic === my) micStop(); };
    rec.onend = () => { if (mic === my) micStop(); };
    try { rec.start(); } catch (_) { return; }
    mic = my;
    micEl.classList.add('is-active');
    micEl.setAttribute('aria-pressed', 'true');
    updateMeta();
  }

  micEl.addEventListener('click', () => {
    if (mic) micStop();
    else micStart();
  });

  /* --- transcript rendering ------------------------------------------------ */

  function chipLabel(a) {
    switch (a.type) {
      case 'navigate': return `Opened ${a.url}`;
      case 'select_on_globe': return `Selected ${a.norad} on the globe`;
      case 'set_table_filter': {
        const parts = [];
        if (a.risk) parts.push(String(a.risk));
        if (a.search) parts.push(`"${a.search}"`);
        return parts.length ? `Table filtered: ${parts.join(', ')}` : 'Table filters cleared';
      }
      case 'set_globe_filter': {
        const parts = [];
        if (Array.isArray(a.types) && a.types.length) {
          parts.push(a.types.map((t) => String(t).toLowerCase()).join(', '));
        }
        if (a.risk_only) parts.push('risk only');
        return parts.length ? `Globe filtered: ${parts.join(', ')}` : 'Globe filters cleared';
      }
      case 'set_globe_speed': return `Clock at ${a.mult}×`;
      case 'show_trajectory':
        return a.partner_norad !== undefined
          ? `Trajectory ${a.norad} and ${a.partner_norad}` : `Trajectory ${a.norad}`;
      case 'show_encounter': {
        const pair = (a.primary_norad !== undefined && a.secondary_norad !== undefined)
          ? `${a.primary_norad} and ${a.secondary_norad}`
          : `#${a.conjunction_id}`;
        return `Encounter ${pair} on the globe`;
      }
      case 'open_simulator':
        return a.apply_plan
          ? `Avoidance plan in the simulator · #${a.conjunction_id}`
          : `Simulator · #${a.conjunction_id}`;
      case 'set_field': {
        const v = a.value.length > 18 ? `${a.value.slice(0, 17)}…` : a.value;
        return v ? `Filled ${a.field_id}: "${v}"` : `Cleared ${a.field_id}`;
      }
      default: return null;
    }
  }

  function chipsHtml(actions, reportUrl) {
    const chips = [];
    for (const a of actions || []) {
      const label = chipLabel(a);
      if (!label) continue;
      /* re-open actions stay actionable from the transcript */
      if (a.type === 'open_simulator') {
        chips.push(`<button type="button" class="kai-chip" data-sim="${FMT.esc(a.conjunction_id)}"` +
          `${a.apply_plan ? ' data-plan="1"' : ''}>${FMT.esc(label)}</button>`);
      } else {
        chips.push(`<span class="kai-chip">${FMT.esc(label)}</span>`);
      }
    }
    if (safeRelUrl(reportUrl)) {
      chips.push(`<a class="kai-chip kai-chip-link" href="${FMT.esc(reportUrl)}" ` +
        'target="_blank" rel="noopener">Open report ↗</a>');
    }
    return chips.join('');
  }

  const QUICK_ACTIONS = [
    /* the prompt is what the user sees in their bubble - keep it natural */
    ['What needs my attention?',
      'What needs my attention right now?'],
    ['Plan avoidance for the top event',
      'Plan an avoidance burn for the riskiest upcoming event and show it to me in the simulator.'],
    ['Watch the riskiest encounter',
      'Open the simulator on the riskiest encounter so I can watch the pass.'],
  ];

  function blankHtml() {
    const chips =
      '<button type="button" class="kai-suggest" data-brief>' +
        '<span class="kai-suggest__t">Brief me</span>' +
        '<span class="kai-suggest__s">The latest watch-log entry, read aloud when voice is set up</span>' +
      '</button>' +
      QUICK_ACTIONS.map(([label], i) =>
        `<button type="button" class="kai-suggest" data-qa="${i}">` +
        `<span class="kai-suggest__t">${label}</span></button>`
      ).join('');
    return '<div class="kai-blank">' +
      `<div class="kai-blank__mark">${MARK_SVG}</div>` +
      '<div class="kai-blank__title">How can I help?</div>' +
      '<div class="kai-blank__sub">Ask about close approaches, risk or your fleet. ' +
        'I can open the simulator, filter the table and plan avoidance burns while we talk.</div>' +
      `<div class="kai-qa">${chips}</div>` +
      '</div>';
  }

  function noticeHtml() {
    return `<div class="kai-notice" title="${FMT.esc(NOKEY_OPS)}">` +
      `<div class="kai-notice__title">${NOKEY_TEXT}</div>` +
      '<p>Everything else keeps working without it: alerts, the simulator and one-click avoidance plans.</p>' +
      '<button type="button" class="btn btn--xs" data-act="dismiss-notice">Got it</button>' +
      '</div>';
  }

  function turnHtml(m) {
    if (m.role === 'user') {
      const un = m.unanswered
        ? `<div class="kai-unanswered" title="${FMT.esc(NOKEY_OPS)}">Not answered: ${NOKEY_TEXT}</div>` : '';
      return '<div class="kai-turn kai-turn--user">' +
        `<div class="kai-body kai-body--user">${FMT.esc(m.content)}</div>${un}</div>`;
    }
    const stopped = m.stopped
      ? '<div class="kai-stopped">Stopped</div>' : '';
    const chips = chipsHtml(m.actions, m.report_url);
    const chipsRow = chips ? `<div class="kai-chips">${chips}</div>` : '';
    return '<div class="kai-turn">' +
      `<div class="kai-body">${renderMarkdown(m.content)}</div>${stopped}${chipsRow}</div>`;
  }

  function updateMeta() {
    let live = null;
    if (stream) {
      const running = stream.steps.filter((s) => !s.done);
      live = running.length ? `${running[running.length - 1].label}…` : 'Writing…';
    }
    metaEl.textContent = live ||
      ((inflight || briefBusy) ? 'Working…'
        : (mic ? 'Listening…' : (aiNoKey ? 'Not set up' : 'Ready')));
    metaEl.title = (!live && !inflight && !briefBusy && !mic && aiNoKey) ? NOKEY_OPS : '';
    panel.classList.toggle('is-busy', !!(stream || inflight || briefBusy));
    panel.classList.toggle('is-nokey', aiNoKey);
  }

  function updateSendButton() {
    if (stream) {
      sendEl.innerHTML = ICON_STOP;
      sendEl.setAttribute('aria-label', 'Stop');
      sendEl.disabled = false;
      sendEl.classList.add('kai-send--stop');
    } else {
      sendEl.innerHTML = ICON_SEND;
      sendEl.setAttribute('aria-label', 'Send');
      sendEl.disabled = inflight || briefBusy;
      sendEl.classList.remove('kai-send--stop');
    }
    /* the mic goes quiet with SEND: dictation finishing mid-stream would be
       silently swallowed by send()'s busy guard, and the only visible button
       reads STOP - a tap meant as 'send' would abort the reply instead */
    micEl.disabled = !!(stream || inflight || briefBusy);
  }

  function render() {
    const idle = !inflight && !stream && !briefActivity;
    let html = (!state.messages.length && idle && !state.notice)
      ? blankHtml()
      : state.messages.map(turnHtml).join('');
    if (state.notice === 'no_key' && !stream && !inflight) html += noticeHtml();
    /* after the setup card is dismissed, an unanswered question must not be
       a dead end: the greeting and suggestions come back under it */
    const last = state.messages[state.messages.length - 1];
    if (state.messages.length && idle && !state.notice && !state.pendingError &&
        last && last.role === 'user' && last.unanswered) html += blankHtml();
    if (stream) {
      html += '<div class="kai-turn kai-turn--live">' +
        thinkingHtml('Thinking') +
        '<div class="kai-steps" hidden></div>' +
        '<div class="kai-body kai-body--live"></div>' +
        '<div class="kai-chips" hidden></div>' +
        '</div>';
    } else if (inflight) {
      html += '<div class="kai-turn">' + thinkingHtml('Thinking') + '</div>';
    } else if (briefActivity) {
      /* SHIFT BRIEF demo path: the watch pass runs server-side (~a minute) */
      html += '<div class="kai-turn">' + thinkingHtml(briefActivity) + '</div>';
    } else if (state.pendingError) {
      html += '<div class="kai-error">' +
        `<span>That didn’t go through: ${FMT.esc(state.pendingError)}</span>` +
        '<button type="button" class="kai-error__retry">Try again</button></div>';
    }
    logEl.innerHTML = html;
    if (stream) {
      const live = logEl.querySelector('.kai-turn--live');
      stream.els = {
        steps: live.querySelector('.kai-steps'),
        body: live.querySelector('.kai-body'),
        chips: live.querySelector('.kai-chips'),
      };
      updateLive(true);
    }
    logEl.scrollTop = logEl.scrollHeight;
    updateSendButton();
    updateMeta();
  }

  /* Live-turn refresh during streaming: the same safe markdown as completed
     turns, re-rendered per delta (chat turns are small). The writing caret
     and the empty-body THINKING shimmer are pure CSS on .kai-body--live. */
  function updateLive(force) {
    if (!stream || !stream.els) return;
    const nearBottom = force ||
      (logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 48);
    stream.els.body.innerHTML = renderMarkdown(stream.text);
    if (stream.steps.length) {
      stream.els.steps.innerHTML = stream.steps.map((s) =>
        `<div class="kai-step${s.done ? ' kai-step--done' : ''}">` +
        '<span class="kai-step__mark" aria-hidden="true"></span>' +
        `${FMT.esc(s.label)}</div>`).join('');
      stream.els.steps.hidden = false;
    }
    if (stream.actions.length) {
      stream.els.chips.innerHTML = chipsHtml(stream.actions, null);
      stream.els.chips.hidden = false;
    }
    if (nearBottom) logEl.scrollTop = logEl.scrollHeight;
    updateMeta();
  }

  function markStepsDone() {
    for (const s of stream.steps) s.done = true;
  }

  function setOpen(open) {
    state.open = !!open;
    save();
    micStop();                          /* listening never survives a toggle */
    if (!state.open) stopBriefAudio();  /* spoken brief dies with the drawer */
    const hadFocus = !state.open && panel.contains(document.activeElement);
    panel.hidden = !state.open;
    fab.hidden = state.open;
    if (hadFocus) fab.focus();   /* closing must not strand keyboard focus */
    if (state.open) {
      positionDrawer();
      render();
      autosize();
      textEl.focus();
    }
  }

  /* --- request plumbing ---------------------------------------------------- */

  function buildContext() {
    const ctx = { page: PAGE, url: window.location.pathname + window.location.search };
    let sel = null;
    if (PAGE === 'dashboard' && window.KesslerUI &&
        typeof window.KesslerUI.selectedNorad === 'function') {
      try { sel = window.KesslerUI.selectedNorad(); } catch (_) { /* hook fault */ }
    } else if (PAGE === 'object') {
      const id = Number(new URLSearchParams(window.location.search).get('id'));
      if (Number.isFinite(id) && id > 0) sel = id;
    }
    if (sel !== null && sel !== undefined) ctx.selected_norad = sel;
    /* ground site saved on any object page - lets get_passes answer
       "when can we see it?" without asking */
    try {
      const site = JSON.parse(localStorage.getItem('kessler.site'));
      if (site && Number.isFinite(Number(site.lat)) && Number.isFinite(Number(site.lon))) {
        ctx.site = { lat: Number(site.lat), lon: Number(site.lon),
                     alt_m: Number(site.alt_m) || 0 };
      }
    } catch (_) { /* storage blocked or malformed - no site */ }
    return ctx;
  }

  function buildPayload() {
    return JSON.stringify({
      messages: state.messages.map((m) => ({ role: m.role, content: m.content })),
      context: buildContext(),
    });
  }

  /* send(text) appends a new user turn; send(null) retries the last user turn
     already in the transcript (the click-to-retry path). */
  async function send(newText) {
    micStop();
    if (inflight || stream || briefBusy) return;
    if (typeof newText === 'string') {
      const text = newText.trim();
      if (!text) return;
      state.messages.push({ role: 'user', content: text });
      textEl.value = '';
      autosize();
    } else {
      /* click-to-retry: a mid-stream failure commits the partial reply as a
         stopped assistant turn - drop trailing stopped turns so the retry
         actually re-sends the last user turn instead of only dismissing */
      let last = state.messages[state.messages.length - 1];
      while (last && last.role === 'assistant' && last.stopped === true) {
        state.messages.pop();
        last = state.messages[state.messages.length - 1];
      }
      if (!last || last.role !== 'user') {
        state.pendingError = null;
        save();
        render();
        return;
      }
    }
    state.pendingError = null;
    state.notice = null;
    save();
    await runStreamTurn();
  }

  function stopStream() {
    if (!stream) return;
    stream.stopped = true;
    try { stream.ctrl.abort(); } catch (_) { /* already settled */ }
  }

  /* Commit the partial turn after a user STOP (or CLEAR-discard mid-stream). */
  function finishStopped() {
    if (!stream) return;
    if (stream.discard) { stream = null; render(); return; }
    if (stream.text) {
      const msg = { role: 'assistant', content: stream.text, stopped: true };
      if (stream.actions.length) msg.actions = stream.actions;
      state.messages.push(msg);
    }
    stream = null;
    save();
    render();
  }

  async function runStreamTurn() {
    const payload = buildPayload();
    const ctrl = new AbortController();
    stream = {
      text: '', actions: [], deferred: [], steps: [],
      ctrl, stopped: false, discard: false, els: null,
    };
    render();

    let outcome = null;   /* {kind:'done',reportUrl} | {kind:'error',detail} */
    let firstByte = false;
    try {
      const res = await fetch('/assistant/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        /* 404/503/... before any byte - throw into the fallback path below */
        let detail = `${res.status} ${res.statusText}`;
        try {
          const j = await res.json();
          if (j && j.detail) detail = String(j.detail);
        } catch (_) { /* non-JSON error body */ }
        const err = new Error(detail);
        err.status = res.status;
        throw err;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parser = createSseParser((data) => {
        let ev = null;
        try { ev = JSON.parse(data); } catch (_) { return; }
        if (!ev || typeof ev !== 'object' || outcome) return;
        if (ev.type === 'text') {
          stream.text += typeof ev.text === 'string' ? ev.text : '';
          markStepsDone();             /* text resumed - steps completed */
          updateLive(false);
        } else if (ev.type === 'tool') {
          markStepsDone();             /* a new step starts - prior ones ran */
          stream.steps.push({ label: toolLabel(ev.name), done: false });
          updateLive(false);
        } else if (ev.type === 'action') {
          const a = sanitizeAction(ev.action);
          if (a) {
            /* the screen changes NOW, while the reply still streams;
               only navigations (and actions this page can't host) wait */
            if (!applyActionLive(a)) stream.deferred.push(a);
            stream.actions.push(a);
            updateLive(false);
          }
        } else if (ev.type === 'done') {
          outcome = {
            kind: 'done',
            reportUrl: typeof ev.report_url === 'string' ? ev.report_url : null,
          };
        } else if (ev.type === 'error') {
          outcome = {
            kind: 'error',
            detail: typeof ev.detail === 'string' && ev.detail
              ? ev.detail : 'the reply stream failed',
          };
        }
      });
      while (!outcome) {
        const step = await reader.read();
        if (step.done) break;
        if (step.value && step.value.length) {
          firstByte = true;
          parser.push(decoder.decode(step.value, { stream: true }));
        }
      }
      if (!outcome) {
        parser.push(decoder.decode());
        parser.end();
      } else {
        try { reader.cancel(); } catch (_) { /* stream already closed */ }
      }
    } catch (err) {
      if (stream.discard) { stream = null; render(); return; }
      if (stream.stopped) { finishStopped(); return; }
      if (!firstByte) {
        /* stream failed before first byte - one retry, non-streaming */
        stream = null;
        await sendNonStreaming(payload);
        return;
      }
      outcome = {
        kind: 'error',
        detail: String((err && err.message) || err || 'the reply stream failed'),
      };
    }

    if (stream.discard) { stream = null; render(); return; }
    if (stream.stopped && (!outcome || outcome.kind !== 'done')) {
      finishStopped();
      return;
    }
    if (!outcome) outcome = { kind: 'error', detail: 'the reply ended unexpectedly' };

    if (outcome.kind === 'done') {
      noteAnswered();
      const msg = { role: 'assistant', content: stream.text };
      if (safeRelUrl(outcome.reportUrl)) msg.report_url = outcome.reportUrl;
      if (stream.actions.length) msg.actions = stream.actions;
      const deferred = stream.deferred;   /* live actions already ran */
      stream = null;
      state.messages.push(msg);
      save();
      render();
      runActions(deferred);
    } else {
      /* mid-stream error: keep any partial text (marked), surface the error */
      if (stream.text) {
        const msg = { role: 'assistant', content: stream.text, stopped: true };
        if (stream.actions.length) msg.actions = stream.actions;
        state.messages.push(msg);
      }
      if (!stream.text && isNoKey(0, outcome.detail)) noteNoKey();
      else state.pendingError = outcome.detail;
      stream = null;
      save();
      render();
    }
  }

  /* Non-streaming fallback (also the graceful-degrade path for 503 no-key). */
  async function sendNonStreaming(payload) {
    inflight = true;
    render();
    try {
      const res = await fetch('/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      });
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
          const j = await res.json();
          if (j && j.detail) detail = String(j.detail);
        } catch (_) { /* non-JSON error body - keep the status line */ }
        const err = new Error(detail);
        err.status = res.status;
        throw err;
      }
      const resp = await res.json();
      inflight = false;
      noteAnswered();
      const msg = {
        role: 'assistant',
        content: typeof resp.reply === 'string' ? resp.reply : '',
      };
      if (safeRelUrl(resp.report_url)) msg.report_url = resp.report_url;
      const actions = Array.isArray(resp.actions)
        ? resp.actions.map(sanitizeAction).filter(Boolean) : [];
      if (actions.length) msg.actions = actions;
      state.messages.push(msg);
      save();
      render();
      runActions(actions);
    } catch (err) {
      inflight = false;
      if (err && isNoKey(err.status, String(err.message))) {
        /* no API key - a friendly setup card, never a red error */
        noteNoKey();
      } else {
        state.pendingError = String((err && err.message) || err || 'the request failed');
      }
      save();
      render();
    }
  }

  /* --- shift brief (watchlog read + TTS, no Claude chat call) -------------- */

  function stopBriefAudio() {
    if (!briefAudio) return;
    const cur = briefAudio;
    briefAudio = null;
    if (cur.el) { try { cur.el.pause(); } catch (_) { /* not started */ } }
    if (cur.url) URL.revokeObjectURL(cur.url);
  }

  /* One clip at a time; a tts failure (503 no key, synth error, autoplay
     denial) is silent - the transcript captions stand on their own. */
  function speakBrief(text, sig) {
    stopBriefAudio();
    if (typeof text !== 'string' || !text.trim() || panel.hidden) return;
    if (typeof sig !== 'string' || !sig) return; /* synthesis is server-signed */
    const mine = { el: null, url: null };
    briefAudio = mine;
    fetch('/assistant/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.slice(0, 600), sig }),
    }).then((r) => {
      if (!r.ok) throw new Error(`tts ${r.status}`);
      return r.blob();
    }).then((blob) => {
      if (briefAudio !== mine) return;   /* stopped/superseded while fetching */
      mine.url = URL.createObjectURL(blob);
      const a = new Audio(mine.url);
      mine.el = a;
      a.addEventListener('ended', () => {
        if (briefAudio === mine) stopBriefAudio();
      });
      a.play().catch(() => { /* autoplay denied - captions carry it */ });
    }).catch(() => {
      if (briefAudio === mine) briefAudio = null;
    });
  }

  async function briefDetail(res) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j && j.detail) detail = String(j.detail);
    } catch (_) { /* non-JSON error body - keep the status line */ }
    return detail;
  }

  /* SHIFT BRIEF: latest watchlog entry into the transcript (+ spoken clip);
     no entries yet -> run the watch pass now (the START SHIFT demo path).
     Failures - including the keyless 503 - render their detail string as a
     normal assistant message (the sendNonStreaming degrade idiom). */
  async function shiftBrief() {
    if (briefBusy || inflight || stream) return;
    briefBusy = true;
    updateSendButton();
    updateMeta();
    let entry = null;
    let failDetail = null;
    try {
      const res = await fetch('/watchlog?limit=1');
      if (!res.ok) failDetail = await briefDetail(res);
      else {
        const j = await res.json();
        if (j && Array.isArray(j.entries) && j.entries.length) entry = j.entries[0];
      }
    } catch (err) {
      failDetail = String((err && err.message) || err || 'The watch log could not be read.');
    }
    if (!entry && !failDetail) {
      briefActivity = 'Running a watch pass — this takes about a minute';
      render();
      try {
        const res = await fetch('/watchlog/run', { method: 'POST' });
        if (!res.ok) failDetail = await briefDetail(res);
        else entry = await res.json();
      } catch (err) {
        failDetail = String((err && err.message) || err || 'The watch pass failed.');
      }
    }
    briefBusy = false;
    briefActivity = null;
    if (entry && typeof entry.summary_md === 'string' && entry.summary_md) {
      /* plain content string - survives the sanitizeMessages round-trip */
      state.messages.push({
        role: 'assistant',
        content: `### Watch log · ${FMT.utc(entry.created_at)}\n\n${entry.summary_md}`,
      });
      save();
      render();
      speakBrief(entry.spoken, entry.spoken_sig);
    } else {
      if (isNoKey(0, failDetail)) noteNoKey();
      else {
        state.messages.push({
          role: 'assistant',
          content: failDetail || 'The watch log is unavailable right now.',
        });
      }
      save();
      render();
    }
  }

  /* --- action executor ----------------------------------------------------- */

  /* Types a value into a field by DOM id, firing the same events as a
     keystroke so live filters, custom dropdowns and form logic all react. */
  function applySetField(a) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(String(a.field_id || ''))) return;
    const field = document.getElementById(a.field_id);
    if (!field) return;   /* wrong page - the model was told to navigate first */
    const tag = field.tagName;
    if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') return;
    try {
      field.value = String(a.value);
      field.dispatchEvent(new Event('input', { bubbles: true }));
      field.dispatchEvent(new Event('change', { bubbles: true }));
    } catch (_) { /* read-only/detached field */ }
  }

  /* Apply one action on the CURRENT page right now (mid-stream). Returns
     true when handled; false when it must wait for end-of-turn - navigate
     always, and dashboard-only actions when this page can't host them
     (those fold into a dashboard URL in runActions). */
  function applyActionLive(a) {
    const onDash = PAGE === 'dashboard';
    const ui = window.KesslerUI;
    switch (a.type) {
      case 'navigate':
        return false;
      case 'select_on_globe': {
        const n = Number(a.norad);
        if (!Number.isFinite(n)) return true;   /* nothing to defer */
        if (!(onDash && typeof KGlobe !== 'undefined')) return false;
        try { KGlobe.select(n); } catch (_) { /* engine fault */ }
        return true;
      }
      case 'set_table_filter': {
        if (!(onDash && ui && typeof ui.setTableFilter === 'function')) return false;
        try {
          ui.setTableFilter(
            a.risk !== undefined ? a.risk : null,
            a.search !== undefined ? a.search : null);
        } catch (_) { /* hook fault */ }
        return true;
      }
      case 'set_globe_filter': {
        if (!(onDash && ui && typeof ui.setGlobeFilter === 'function')) return false;
        try {
          ui.setGlobeFilter({
            types: a.types,
            riskOnly: a.risk_only !== undefined ? a.risk_only : a.riskOnly,
          });
        } catch (_) { /* hook fault */ }
        return true;
      }
      case 'set_globe_speed': {
        if (!(onDash && ui && typeof ui.setGlobeSpeed === 'function')) return false;
        try { ui.setGlobeSpeed(a.mult); } catch (_) { /* hook fault */ }
        return true;
      }
      case 'show_encounter': {
        const p = Number(a.primary_norad);
        const s = Number(a.secondary_norad);
        if (!(onDash && typeof KGlobe !== 'undefined' &&
              typeof KGlobe.showEncounter === 'function' &&
              Number.isFinite(p) && Number.isFinite(s))) return false;
        try {
          KGlobe.showEncounter({
            primaryNorad: p,
            secondaryNorad: s,
            tcaIso: typeof a.tca === 'string' ? a.tca : undefined,
            watch: true,
          });
        } catch (_) { /* engine fault */ }
        return true;
      }
      case 'set_field':
        applySetField(a);
        return true;
      case 'show_trajectory':
        openTrajectory(a.norad, a.partner_norad);
        return true;
      case 'open_simulator': {
        /* the encounter simulator loads on every page - no fold-in */
        const id = Number(a.conjunction_id);
        if (Number.isFinite(id) && typeof KSim !== 'undefined' &&
            typeof KSim.open === 'function') {
          try {
            KSim.open({ conjunctionId: id, applyPlan: a.apply_plan === true });
          } catch (_) { /* module fault */ }
        }
        return true;
      }
      default:
        return true; /* unknown action types are ignored (forward-compatible) */
    }
  }

  /* End-of-turn pass: everything applyActionLive couldn't do in place -
     navigations, plus dashboard-only actions folded into dashboard URL
     params. Also the whole path for the non-streaming fallback. */
  function runActions(actions) {
    let navUrl = null;      /* last valid navigate action wins, applied last */
    const folded = {};      /* URL params queued for a dashboard navigation */
    let foldNeeded = false; /* a globe/table action could not apply in place */

    for (const a of actions) {
      if (!a || typeof a !== 'object') continue;
      if (a.type === 'navigate') {
        const u = safeRelUrl(a.url);
        if (u) navUrl = u;
        continue;
      }
      if (applyActionLive(a)) continue;
      switch (a.type) {   /* not hostable here - fold into a dashboard URL */
        case 'select_on_globe':
          folded.sel = String(Number(a.norad));
          foldNeeded = true;
          break;
        case 'set_table_filter':
          if (a.risk !== undefined && a.risk !== null) folded.risk = String(a.risk);
          if (a.search !== undefined && a.search !== null) folded.search = String(a.search);
          foldNeeded = true;
          break;
        case 'show_encounter': {
          const id = Number(a.conjunction_id);
          if (Number.isFinite(id)) folded.conj = String(id);
          foldNeeded = true;
          break;
        }
        default:
          foldNeeded = true; /* no URL param exists - land on the dashboard */
          break;
      }
    }

    if (foldNeeded && !navUrl) navUrl = '/';
    if (!navUrl) return;
    let target = navUrl;
    if (Object.keys(folded).length) {
      try {
        const u = new URL(navUrl, window.location.origin);
        if (u.pathname === '/' || u.pathname === '/index.html') {
          for (const [k, v] of Object.entries(folded)) {
            if (!u.searchParams.has(k)) u.searchParams.set(k, v);
          }
          target = u.pathname + u.search + u.hash;
        }
      } catch (_) { /* malformed url - navigate as given */ }
    }
    window.location.assign(target);
  }

  /* --- trajectory modal ---------------------------------------------------- */

  let modal = null;
  let modalToken = 0; /* bumped on open/close to cancel stale async renders */

  function buildModal() {
    const root = document.createElement('div');
    root.className = 'kai-modal';
    root.hidden = true;
    root.innerHTML =
      '<div class="kai-modal__backdrop"></div>' +
      '<section class="kai-modal__panel" role="dialog" aria-modal="true" ' +
        'aria-label="Trajectory display">' +
        '<div class="kai-hd">' +
          '<span class="kai-title">Trajectory</span>' +
          '<button type="button" class="kai-hd-btn" data-act="close" ' +
            `aria-label="Close trajectory">${ICON_CLOSE}</button>` +
        '</div>' +
        '<div class="kai-modal__bd"></div>' +
      '</section>';
    document.body.appendChild(root);
    modal = {
      root,
      title: root.querySelector('.kai-title'),
      body: root.querySelector('.kai-modal__bd'),
    };
    root.querySelector('[data-act="close"]').addEventListener('click', closeModal);
    root.querySelector('.kai-modal__backdrop').addEventListener('click', closeModal);
    /* ESC is handled by the unified document keydown handler (modal wins). */
  }

  function closeModal() {
    if (!modal) return;
    modalToken++;
    modal.root.hidden = true;
    if (modal.prevFocus && modal.prevFocus.focus) {
      try { modal.prevFocus.focus(); } catch (_) { /* element gone */ }
    }
    modal.prevFocus = null;
  }

  async function fetchTrack(norad) {
    /* js/api.js is a frozen shared foundation without a track wrapper -
       mirror its error shape for the one extra endpoint (object.js idiom). */
    const res = await fetch(`/objects/${encodeURIComponent(String(norad))}/track`);
    if (!res.ok) throw new Error(`${res.status}: track unavailable`);
    return res.json();
  }

  let coastPromise = null;
  function fetchCoastlines() {
    if (!coastPromise) {
      coastPromise = fetch('/data/coastlines.json')
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
    }
    return coastPromise;
  }

  async function openTrajectory(norad, partnerNorad) {
    const n = Number(norad);
    if (!Number.isFinite(n)) return;
    if (!modal) buildModal();
    const token = ++modalToken;
    modal.prevFocus = document.activeElement;
    modal.root.hidden = false;
    const closeBtn = modal.root.querySelector('[data-act="close"]');
    if (closeBtn) closeBtn.focus();   /* aria-modal dialogs must take focus */
    modal.title.textContent = `Trajectory · ${n}`;
    if (typeof OrbitPlot === 'undefined' || typeof drawGroundTrack === 'undefined') {
      modal.body.innerHTML = '<div class="kai-modal-loading">' +
        'The trajectory display is unavailable on this page.</div>';
      return;
    }
    modal.body.innerHTML =
      '<div class="kai-modal-loading">Loading the trajectory…</div>';

    let obj;
    try {
      obj = await API.object(n);
    } catch (err) {
      if (token !== modalToken) return;
      modal.body.innerHTML = '<div class="kai-modal-loading">The trajectory could not be loaded (' +
        `${FMT.esc((err && err.message) || 'fetch failed')}).</div>`;
      return;
    }

    const pn = Number(partnerNorad);
    const wantPartner = partnerNorad !== undefined && partnerNorad !== null &&
      Number.isFinite(pn) && pn !== n;
    const settled = await Promise.allSettled([
      wantPartner ? API.object(pn) : Promise.resolve(null),
      fetchTrack(n),
      fetchCoastlines(),
    ]);
    if (token !== modalToken) return;
    const partner = settled[0].status === 'fulfilled' ? settled[0].value : null;
    const track = settled[1].status === 'fulfilled' ? settled[1].value : null;
    const coastlines = settled[2].status === 'fulfilled' ? settled[2].value : null;
    renderTrajectory(n, obj, partner, track, coastlines);
  }

  function renderTrajectory(norad, obj, partner, track, coastlines) {
    const name = String(obj.name || `NORAD ${norad}`);
    const partnerName = partner ? String(partner.name || '') : '';
    modal.title.textContent = partner
      ? `${name} and ${partnerName}`
      : `${name} · ${norad}`;

    const hasTrack = !!(track && Array.isArray(track.samples) && track.samples.length > 1);
    const legend = partner
      ? '<div class="chart-legend kai-legend">' +
        `<span class="key" style="--key-color: var(--k-data-1)">${FMT.esc(name)}</span>` +
        `<span class="key" style="--key-color: var(--k-data-2)">${FMT.esc(partnerName)}</span>` +
        '</div>'
      : '';
    let meta = '';
    if (hasTrack && track.subpoint &&
        Number.isFinite(track.subpoint.lat_deg) && Number.isFinite(track.subpoint.lon_deg)) {
      const sp = track.subpoint;
      const latTxt = `${Math.abs(sp.lat_deg).toFixed(1)}°${sp.lat_deg < 0 ? 'S' : 'N'}`;
      const lonTxt = `${Math.abs(sp.lon_deg).toFixed(1)}°${sp.lon_deg < 0 ? 'W' : 'E'}`;
      meta = `<div class="kai-modal-meta">Below the satellite: ${FMT.esc(latTxt)} ${FMT.esc(lonTxt)}` +
        ` · altitude ${FMT.esc(FMT.km(sp.alt_km))} · as of ${FMT.esc(FMT.utc(track.epoch))}</div>`;
    }

    modal.body.innerHTML =
      '<div class="kai-sec">Orbit <span>Earth-centered frame · drag to rotate</span></div>' +
      '<canvas class="kai-orbit-canvas"></canvas>' +
      legend +
      '<div class="kai-sec">Ground track <span>one orbit</span></div>' +
      (hasTrack
        ? `<canvas class="kai-track-canvas"></canvas>${meta}`
        : '<div class="empty">No ground track: propagation failed for this object.</div>');

    /* modal is visible and laid out - both modules read CSS size * DPR now */
    OrbitPlot.draw(modal.body.querySelector('.kai-orbit-canvas'), {
      primary: { label: obj.name, path: Array.isArray(obj.orbit_path) ? obj.orbit_path : [] },
      secondary: partner
        ? { label: partner.name, path: Array.isArray(partner.orbit_path) ? partner.orbit_path : [] }
        : null,
    });
    if (hasTrack) {
      drawGroundTrack(modal.body.querySelector('.kai-track-canvas'), track, {
        coastlines: coastlines || undefined,
        label: obj.name,
      });
    }
  }

  /* --- keyboard shortcuts (unified - modal ESC precedence) ----------------- */

  document.addEventListener('keydown', (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && !ev.altKey &&
        (ev.key === 'k' || ev.key === 'K')) {
      ev.preventDefault();
      setOpen(panel.hidden);
      return;
    }
    if (ev.key === 'Escape') {
      if (modal && !modal.root.hidden) { closeModal(); return; }
      if (!panel.hidden && !ev.defaultPrevented) setOpen(false);
      return;
    }
    if (ev.key === '/' && !ev.metaKey && !ev.ctrlKey && !ev.altKey && panel.hidden) {
      const t = ev.target;
      const inInput = !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
        t.tagName === 'SELECT' || t.isContentEditable);
      if (!inInput) {
        ev.preventDefault();
        setOpen(true);
      }
    }
  });

  /* --- events / init ------------------------------------------------------- */

  fab.addEventListener('click', () => setOpen(true));
  panel.querySelector('[data-act="close"]').addEventListener('click', () => setOpen(false));
  panel.querySelector('[data-act="clear"]').addEventListener('click', () => {
    if (stream) {
      stream.discard = true;
      stopStream();
    }
    stopBriefAudio();   /* the transcript the clip narrates is gone */
    state.messages = [];
    state.pendingError = null;
    state.notice = null;
    save();
    render();
    textEl.focus();
  });
  sendEl.addEventListener('click', () => {
    if (stream) stopStream();
    else send(textEl.value);
  });
  textEl.addEventListener('keydown', (ev) => {
    /* isComposing: Enter confirms an IME candidate, it must not send */
    if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      send(textEl.value);
    }
  });
  logEl.addEventListener('click', (ev) => {
    if (ev.target.closest('.kai-error__retry')) { send(null); return; } /* retry */
    if (ev.target.closest('[data-act="dismiss-notice"]')) {
      state.notice = null;
      save();
      render();
      textEl.focus();
      return;
    }
    const simChip = ev.target.closest('[data-sim]');
    if (simChip) {
      const id = Number(simChip.getAttribute('data-sim'));
      if (Number.isFinite(id) && typeof KSim !== 'undefined' && typeof KSim.open === 'function') {
        try {
          KSim.open({ conjunctionId: id, applyPlan: simChip.hasAttribute('data-plan') });
        } catch (_) { /* module fault */ }
      }
      return;
    }
    if (ev.target.closest('[data-brief]')) { shiftBrief(); return; }
    const qa = ev.target.closest('[data-qa]');
    if (qa) {
      const i = Number(qa.getAttribute('data-qa'));
      if (QUICK_ACTIONS[i]) send(QUICK_ACTIONS[i][1]);
    }
  });

  render();
  setOpen(state.open);
})();
