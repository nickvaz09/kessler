/* Kessler API client. All pages use these wrappers - never raw fetch(). */
'use strict';

const API = (() => {
  /* FastAPI 422 bodies carry `detail` as a list of validation-error objects -
     render the first one's msg + loc path instead of "[object Object]";
     any other non-string detail falls back to JSON. '' keeps statusText. */
  function detailText(detail) {
    if (detail == null) return '';
    if (typeof detail === 'string') return detail;
    const first = Array.isArray(detail) ? detail[0] : detail;
    if (first && typeof first === 'object' && typeof first.msg === 'string') {
      const loc = Array.isArray(first.loc) && first.loc.length
        ? ` (${first.loc.join('.')})` : '';
      return first.msg + loc;
    }
    try { return JSON.stringify(detail) || ''; } catch (_) { return String(detail); }
  }

  async function request(method, path, params) {
    const url = new URL(path, window.location.origin);
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
      }
    }
    const res = await fetch(url, { method });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = detailText((await res.json()).detail) || detail; } catch (_) { /* keep statusText */ }
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  return {
    status: () => request('GET', '/status'),
    objects: (params) => request('GET', '/objects', params),
    object: (noradId) => request('GET', `/objects/${noradId}`),
    conjunctions: (params) => request('GET', '/conjunctions', params),
    conjunction: (id) => request('GET', `/conjunctions/${id}`),
    risk: (conjunctionId) => request('GET', `/risk/${conjunctionId}`),
    refresh: (demo) => request('POST', '/ingest/refresh', demo ? { demo: 'true' } : undefined),
  };
})();
