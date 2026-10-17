/* Kessler custom dropdown - progressive enhancement of native <select.input>.
   The native select stays in the DOM (hidden) as the source of truth: value
   reads, programmatic writes and 'change' listeners all keep working. The
   visible control is a .kdd button + hairline menu matching the typeahead
   idiom. Auto-enhances every select.input at load and any added later
   (MutationObserver), so dynamically-created pickers are covered without
   coordination. Keyboard: Enter/Space opens, arrows navigate, Enter selects,
   Esc closes; click-outside closes. */
'use strict';

const KDropdown = (() => {
  let openMenu = null; /* the single open .kdd root, if any */

  function closeOpen() {
    if (!openMenu) return;
    openMenu.classList.remove('is-open');
    openMenu.querySelector('.kdd-menu').hidden = true;
    openMenu.querySelector('.kdd-btn').setAttribute('aria-expanded', 'false');
    openMenu = null;
  }

  document.addEventListener('pointerdown', (e) => {
    if (openMenu && !openMenu.contains(e.target)) closeOpen();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && openMenu) { closeOpen(); }
  });

  function labelOf(sel) {
    const opt = sel.options[sel.selectedIndex];
    return opt ? opt.textContent : '';
  }

  function rebuildMenu(root, sel) {
    const menu = root.querySelector('.kdd-menu');
    menu.innerHTML = '';
    for (let i = 0; i < sel.options.length; i++) {
      const opt = sel.options[i];
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'kdd-item' + (i === sel.selectedIndex ? ' is-active' : '');
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', i === sel.selectedIndex ? 'true' : 'false');
      item.textContent = opt.textContent;
      item.dataset.i = String(i);
      menu.appendChild(item);
    }
  }

  function enhance(sel) {
    if (!sel || sel.dataset.kdd === '1' || sel.tagName !== 'SELECT') return;
    sel.dataset.kdd = '1';

    const root = document.createElement('div');
    root.className = 'kdd';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'kdd-btn input';
    btn.setAttribute('aria-haspopup', 'listbox');
    btn.setAttribute('aria-expanded', 'false');
    if (sel.getAttribute('aria-label')) {
      btn.setAttribute('aria-label', sel.getAttribute('aria-label'));
    }
    btn.innerHTML = `<span class="kdd-label"></span><span class="kdd-caret">▾</span>`;
    const menu = document.createElement('div');
    menu.className = 'kdd-menu';
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;
    root.appendChild(btn);
    root.appendChild(menu);

    sel.after(root);
    sel.tabIndex = -1;
    sel.setAttribute('aria-hidden', 'true');
    sel.classList.add('kdd-native');
    sel.kddRoot = root;         /* lets owners find the visible control */
    root.hidden = sel.hidden;   /* owners toggle sel.hidden; mirror it */

    const syncLabel = () => {
      root.querySelector('.kdd-label').textContent = labelOf(sel);
    };
    rebuildMenu(root, sel);
    syncLabel();

    btn.addEventListener('click', () => {
      const isOpen = root === openMenu;
      closeOpen();
      if (!isOpen) {
        rebuildMenu(root, sel);
        menu.hidden = false;
        root.classList.add('is-open');
        btn.setAttribute('aria-expanded', 'true');
        openMenu = root;
      }
    });

    menu.addEventListener('click', (e) => {
      const item = e.target.closest('.kdd-item');
      if (!item) return;
      sel.selectedIndex = Number(item.dataset.i);
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      syncLabel();
      closeOpen();
      btn.focus();
    });

    btn.addEventListener('keydown', (e) => {
      const open = root === openMenu;
      if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && open) {
        e.preventDefault();
        const items = [...menu.querySelectorAll('.kdd-item')];
        const cur = items.findIndex((it) => it.classList.contains('is-active'));
        const next = Math.max(0, Math.min(items.length - 1,
          cur + (e.key === 'ArrowDown' ? 1 : -1)));
        items.forEach((it, i) => {
          it.classList.toggle('is-active', i === next);
          it.setAttribute('aria-selected', i === next ? 'true' : 'false');
        });
      } else if (e.key === 'Enter' && open) {
        e.preventDefault();
        const active = menu.querySelector('.kdd-item.is-active');
        if (active) active.click();
      } else if ((e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')
          && !open) {
        e.preventDefault();
        btn.click();
      }
    });

    /* keep the visible control in sync with programmatic changes */
    sel.addEventListener('change', syncLabel);
    new MutationObserver(() => {
      root.hidden = sel.hidden;
      rebuildMenu(root, sel);
      syncLabel();
    }).observe(sel, { childList: true, subtree: true, attributes: true });
  }

  function scan(rootNode) {
    const scope = rootNode && rootNode.querySelectorAll ? rootNode : document;
    scope.querySelectorAll('select.input').forEach(enhance);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => scan());
  } else {
    scan();
  }

  /* enhance selects added after load (e.g. dynamically-built pickers) */
  new MutationObserver((muts) => {
    for (const m of muts) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.matches && node.matches('select.input')) enhance(node);
        else if (node.querySelectorAll) scan(node);
      }
    }
  }).observe(document.documentElement, { childList: true, subtree: true });

  return { enhance, scan };
})();
