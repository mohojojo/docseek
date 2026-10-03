from __future__ import annotations

import base64
import logging
from contextlib import contextmanager


from .models import ANode, ElementRegistry, FullElement
from .proxy import browser_context, close_context, launch_browser, new_page, sync_playwright, wait_out_challenge

logger = logging.getLogger(__name__)

# Injected into the page to extract interactive elements from the live DOM
_EXTRACT_ELEMENTS_JS = """() => {
    const KEEP_ROLES = new Set([
        'button', 'link', 'textbox', 'checkbox', 'radio', 'combobox',
        'listbox', 'menuitem', 'tab', 'heading', 'dialog', 'banner',
        'navigation', 'main', 'alert'
    ]);

    const TAG_TO_ROLE = {
        A: 'link', BUTTON: 'button', SELECT: 'combobox', TEXTAREA: 'textbox',
        H1: 'heading', H2: 'heading', H3: 'heading',
        H4: 'heading', H5: 'heading', H6: 'heading',
        NAV: 'navigation', HEADER: 'banner', MAIN: 'main', DIALOG: 'dialog',
    };

    const INPUT_TYPE_TO_ROLE = {
        checkbox: 'checkbox', radio: 'radio',
        submit: 'button', button: 'button', reset: 'button',
    };

    function resolveRole(el) {
        const aria = el.getAttribute('role');
        if (aria) return aria;
        if (el.tagName === 'INPUT') {
            return INPUT_TYPE_TO_ROLE[(el.type || 'text').toLowerCase()] || 'textbox';
        }
        return TAG_TO_ROLE[el.tagName] || null;
    }

    function accessibleName(el) {
        const ariaLabel = (el.getAttribute('aria-label') || '').trim();
        if (ariaLabel) return ariaLabel;

        const labelledBy = (el.getAttribute('aria-labelledby') || '').trim();
        if (labelledBy) {
            for (const id of labelledBy.split(' ')) {
                const ref = document.getElementById(id);
                if (ref) {
                    const t = ref.textContent.trim();
                    if (t) return t;
                }
            }
        }

        const id = el.getAttribute('id');
        if (id) {
            const label = document.querySelector('label[for="' + CSS.escape(id) + '"]');
            if (label) {
                const t = label.textContent.trim();
                if (t) return t;
            }
        }

        const placeholder = (el.getAttribute('placeholder') || '').trim();
        if (placeholder) return placeholder;

        const title = (el.getAttribute('title') || '').trim();
        if (title) return title;

        const text = (el.textContent || '').replace(/\\s+/g, ' ').trim();
        if (text) return text;

        const value = (el.getAttribute('value') || '').trim();
        if (value) return value;

        // Icon-only links inside table cells: build name from row label + column header
        // so the LLM gets "Example Fund A - Monthly report" instead
        // of just the URL slug "example-fund-a".
        if (el.tagName === 'A' && el.getAttribute('href')) {
            const cell = el.closest('td, th');
            if (cell) {
                const row = cell.parentElement;
                const table = row ? row.closest('table') : null;
                if (table) {
                    const rowCells = Array.from(row.children).filter(c => c.tagName === 'TD' || c.tagName === 'TH');
                    const colIdx = rowCells.indexOf(cell);
                    const firstRow = table.querySelector('tr');
                    let colHeader = '';
                    if (firstRow && firstRow !== row) {
                        const headerCells = Array.from(firstRow.children).filter(c => c.tagName === 'TD' || c.tagName === 'TH');
                        colHeader = ((headerCells[colIdx] || {}).textContent || '').replace(/\\s+/g, ' ').trim();
                    }
                    const rowLabel = rowCells[0] && rowCells[0] !== cell
                        ? (rowCells[0].textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 60)
                        : '';
                    if (colHeader || rowLabel) {
                        return [rowLabel, colHeader].filter(Boolean).join(' – ');
                    }
                }
            }
        }

        // Icon / ::after-only labels often leave <a href> with no text; still need a
        // stable name for extraction + LLM matching (fund cards, icon-only CTAs).
        if (el.tagName === 'A' && el.getAttribute('href')) {
            try {
                const u = new URL(el.href, document.baseURI);
                const segments = u.pathname.split('/').filter(Boolean);
                const last = segments[segments.length - 1];
                if (last && last.length > 1) {
                    const slug = last.replace(/[-_]+/g, ' ').replace(/\\s+/g, ' ').trim();
                    if (slug) return slug;
                }
            } catch (e) {}
        }

        return null;
    }

    const selector = [
        'a[href]', 'button', 'input', 'select', 'textarea',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'nav', 'header', 'main', 'dialog', '[role]'
    ].join(', ');

    const seen = new Set();
    const results = [];
    let counter = 0;

    for (const el of document.querySelectorAll(selector)) {
        if (seen.has(el)) continue;
        seen.add(el);

        const role = resolveRole(el);
        if (!role || !KEEP_ROLES.has(role)) continue;

        const name = accessibleName(el);
        if (!name) continue;

        counter++;
        const id = String(counter);
        el.setAttribute('data-ml-id', id);

        // Collect all HTML attributes except our injected one
        const attrs = {};
        for (const attr of el.attributes) {
            if (attr.name !== 'data-ml-id') attrs[attr.name] = attr.value;
        }
        if (el.tagName === 'A' && el.href) {
            attrs.resolved_href = el.href;
        }
        if (el.tagName === 'SELECT') {
            const optTexts = Array.from(el.options)
                .slice(0, 20)
                .map(o => (o.text || '').replace(/\\s+/g, ' ').trim())
                .filter(Boolean);
            if (optTexts.length) attrs.select_options = optTexts.join(' | ');
        }

        results.push({
            role,
            name,
            ml_id: id,
            html_tag: el.tagName.toLowerCase(),
            attributes: attrs,
        });
    }

    // Detect elements with data-href (JS-navigation rows/cards that have no <a> tag).
    // Common in fund/product tables where clicking a <tr> navigates via JS.
    for (const el of document.querySelectorAll('[data-href]')) {
        if (seen.has(el)) continue;
        const href = (el.getAttribute('data-href') || '').trim();
        if (!href) continue;
        seen.add(el);

        // For table rows use first non-empty cell; otherwise fall back to full text.
        let elName = '';
        if (el.tagName === 'TR') {
            for (const cell of el.querySelectorAll('td, th')) {
                const t = (cell.textContent || '').replace(/\\s+/g, ' ').trim();
                if (t) { elName = t.slice(0, 120); break; }
            }
        }
        if (!elName) elName = (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
        if (!elName) continue;

        counter++;
        const id = String(counter);
        el.setAttribute('data-ml-id', id);

        let resolvedHref = href;
        try { resolvedHref = new URL(href, document.baseURI).href; } catch(e) {}

        results.push({
            role: 'link',
            name: elName,
            ml_id: id,
            html_tag: el.tagName.toLowerCase(),
            attributes: {resolved_href: resolvedHref, href},
        });
    }

    // Clickable table rows with no data-href (e.g. Ant Design tables where React handles
    // the click to expand an inline sub-table or navigate). Identified by tabindex + cursor:pointer.
    for (const tr of document.querySelectorAll('tr[tabindex]')) {
        if (seen.has(tr)) continue;
        if (window.getComputedStyle(tr).cursor !== 'pointer') continue;
        seen.add(tr);
        let trName = '';
        for (const td of tr.querySelectorAll('td')) {
            const t = (td.textContent || '').replace(/\\s+/g, ' ').trim();
            if (t && t.length > 1) { trName = t.slice(0, 120); break; }
        }
        if (!trName) continue;
        counter++;
        const id = String(counter);
        tr.setAttribute('data-ml-id', id);
        results.push({ role: 'button', name: trName, ml_id: id, html_tag: 'tr', attributes: {} });
    }

    // Pagination items (Ant Design li.ant-pagination-item, Bootstrap, and generic patterns).
    // Skip the currently-active page and ellipsis items.
    for (const li of document.querySelectorAll('li[class*="pagination"]')) {
        if (seen.has(li)) continue;
        const cls = li.className;
        if (cls.includes('active') || cls.includes('disabled')) continue;
        const text = (li.textContent || '').replace(/\\s+/g, ' ').trim();
        if (!text || text === '•••' || text === '…') continue;
        seen.add(li);
        counter++;
        const id = String(counter);
        li.setAttribute('data-ml-id', id);
        const isNext = cls.includes('next');
        const isPrev = cls.includes('prev');
        const label = isNext ? 'Next page' : isPrev ? 'Previous page' : `Page ${text}`;
        results.push({ role: 'button', name: label, ml_id: id, html_tag: 'li', attributes: {} });
    }

    return {
        elements: results,
        pageLang: document.documentElement.getAttribute('lang') || '',
    };
}"""

_GET_SELECT_OPTIONS_JS = """([maxSelects, maxOptions]) => {
    const selects = Array.from(document.querySelectorAll('select')).slice(0, maxSelects);
    return selects.map((sel, selectIndex) => {
        // Find the tab/accordion button that controls this select's panel (if hidden)
        let tabBtnId = null;
        if (sel.offsetHeight === 0) {
            let el = sel;
            while (el && el.tagName !== 'BODY') {
                const isPanel = (
                    el.getAttribute('role') === 'tabpanel' ||
                    el.classList.contains('e-n-tab-content') ||
                    el.classList.contains('tab-pane') ||
                    el.classList.contains('accordion-collapse')
                );
                if (isPanel && el.id) {
                    const btn = document.querySelector(
                        '[aria-controls="' + el.id + '"], [data-bs-target="#' + el.id + '"]'
                    );
                    if (btn && btn.id) { tabBtnId = btn.id; }
                    break;
                }
                el = el.parentElement;
            }
        }
        return {
            selectIndex,
            tabBtnId,
            originalIndex: sel.selectedIndex,
            label: (sel.getAttribute('aria-label') || sel.id || sel.name || '').trim(),
            options: Array.from(sel.options).slice(0, maxOptions).map((opt, i) => ({
                index: i,
                value: opt.value,
                text: (opt.textContent || '').trim(),
            })),
        };
    });
}"""

_TRIGGER_SELECT_CHANGE_JS = """([selectIndex, optionIndex]) => {
    const selects = document.querySelectorAll('select');
    const sel = selects[selectIndex];
    if (!sel) return false;
    sel.selectedIndex = optionIndex;
    // Use jQuery trigger when available (WordPress AJAX handlers use jQuery events).
    const jq = window.jQuery || window.$;
    if (jq) {
        jq(sel).trigger('change').trigger('input');
    } else {
        sel.dispatchEvent(new Event('change', {bubbles: true}));
        sel.dispatchEvent(new Event('input', {bubbles: true}));
    }
    return true;
}"""

_DISMISS_MODAL_JS = """() => {
    // Dismiss visible modal/disclaimer overlays by:
    //   1. Auto-filling any empty <select> elements inside them.
    //   2. Clicking the primary proceed button (multilingual text list, scoped to the modal).
    // Returns {filled: N, clickedText: string|null}.

    const MODAL_SELECTORS = [
        'dialog',
        '[role="dialog"]',
        '[aria-modal="true"]',
        '.modal-dialog', '.modal-content', '.modal-body',
        '[class*="disclaimer"]', '[class*="overlay"]', '[class*="popup"]',
        '[class*="modal"]',
        '[class*="welcome"]', '[class*="gate"]', '[class*="intro"]',
        '[class*="lightbox"]', '[class*="layer"]',
    ];

    // Button texts to click — broad list, safe because scoped to detected modals only.
    const PROCEED_RE = [
        /^ok$/i, /^yes$/i,
        /accept/i, /agree/i, /confirm/i, /continue/i, /proceed/i,
        /enter/i,
        /weiter/i, /fortfahren/i, /akzeptieren/i, /bestätig/i,
        /zustimm/i, /einverstand/i,
        /elfogad/i, /tovább/i,
        /accepter/i, /continuer/i, /valider/i,
        /aceptar/i, /accetto/i, /continua/i, /avanti/i,
        /accepteren/i, /doorgaan/i,
    ];

    const seen = new Set();
    let filled = 0;
    let clickedText = null;

    function tryDismiss(modal) {
        const style = window.getComputedStyle(modal);
        if (style.display === 'none' || style.visibility === 'hidden') return;
        const rect = modal.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;

        for (const sel of modal.querySelectorAll('select')) {
            if (sel.value && sel.value !== '') continue;
            for (let i = 0; i < sel.options.length; i++) {
                if (sel.options[i].value && sel.options[i].value !== '') {
                    sel.selectedIndex = i;
                    const jq = window.jQuery || window.$;
                    if (jq) jq(sel).trigger('change').trigger('input');
                    else {
                        sel.dispatchEvent(new Event('change', {bubbles: true}));
                        sel.dispatchEvent(new Event('input', {bubbles: true}));
                    }
                    filled++;
                    break;
                }
            }
        }

        if (!clickedText) {
            const btns = modal.querySelectorAll(
                'button, a[role="button"], input[type="button"], input[type="submit"]'
            );
            for (const btn of btns) {
                const text = (btn.textContent || btn.value || '').trim();
                // Real proceed buttons have short labels; skip long content blobs
                if (text.length > 120) continue;
                if (PROCEED_RE.some(re => re.test(text))) {
                    btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                    clickedText = text;
                    break;
                }
            }
        }
    }

    for (const modalSel of MODAL_SELECTORS) {
        for (const modal of document.querySelectorAll(modalSel)) {
            if (seen.has(modal)) continue;
            seen.add(modal);
            tryDismiss(modal);
        }
    }

    // Fallback: any large visible element (covers ≥30% viewport) with a proceed button.
    // Catches fullscreen gates that don't use standard modal classes.
    let fallbackEl = null;
    if (!clickedText) {
        const vw = window.innerWidth, vh = window.innerHeight;
        for (const el of document.querySelectorAll('div, section, aside, main')) {
            if (seen.has(el)) continue;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;
            const z = parseInt(style.zIndex) || 0;
            if (z < 10) continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < vw * 0.3 || rect.height < vh * 0.3) continue;
            seen.add(el);
            fallbackEl = el.tagName + '#' + (el.id || '') + '.' + Array.from(el.classList).join('.');
            tryDismiss(el);
            if (clickedText) break;
        }
    }

    return {filled, clickedText, fallbackEl};
}"""

_COOKIE_ACCEPT_JS = """() => {
    // Ordered by specificity: longer / more specific patterns first.
    const ACCEPT_TEXTS = [
        /mindennek.*megenged/i,
        /minden.*elfogad/i,
        /összes.*elfogad/i,
        /accept.*all/i,
        /all.*accept/i,
        /allow.*all/i,
        /allow.*cookie/i,
        /consent.*all/i,
        /elfogadom/i,
        /elfogad/i,
        /agree/i,
        /^ok$/i,
    ];
    const CONSENT_SELECTORS = [
        '[class*="cookie"] button',
        '[id*="cookie"] button',
        '[class*="consent"] button',
        '[class*="gdpr"] button',
        '[class*="cmp"] button',
        'button',
        'a[role="button"]',
        'input[type="button"]',
        'input[type="submit"]',
    ];
    const seen = new Set();
    for (const sel of CONSENT_SELECTORS) {
        for (const el of document.querySelectorAll(sel)) {
            if (seen.has(el)) continue;
            seen.add(el);
            const text = (el.textContent || el.value || '').trim();
            if (ACCEPT_TEXTS.some(pat => pat.test(text))) {
                el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                return text;
            }
        }
    }
    return null;
}"""

_COLLECT_LINKS_JS = """() => {
    const links = [];
    for (const a of document.querySelectorAll('a[href]')) {
        const href = a.href;
        if (!href || href.startsWith('javascript:')) continue;
        const name = (a.textContent || '').replace(/\\s+/g, ' ').trim()
                     || (a.getAttribute('aria-label') || '').trim()
                     || (a.getAttribute('title') || '').trim();
        links.push({href, name: name || href});
    }
    // Also collect [data-href] elements (JS-navigation rows/cards with no <a> tag).
    for (const el of document.querySelectorAll('[data-href]')) {
        const raw = (el.getAttribute('data-href') || '').trim();
        if (!raw) continue;
        let href = raw;
        try { href = new URL(raw, document.baseURI).href; } catch(e) {}
        if (href.startsWith('javascript:')) continue;
        let name = '';
        if (el.tagName === 'TR') {
            for (const cell of el.querySelectorAll('td, th')) {
                const t = (cell.textContent || '').replace(/\\s+/g, ' ').trim();
                if (t) { name = t.slice(0, 120); break; }
            }
        }
        if (!name) name = (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
        links.push({href, name: name || href});
    }
    return links;
}"""


_DETECT_CANVAS_PAGE_JS = """() => {
    // Flutter web fingerprint
    if (document.querySelector('flt-glass-pane')) return 'flutter';
    if (window._flutter || window.flutter_state) return 'flutter';

    // Generic: a canvas element covering ≥70% of the viewport
    const vw = window.innerWidth || 800;
    const vh = window.innerHeight || 600;
    for (const c of document.querySelectorAll('canvas')) {
        const r = c.getBoundingClientRect();
        if (r.width >= vw * 0.7 && r.height >= vh * 0.7) return 'canvas';
    }

    return null;
}"""


def detect_canvas_page(page) -> str | None:
    """Return 'flutter', 'canvas', or None when the page renders via canvas with no DOM elements."""
    try:
        return page.evaluate(_DETECT_CANVAS_PAGE_JS)
    except Exception as exc:
        logger.debug('[detect_canvas] evaluation failed: %s', exc)
        return None


def _build_tree_from_elements(elements: list[dict]) -> tuple[ANode, ElementRegistry]:
    """Convert raw element dicts (from JS extraction) into an ANode tree and registry."""
    registry: ElementRegistry = {}
    children: list[ANode] = []

    for el in elements:
        ml_id = el['ml_id']
        attrs = el.get('attributes', {})
        registry[ml_id] = FullElement(
            ml_id=ml_id,
            role=el['role'],
            name=el['name'],
            html_tag=el['html_tag'],
            attributes=attrs,
            url=attrs.get('resolved_href') or attrs.get('href') or None,
        )
        anode_attrs: dict[str, str] = {'ml_id': ml_id, 'html_tag': el['html_tag']}
        attrs = el.get('attributes', {})
        href = attrs.get('resolved_href') or attrs.get('href', '')
        if href:
            anode_attrs['href'] = href
        children.append(ANode(
            role=el['role'],
            name=el['name'],
            attributes=anode_attrs,
        ))

    return ANode(role='main', children=children), registry


def _expand_selects(
    page,
    initial_hrefs: set[str],
    next_ml_id: int,
    js_wait_ms: int,
    max_selects: int = 10,
    max_options: int = 50,
) -> list[dict]:
    """Iterate <select> options and return newly-revealed link elements."""
    try:
        select_infos: list[dict] = page.evaluate(_GET_SELECT_OPTIONS_JS, [max_selects, max_options])
    except Exception as exc:
        logger.warning('[expand_selects] failed to get select infos: %s', exc)
        return []

    logger.info('[expand_selects] found %d select(s)', len(select_infos))
    for si in select_infos:
        logger.info('[expand_selects] select #%d label=%r options=%d', si['selectIndex'], si.get('label', ''), len(si.get('options', [])))

    if not select_infos:
        return []

    new_elements: list[dict] = []
    seen_hrefs: set[str] = set(initial_hrefs)
    wait_ms = max(1000, js_wait_ms // 2)

    for sel_info in select_infos:
        select_idx: int = sel_info['selectIndex']
        original_idx: int = sel_info.get('originalIndex', 0)
        options: list[dict] = sel_info.get('options', [])
        sel_label: str = sel_info.get('label', f'select#{select_idx}')
        found_for_select = 0

        # Activate the tab containing this select if it is currently hidden.
        tab_btn_id: str | None = sel_info.get('tabBtnId')
        if tab_btn_id:
            try:
                loc = page.locator(f'#{tab_btn_id}')
                if loc.count() > 0:
                    loc.click(timeout=5000)
                    page.wait_for_timeout(1000)
                    logger.info('[expand_selects] activated tab #%s for select %r', tab_btn_id, sel_label)
            except Exception as exc:
                logger.debug('[expand_selects] tab activation failed for #%s: %s', tab_btn_id, exc)

        for opt in options:
            try:
                # Use Playwright's native select_option (fires trusted events, works
                # with jQuery / WordPress AJAX handlers that check event.isTrusted).
                # Fall back to JS dispatch if the locator approach fails.
                try:
                    page.locator('select').nth(select_idx).select_option(index=opt['index'], timeout=5000)
                except Exception:
                    page.evaluate(_TRIGGER_SELECT_CHANGE_JS, [select_idx, opt['index']])
                # Wait for AJAX-driven content. Some sites fire several sequential
                # fetch() requests - networkidle can fire between them, so we always
                # add a fixed post-networkidle wait to capture late DOM mutations.
                try:
                    page.wait_for_load_state('networkidle', timeout=wait_ms * 3)
                except Exception:
                    pass
                page.wait_for_timeout(js_wait_ms)
                links: list[dict] = page.evaluate(_COLLECT_LINKS_JS)
            except Exception as exc:
                logger.warning('[expand_selects] option %d of %r failed: %s', opt['index'], sel_label, exc)
                continue

            new_for_opt = 0
            for link in links:
                href: str = link.get('href', '')
                if not href or href in seen_hrefs:
                    continue
                seen_hrefs.add(href)
                next_ml_id += 1
                new_for_opt += 1
                new_elements.append({
                    'role': 'link',
                    'name': link.get('name') or href,
                    'ml_id': str(next_ml_id),
                    'html_tag': 'a',
                    'attributes': {
                        'href': href,
                        'resolved_href': href,
                    },
                })
            found_for_select += new_for_opt
            if new_for_opt:
                logger.info('[expand_selects] %r opt[%d]=%r → %d new link(s)', sel_label, opt['index'], opt.get('text', ''), new_for_opt)

        logger.info('[expand_selects] %r total new links: %d', sel_label, found_for_select)

        try:
            page.evaluate(_TRIGGER_SELECT_CHANGE_JS, [select_idx, original_idx])
        except Exception:
            pass

    logger.info('[expand_selects] grand total new elements: %d', len(new_elements))
    return new_elements


_FIND_HIDDEN_SELECT_TRIGGERS_JS = """() => {
    // For each hidden <select>, walk up the DOM to find its tab/accordion panel,
    // then return the ID of the button that controls it (so Python can click natively).
    const result = [];
    const seen = new Set();
    for (const sel of document.querySelectorAll('select')) {
        if (sel.offsetHeight > 0) continue; // already visible
        let el = sel;
        while (el && el.tagName !== 'BODY') {
            const isPanel = (
                el.getAttribute('role') === 'tabpanel' ||
                el.classList.contains('e-n-tab-content') ||
                el.classList.contains('tab-pane') ||
                el.classList.contains('accordion-collapse')
            );
            if (isPanel && el.id) {
                const btn = document.querySelector(
                    '[aria-controls="' + el.id + '"], [data-bs-target="#' + el.id + '"]'
                );
                if (btn && btn.id && !seen.has(btn.id)) {
                    seen.add(btn.id);
                    result.push({
                        btnId: btn.id,
                        label: btn.textContent.trim().slice(0, 50),
                    });
                }
                break;
            }
            el = el.parentElement;
        }
    }
    return result;
}"""


_PICK_ARIA_OPTION_JS = """(wanted) => {
    // Options of an open ARIA listbox/menu, as rendered by component libraries that do not use <select>.
    const norm = t => (t || '').replace(/\\s+/g, ' ').trim().toLowerCase();
    const target = norm(wanted);
    const nodes = [...document.querySelectorAll('[role=option], [role=listbox] li, [role=menu] [role=menuitem], ul[role] li')];
    const visible = nodes.filter(el => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden';
    });
    const pool = visible.length ? visible : nodes;
    let hit = pool.find(el => norm(el.textContent) === target)
           || pool.find(el => norm(el.textContent).includes(target));
    if (!hit) return {ok: false, options: pool.slice(0, 12).map(el => (el.textContent || '').trim().slice(0, 40))};
    hit.click();
    return {ok: true, text: (hit.textContent || '').trim().slice(0, 60)};
}"""


def select_option_anywhere(page, ml_id: str, value: str, wait_ms: int = 1000) -> str:
    """Choose `value` in the control `ml_id`, whether it is a <select> or an ARIA combobox.

    Playwright's select_option only drives real <select> elements. Component libraries render a
    button plus a hidden listbox instead, and select_option then waits for an element that will never
    become a select, so the call times out (report listings often sit behind exactly such a filter).
    """
    loc = page.locator(f'[data-ml-id="{ml_id}"]')
    if loc.count() == 0:
        return f'Element ml_id={ml_id} not found'

    tag = (loc.evaluate('el => el.tagName') or '').upper()
    if tag == 'SELECT':
        try:
            loc.select_option(label=value, timeout=5000)
        except Exception:
            loc.select_option(value=value, timeout=5000)
        return f'Selected "{value}" in <select> ml_id={ml_id}'

    # ARIA combobox: open it, then click the option that matches.
    try:
        loc.click(timeout=5000)
    except Exception:
        loc.dispatch_event('click')
    page.wait_for_timeout(min(wait_ms, 800))
    result = page.evaluate(_PICK_ARIA_OPTION_JS, value)
    if result.get('ok'):
        page.wait_for_timeout(min(wait_ms, 800))
        return f'Selected "{result.get("text", value)}" in combobox ml_id={ml_id}'
    offered = result.get('options') or []
    return (f'No option matching "{value}" in ml_id={ml_id}. '
            + (f'Options: {offered}' if offered else 'The control showed no options when opened.'))


def _try_dismiss_form_disclaimer(page, wait_ms: int = 500) -> bool:
    """Dismiss visible modal/gate overlays before the agent sees the page.

    Fills empty <select> elements and clicks the primary proceed button
    (Accept / Weiter / Continue / …), both scoped to detected modal elements.
    Falls back to a z-index+size heuristic for fullscreen gates without modal classes.
    Returns True if anything was interacted with.
    """
    try:
        result = page.evaluate(_DISMISS_MODAL_JS)
        filled = result.get('filled', 0) if isinstance(result, dict) else 0
        clicked = result.get('clickedText') if isinstance(result, dict) else None
        fallback_el = result.get('fallbackEl') if isinstance(result, dict) else None
        logger.debug('dismiss_modal: filled=%s clicked=%r fallbackEl=%r', filled, clicked, fallback_el)
        if filled or clicked:
            logger.info('[dismiss_modal] filled=%d selects, clicked=%r', filled, clicked)
            page.wait_for_timeout(wait_ms)
            return True
    except Exception as exc:
        logger.debug('[dismiss_modal] failed: %s', exc)
    return False


def apply_pre_interactions(page, steps: list[dict], wait_ms: int = 1000) -> None:
    """Execute a list of browser interaction steps before handing the page to the agent.

    Supported actions:
      {"action": "select", "selector": "...", "value": "..."}  - select by option value
      {"action": "select", "selector": "...", "label": "..."}  - select by visible text
      {"action": "click",  "selector": "..."}                   - click an element
      {"action": "check",  "selector": "..."}                   - check a checkbox (handles CSS-hidden inputs)
      {"action": "wait",   "ms": N}                             - pause N milliseconds

    Use to pre-fill listing-page filter forms (e.g. country + investor-type selects)
    that must be populated before content loads.
    """
    for step in steps:
        action = step.get('action', 'select')
        selector = step.get('selector', '')
        try:
            if action == 'select':
                loc = page.locator(selector).first
                if step.get('value') is not None:
                    loc.select_option(value=str(step['value']), timeout=5000)
                elif step.get('label') is not None:
                    loc.select_option(label=str(step['label']), timeout=5000)
                try:
                    page.wait_for_load_state('networkidle', timeout=wait_ms * 3)
                except Exception:
                    page.wait_for_timeout(wait_ms)
                logger.info('[pre_interactions] selected %r in %r', step.get('value') or step.get('label'), selector)
            elif action == 'click':
                page.locator(selector).first.click(timeout=5000)
                try:
                    page.wait_for_load_state('networkidle', timeout=wait_ms * 3)
                except Exception:
                    page.wait_for_timeout(wait_ms)
                logger.info('[pre_interactions] clicked %r', selector)
            elif action == 'check':
                loc = page.locator(selector).first
                loc.check(timeout=5000)
                page.wait_for_timeout(300)
                logger.info('[pre_interactions] checked %r', selector)
            elif action == 'wait':
                page.wait_for_timeout(step.get('ms', 1000))
        except Exception as exc:
            logger.debug('[pre_interactions] step %s failed: %s', step, exc)


def _try_accept_cookies(page, wait_ms: int = 1500) -> bool:
    """Click a cookie-consent accept button if one is present; return True if clicked."""
    try:
        clicked_text = page.evaluate(_COOKIE_ACCEPT_JS)
        logger.debug('dismiss_cookies: clicked=%r', clicked_text)
        if clicked_text:
            logger.info('[cookies] clicked accept button: %r', clicked_text)
            try:
                page.wait_for_load_state('networkidle', timeout=wait_ms * 2)
            except Exception:
                page.wait_for_timeout(wait_ms)
            return True
    except Exception as exc:
        logger.debug('[cookies] accept attempt failed: %s', exc)
    return False


def capture_screenshot(page, max_width: int | None = 800) -> str | None:
    """Capture a viewport PNG screenshot and return it as a base64 string, or None on failure.

    Pass max_width=None to skip resizing - required for canvas pages so that the pixel
    coordinates in the image match the coordinates used by page.mouse.click(x, y).
    """
    try:
        if max_width is not None:
            original_viewport = page.viewport_size  # {'width': N, 'height': N} or None
            if original_viewport and original_viewport.get('width', 0) > max_width:
                new_height = round(original_viewport['height'] * max_width / original_viewport['width'])
                try:
                    page.set_viewport_size({'width': max_width, 'height': new_height})
                    screenshot_bytes = page.screenshot(type='png', full_page=False)
                finally:
                    try:
                        page.set_viewport_size(original_viewport)
                    except Exception as exc:
                        logger.debug('[screenshot] viewport restore failed: %s', exc)
            else:
                screenshot_bytes = page.screenshot(type='png', full_page=False)
        else:
            screenshot_bytes = page.screenshot(type='png', full_page=False)
        return base64.b64encode(screenshot_bytes).decode('ascii')
    except Exception as exc:
        logger.warning('[screenshot] capture failed: %s', exc)
        return None


def _reveal_hidden_selects(page, wait_ms: int = 1500) -> list[str]:
    """Find tab/accordion buttons that hide selects, click them with native Playwright events.

    Returns labels of buttons successfully clicked (for logging).
    """
    try:
        triggers: list[dict] = page.evaluate(_FIND_HIDDEN_SELECT_TRIGGERS_JS)
    except Exception as exc:
        logger.debug('[reveal_selects] find triggers failed: %s', exc)
        return []

    clicked: list[str] = []
    for t in triggers:
        btn_id = t.get('btnId', '')
        label = t.get('label', btn_id)
        if not btn_id:
            continue
        try:
            loc = page.locator(f'#{btn_id}')
            if loc.count() > 0:
                loc.click(timeout=5000)
                clicked.append(label)
                logger.info('[reveal_selects] clicked tab button %r (#%s)', label, btn_id)
        except Exception as exc:
            logger.debug('[reveal_selects] failed to click #%s: %s', btn_id, exc)

    if clicked:
        page.wait_for_timeout(1000)  # CSS animation settle - no network event to wait for

    return clicked


_DEFAULT_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)

_TRACKING_SCRIPT_HOSTS = frozenset({
    # Pure analytics - data collection only, never gate content rendering
    'www.google-analytics.com', 'google-analytics.com', 'analytics.google.com',
    'www.googletagservices.com',
    'www.googleadservices.com', 'doubleclick.net',
    'connect.facebook.net',
    'static.hotjar.com', 'script.hotjar.com',
    # NOTE: TMS / consent platforms (tagcommander, trustcommander, googletagmanager,
    # cookielaw, cookiebot, onetrust) are intentionally excluded - they often act as
    # consent gates and blocking them prevents content from rendering.
})


def _make_cross_origin_script_blocker(page_url: str):  # noqa: ARG001 - kept for API compat
    """Return a route handler that aborts known tracking/analytics scripts.

    Only blocks scripts from well-known third-party tracking hosts so that
    sites whose JS bundles are served from a CDN (different domain) still load.
    """
    def _handler(route, request) -> None:
        from urllib.parse import urlparse
        if request.resource_type == 'script':
            host = urlparse(request.url).netloc
            if host in _TRACKING_SCRIPT_HOSTS:
                route.abort()
                return
        route.continue_()

    return _handler


@contextmanager
def open_page(
    url: str,
    *,
    timeout: int = 60_000,
    user_agent: str = _DEFAULT_USER_AGENT,
    js_wait_ms: int = 1500,
    wait_until: str = 'domcontentloaded',
    accept_downloads: bool = False,
    headless: bool = True,
    storage_state: dict | None = None,
):
    """Context manager: launch browser, load URL, yield the live Playwright page.

    Keeps the browser open for the duration of the with-block - use this when you
    need to interact with the page (click, fill) before or between extractions.
    Set headless=False to watch the browser visually (useful for debugging).
    Pass storage_state (from context.storage_state()) to restore cookies/localStorage
    from a previous visit so sites don't re-show cookie banners or filter forms.
    """
    with sync_playwright() as p:
        browser = launch_browser(p, headless)
        ctx = browser_context(
            browser,
            user_agent=user_agent,
            accept_downloads=accept_downloads,
            storage_state=storage_state,
        )
        page = ctx.new_page()
        if headless:
            page.route('**/*', _make_cross_origin_script_blocker(url))
        page.goto(url, wait_until=wait_until, timeout=timeout)
        wait_out_challenge(page)
        page.wait_for_timeout(js_wait_ms)
        try:
            yield page
        finally:
            close_context(ctx)


def extract_tree_from_page(
    page,
    *,
    roles: frozenset[str] | None = None,
    js_wait_ms: int = 0,
) -> tuple[ANode, ElementRegistry, str | None]:
    """Extract the accessibility tree from an already-loaded Playwright page.

    Call after open_page() or after executing browser interactions.
    Pass js_wait_ms to allow JS rendering time before extracting.
    """
    if js_wait_ms:
        page.wait_for_timeout(js_wait_ms)
    try:
        js_result: dict = page.evaluate(_EXTRACT_ELEMENTS_JS)
    except Exception as exc:
        if 'context was destroyed' in str(exc).lower() or 'execution context' in str(exc).lower():
            page.wait_for_load_state('domcontentloaded')
            page.wait_for_timeout(500)
            js_result = page.evaluate(_EXTRACT_ELEMENTS_JS)
        else:
            raise
    elements: list[dict] = js_result['elements']
    page_lang: str | None = js_result.get('pageLang', '').strip().lower() or None
    if roles is not None:
        elements = [el for el in elements if el['role'] in roles]
    tree, registry = _build_tree_from_elements(elements)
    return tree, registry, page_lang


_SEARCH_INPUT_SELECTORS = [
    'input[type="search"]',
    'input[name*="search" i]',
    'input[placeholder*="search" i]',
    'input[placeholder*="keres" i]',
    'input[placeholder*="suche" i]',
    'form[role="search"] input[type="text"]',
    'form[role="search"] input:not([type])',
]


def search_in_page(
    page,
    query: str,
    wait_ms: int = 2000,
) -> tuple[dict | None, str]:
    """Fill the site search box with query and submit.

    Returns (new_registry, result_text). new_registry is None when no search input found.
    """
    for selector in _SEARCH_INPUT_SELECTORS:
        try:
            loc = page.locator(selector).first
            if loc.count() > 0 and loc.is_visible(timeout=1000):
                loc.fill(query)
                loc.press('Enter')
                page.wait_for_timeout(wait_ms)
                try:
                    page.wait_for_load_state('networkidle', timeout=wait_ms * 2)
                except Exception:
                    pass
                _, registry, _ = extract_tree_from_page(page)
                return registry, f'Search submitted: "{query}". Page re-extracted with {len(registry)} elements.'
        except Exception:
            continue
    return None, 'No search input found on this page.'


def scroll_page_to_load(
    page,
    wait_ms: int = 2000,
) -> tuple[dict | None, int, str]:
    """Scroll page bottom repeatedly to trigger lazy-loaded content.

    Returns (new_registry, new_element_delta, result_text).
    new_registry is None when scroll changes nothing.
    """
    try:
        _, initial_registry, _ = extract_tree_from_page(page)
        initial_count = len(initial_registry)
        prev_height = page.evaluate('document.body.scrollHeight')

        changed = False
        for _ in range(5):
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(wait_ms)
            new_height = page.evaluate('document.body.scrollHeight')
            if new_height == prev_height:
                break
            prev_height = new_height
            changed = True

        if not changed:
            return None, 0, 'No new content loaded after scrolling (page height unchanged).'

        _, registry, _ = extract_tree_from_page(page)
        delta = len(registry) - initial_count
        return registry, delta, (
            f'Scrolled to load more content. Page re-extracted with {len(registry)} elements '
            f'(+{delta} new).'
        )
    except Exception as exc:
        return None, 0, f'Scroll failed: {exc}'


def fetch_and_build_tree(
    url: str,
    *,
    roles: frozenset[str] | None = None,
    timeout: int = 60_000,
    user_agent: str = _DEFAULT_USER_AGENT,
    js_wait_ms: int = 1500,
    expand_selects: bool = False,
    max_selects: int = 10,
    max_options: int = 50,
    wait_until: str = 'domcontentloaded',
    headless: bool = True,
) -> tuple[ANode, ElementRegistry, str | None]:
    """Launch a headless browser, load the URL, and extract interactive elements.

    Returns the ANode tree (for Claude), an ElementRegistry (for result enrichment),
    and the page language string from <html lang="..."> (None if absent or empty).
    Raise js_wait_ms for SPAs that render navigation/content after the load event.
    Set expand_selects=True to trigger each <select> option and collect revealed links.
    Set headless=False to watch the browser visually (useful for debugging).
    """
    with sync_playwright() as p:
        browser = launch_browser(p, headless)
        page = new_page(browser, user_agent)
        if headless:
            page.route('**/*', _make_cross_origin_script_blocker(url))
        page.goto(url, wait_until=wait_until, timeout=timeout)
        wait_out_challenge(page)
        page.wait_for_timeout(js_wait_ms)

        try:
            js_result: dict = page.evaluate(_EXTRACT_ELEMENTS_JS)
        except Exception as exc:
            if 'context was destroyed' in str(exc).lower() or 'execution context' in str(exc).lower():
                page.wait_for_load_state('domcontentloaded', timeout=timeout)
                page.wait_for_timeout(js_wait_ms)
                js_result = page.evaluate(_EXTRACT_ELEMENTS_JS)
            else:
                raise

        elements: list[dict] = js_result['elements']

        _try_dismiss_form_disclaimer(page)
        _try_accept_cookies(page, wait_ms=js_wait_ms)

        if expand_selects:
            initial_hrefs = {
                el['attributes'].get('resolved_href') or el['attributes'].get('href', '')
                for el in elements
                if el.get('attributes') and (el['attributes'].get('resolved_href') or el['attributes'].get('href'))
            }
            next_ml_id = max((int(el['ml_id']) for el in elements), default=0)
            logger.info('[fetch] expand_selects=True for %s (initial_hrefs=%d)', url, len(initial_hrefs))
            extra = _expand_selects(page, initial_hrefs, next_ml_id, js_wait_ms, max_selects=max_selects, max_options=max_options)
            # Put expanded elements first so they survive any max_elements cap downstream.
            elements = extra + elements

        browser.close()

    page_lang: str | None = js_result.get('pageLang', '').strip().lower() or None

    if roles is not None:
        elements = [el for el in elements if el['role'] in roles]

    tree, registry = _build_tree_from_elements(elements)
    return tree, registry, page_lang
