/* Local reading controls. No network requests, analytics, or remote storage. */
(() => {
  'use strict';
  const article = document.querySelector('#chapter-content');
  const toc = document.querySelector('#chapter-toc');
  if (article && toc) {
    const list = document.createElement('ol');
    article.querySelectorAll('h2').forEach((heading, i) => {
      if (!heading.id) heading.id = `section-${i + 1}`;
      const li = document.createElement('li');
      const a = document.createElement('a'); a.href = `#${heading.id}`;
      a.textContent = heading.textContent; li.append(a); list.append(li);
    });
    const disclosure = document.createElement('details');
    disclosure.open = window.matchMedia('(min-width: 901px)').matches;
    const summary = document.createElement('summary');
    summary.textContent = 'On this page';
    toc.querySelector('h2')?.remove();
    disclosure.append(summary, list); toc.append(disclosure);
  }
  document.querySelectorAll('table').forEach(table => {
    if (table.parentElement.classList.contains('table-wrap')) return;
    const wrap = document.createElement('div'); wrap.className = 'table-wrap';
    table.before(wrap); wrap.append(table);
  });
  const preference = (key, value) => { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (_) {} };
  for (const [id, cls] of [['theme-toggle', 'light'], ['text-toggle', 'large-text']]) {
    const button = document.getElementById(id); if (!button) continue;
    if (preference(`book-${cls}`) === 'true') document.body.classList.add(cls);
    button.setAttribute('aria-pressed', String(document.body.classList.contains(cls)));
    button.addEventListener('click', () => { const active = document.body.classList.toggle(cls); button.setAttribute('aria-pressed', String(active)); preference(`book-${cls}`, String(active)); });
  }
  const solutions = document.querySelector('#solutions-toggle');
  if (solutions) solutions.addEventListener('click', () => {
    const open = solutions.getAttribute('aria-pressed') !== 'true';
    document.querySelectorAll('article details').forEach(d => d.open = open);
    solutions.setAttribute('aria-pressed', String(open));
  });
  let beforePrintState = [];
  window.addEventListener('beforeprint', () => { beforePrintState = [...document.querySelectorAll('details')].map(d => [d, d.open]); beforePrintState.forEach(([d]) => d.open = true); });
  window.addEventListener('afterprint', () => beforePrintState.forEach(([d, open]) => d.open = open));
  document.querySelector('#print-page')?.addEventListener('click', () => window.print());
  document.querySelectorAll('pre').forEach(pre => {
    const code = pre.querySelector('code'); if (!code) return;
    const button = document.createElement('button'); button.type = 'button'; button.className = 'copy-code'; button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code to clipboard');
    button.addEventListener('click', async () => {
      const text = code.textContent; let copied = false;
      try { await navigator.clipboard.writeText(text); copied = true; } catch (_) {
        const field = document.createElement('textarea'); field.value = text; field.style.position = 'fixed'; field.style.left = '-9999px'; document.body.append(field); field.select();
        try { copied = document.execCommand('copy'); } catch (_) {} field.remove();
      }
      button.textContent = copied ? 'Copied' : 'Select code to copy';
      if (!copied) { const range = document.createRange(); range.selectNodeContents(code); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); }
      setTimeout(() => button.textContent = 'Copy', 2000);
    }); pre.prepend(button);
  });
  const search = document.getElementById('lecture-search');
  if (search) search.addEventListener('input', () => {
    const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean); let count = 0;
    document.querySelectorAll('[data-lecture]').forEach(item => { item.hidden = !terms.every(term => item.textContent.toLowerCase().includes(term)); if (!item.hidden) count++; });
    document.getElementById('search-status').textContent = `${count} recordings shown`;
  });
})();
