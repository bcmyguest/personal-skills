#!/usr/bin/env node
// Turn a short JSON spec into one offline HTML page.
//
//   node page.mjs SPEC.json [-o OUT.html]
//
// The spec is { "title", "point"?, "blocks": [ ... ] }. Block kinds:
//   tree       { "text" }          indented names, two spaces per level; "name # note" puts the note on the node
//   components { "text" }          indented <Component> (path) lines, hooks on their own lines
//   topology   { "rows" }          each row is a layer: parallel items side by side, color is the layer
//   pre        { "text" }          pseudocode, monospaced
//   diff       { "text" }          lines starting with + or - are the change
//   columns    { "items": [ { "title", "text" } ] }
//   screen     { "regions": [ { "name", "note"? } ] }
//   mermaid    { "source" }        checked with diagram.mjs before the page is written
// Each block may also have "title". No network, no navigation. A mermaid block
// inlines Mermaid's own browser build so the file still opens offline.

import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readMermaidJs } from './diagram.mjs';

const KINDS = new Set(['tree', 'components', 'topology', 'pre', 'diff', 'columns', 'screen', 'mermaid']);

export function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function problems(spec) {
  const out = [];
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) return ['spec must be an object'];
  if (typeof spec.title !== 'string' || !spec.title.trim()) out.push('title is required');
  if (spec.point !== undefined && typeof spec.point !== 'string') out.push('point must be a string');
  if (!Array.isArray(spec.blocks) || spec.blocks.length === 0) out.push('blocks must be a non-empty array');
  else spec.blocks.forEach((b, i) => out.push(...blockProblems(b, i)));
  return out;
}

function blockProblems(b, i) {
  const at = `blocks[${i}]`;
  if (!b || typeof b !== 'object') return [`${at} must be an object`];
  if (!KINDS.has(b.kind)) return [`${at}.kind must be one of ${[...KINDS].join(', ')}`];
  if (b.title !== undefined && typeof b.title !== 'string') return [`${at}.title must be a string`];
  if (b.kind === 'tree' || b.kind === 'components' || b.kind === 'pre' || b.kind === 'diff') {
    if (typeof b.text !== 'string' || !b.text.trim()) return [`${at}.text is required`];
  }
  if (b.kind === 'topology') {
    if (!Array.isArray(b.rows) || b.rows.length === 0) return [`${at}.rows is required`];
    return b.rows.flatMap((row, j) => {
      if (!row || !Array.isArray(row.items) || row.items.length === 0) return [`${at}.rows[${j}].items is required`];
      return row.items.flatMap((item, k) => {
        if (!item || typeof item.name !== 'string' || !item.name.trim()) return [`${at}.rows[${j}].items[${k}].name is required`];
        if (item.note !== undefined && typeof item.note !== 'string') return [`${at}.rows[${j}].items[${k}].note must be a string`];
        return [];
      });
    });
  }
  if (b.kind === 'mermaid' && (typeof b.source !== 'string' || !b.source.trim())) return [`${at}.source is required`];
  if (b.kind === 'columns') {
    if (!Array.isArray(b.items) || b.items.length < 2) return [`${at}.items needs at least two`];
    return b.items.flatMap((item, j) => {
      if (!item || typeof item.title !== 'string' || typeof item.text !== 'string') {
        return [`${at}.items[${j}] needs title and text`];
      }
      return [];
    });
  }
  if (b.kind === 'screen') {
    if (!Array.isArray(b.regions) || b.regions.length === 0) return [`${at}.regions is required`];
    return b.regions.flatMap((r, j) => {
      if (!r || typeof r.name !== 'string' || !r.name.trim()) return [`${at}.regions[${j}].name is required`];
      if (r.note !== undefined && typeof r.note !== 'string') return [`${at}.regions[${j}].note must be a string`];
      return [];
    });
  }
  return [];
}

function parseTree(text) {
  const root = { children: [] };
  const stack = [{ depth: -1, node: root }];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const depth = line.match(/^ */)[0].length;
    const node = { label: line.trim(), children: [] };
    while (stack[stack.length - 1].depth >= depth) stack.pop();
    stack[stack.length - 1].node.children.push(node);
    stack.push({ depth, node });
  }
  return root.children;
}

function splitNote(label) {
  const m = label.match(/^(.*?)\s+#\s+(.+)$/);
  return m ? { name: m[1], note: m[2] } : { name: label, note: '' };
}

function renderNodes(nodes) {
  if (nodes.length === 0) return '';
  const items = nodes.map((n) => {
    const { name, note } = splitNote(n.label);
    const aside = note ? `<span class="note">${escapeHtml(note)}</span>` : '';
    return `<li><span class="name">${escapeHtml(name)}</span>${aside}${renderNodes(n.children)}</li>`;
  }).join('');
  return `<ul>${items}</ul>`;
}

function renderComponents(nodes) {
  if (nodes.length === 0) return '';
  const items = nodes.map((n) => {
    const comp = n.label.match(/^<([^>]+)>\s*(?:\(([^)]+)\))?$/);
    const inner = renderComponents(n.children);
    if (!comp) return `<li class="hook">${escapeHtml(n.label)}${inner}</li>`;
    const path = comp[2] ? `<span class="path">${escapeHtml(comp[2])}</span>` : '';
    return `<li class="box"><strong>${escapeHtml(comp[1])}</strong>${path}${inner}</li>`;
  }).join('');
  return `<ul>${items}</ul>`;
}

function renderTopology(rows) {
  const legend = rows.length > 1 ? '<p class="legend">Color is the layer.</p>' : '';
  const body = rows.map((row, i) => {
    const label = row.label ? `<p class="rowlabel">${escapeHtml(row.label)}</p>` : '';
    const items = row.items.map((item) => {
      const note = item.note ? `<p>${escapeHtml(item.note)}</p>` : '';
      return `<article class="t${i % 3}"><strong>${escapeHtml(item.name)}</strong>${note}</article>`;
    }).join('');
    return `<div class="topo-row">${label}<div class="topo-items">${items}</div></div>`;
  }).join('');
  return `<div class="topo">${legend}${body}</div>`;
}

function renderDiff(text) {
  return text.split('\n').map((line) => {
    const cls = line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : 'ctx';
    return `<div class="${cls}">${escapeHtml(line) || '&nbsp;'}</div>`;
  }).join('');
}

function renderBlock(b) {
  const heading = b.title ? `<h2>${escapeHtml(b.title)}</h2>` : '';
  let body = '';
  if (b.kind === 'tree') body = `<div class="tree">${renderNodes(parseTree(b.text))}</div>`;
  else if (b.kind === 'components') body = `<div class="comp">${renderComponents(parseTree(b.text))}</div>`;
  else if (b.kind === 'topology') body = renderTopology(b.rows);
  else if (b.kind === 'pre') body = `<pre>${escapeHtml(b.text)}</pre>`;
  else if (b.kind === 'diff') body = `<div class="diff">${renderDiff(b.text)}</div>`;
  else if (b.kind === 'columns') {
    const cols = b.items.map((item) => `<section><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.text)}</p></section>`).join('');
    body = `<div class="cols">${cols}</div>`;
  } else if (b.kind === 'screen') {
    const regions = b.regions.map((r) => {
      const note = r.note ? `<p>${escapeHtml(r.note)}</p>` : '';
      return `<div class="region"><strong>${escapeHtml(r.name)}</strong>${note}</div>`;
    }).join('');
    body = `<div class="screen">${regions}</div>`;
  } else body = `<figure class="diagram"><pre class="mermaid">${escapeHtml(b.source)}</pre>
<div class="srcwrap"><button type="button">Copy</button><pre class="source">${escapeHtml(b.source)}</pre></div></figure>`;
  return `<section class="block">${heading}${body}</section>`;
}

const CSS = `
:root {
  --bg: #f3efe6;
  --ink: #1c1915;
  --muted: #6f675e;
  --card: #fffcf7;
  --fill: #efeae0;
  --line: #e3dcd0;
  --add: #e5f3e8;
  --del: #f8e6e2;
  --t0: #ffffff;
  --t1: #e5f3e8;
  --t2: #f3e6c8;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161412;
    --ink: #f4efe6;
    --muted: #b3a89c;
    --card: #221f1b;
    --fill: #2b2723;
    --line: #3c362f;
    --add: #1d3326;
    --del: #3c2420;
    --t0: #2a2622;
    --t1: #243028;
    --t2: #3a3020;
  }
}
body {
  font: 17px/1.5 ui-sans-serif, system-ui, sans-serif;
  margin: 0 auto; max-width: 820px;
  padding: 56px 28px 96px;
  background: var(--bg); color: var(--ink);
}
h1 {
  font-family: ui-serif, Georgia, Palatino, serif;
  font-size: 2.35rem; font-weight: 500; letter-spacing: -0.03em;
  line-height: 1.12; margin: 0 0 12px;
}
.point { margin: 0 0 40px; max-width: 38rem; color: var(--muted); font-size: 1.05rem; }
h2 {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 14px;
}
h3 { font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; }
.block { margin: 0 0 40px; }
.tree ul { list-style: none; margin: 0; padding: 2px 0 2px 18px; border-left: 1px solid var(--line); }
.tree > ul { padding-left: 0; border-left: 0; }
.tree li { margin: 0; padding: 7px 0 7px 2px; }
.tree .name { font-weight: 600; }
.tree .note { color: var(--muted); margin-left: 8px; }
.comp ul { list-style: none; margin: 8px 0 0; padding: 0; display: grid; gap: 8px; }
.comp > ul { margin-top: 0; }
.comp .box { border: 1.5px solid var(--ink); border-radius: 14px; padding: 10px 12px; background: var(--card); }
.comp .box > strong { display: block; }
.comp .path, .comp .hook { color: var(--muted); font-size: 0.85rem; font-weight: 400; }
.comp .path { display: block; margin-top: 2px; }
.topo { display: grid; gap: 14px; }
.legend, .rowlabel { margin: 0 0 8px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.topo-items { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
.topo article { border: 1.5px solid var(--ink); border-radius: 14px; padding: 12px 14px; }
.topo article p { margin: 4px 0 0; color: var(--muted); }
.topo .t0 { background: var(--t0); }
.topo .t1 { background: var(--t1); }
.topo .t2 { background: var(--t2); }
pre, .diff {
  font: 13.5px/1.55 ui-monospace, monospace;
  margin: 0; white-space: pre-wrap;
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px 18px;
}
.diff .add, .diff .del, .diff .ctx { padding: 1px 8px; border-radius: 6px; }
.diff .add { background: var(--add); }
.diff .del { background: var(--del); }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.cols section { background: var(--card); border-radius: 16px; padding: 18px 20px; }
.cols p, .region p { margin: 0; white-space: pre-wrap; color: var(--muted); }
.region strong { display: block; font-weight: 600; margin-bottom: 2px; }
.screen {
  display: grid; gap: 8px; max-width: 360px;
  background: var(--card); border-radius: 28px; padding: 14px;
  box-shadow: 0 1px 0 var(--line), 0 18px 40px color-mix(in srgb, var(--ink) 6%, transparent);
}
.region { background: var(--fill); border-radius: 16px; padding: 14px 16px; }
.mermaid { background: var(--card); border-radius: 16px; padding: 12px; }
figure.diagram { margin: 0; }
.srcwrap { margin-top: 12px; }
.srcwrap button {
  font: 600 0.72rem/1 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.08em; text-transform: uppercase;
  background: var(--ink); color: var(--bg);
  border: 0; border-radius: 999px; padding: 7px 12px; cursor: pointer;
}
.srcwrap .source { margin-top: 8px; }
`;

export function renderPage(spec, mermaidJs = '') {
  const blocks = spec.blocks.map(renderBlock).join('\n');
  const point = spec.point ? `<p class="point">${escapeHtml(spec.point)}</p>` : '';
  const js = mermaidJs ? `<script>${mermaidJs.replace(/<\/script/gi, '<\\/script')}</script>
<script>
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = dark ? '#f4efe6' : '#1a1612';
const paper = dark ? '#161412' : '#ffffff';
mermaid.initialize({
  startOnLoad: true,
  securityLevel: 'strict',
  theme: 'base',
  themeVariables: {
    darkMode: dark,
    background: dark ? '#221f1b' : '#fffcf7',
    primaryColor: paper, primaryTextColor: ink, primaryBorderColor: ink,
    secondaryColor: dark ? '#243028' : '#e5f3e8',
    tertiaryColor: dark ? '#221f1b' : '#fffcf7',
    lineColor: ink, textColor: ink,
    actorBkg: paper, actorBorder: ink, actorTextColor: ink, actorLineColor: ink,
    signalColor: ink, signalTextColor: ink,
    noteBkgColor: dark ? '#243028' : '#e5f3e8', noteTextColor: ink, noteBorderColor: ink,
    fontFamily: 'ui-sans-serif, system-ui, sans-serif'
  },
  themeCSS: '.actor-line, .messageLine0, .messageLine1, .flowchart-link, .edgePath .path { stroke: ' + ink + '; stroke-width: 1.75px; } .actor, .node rect, .node polygon, .node circle { stroke: ' + ink + '; stroke-width: 1.75px; }'
});
document.querySelectorAll('.srcwrap button').forEach((btn) => {
  btn.addEventListener('click', () => {
    const text = btn.nextElementSibling.textContent;
    const mark = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); };
    const fallback = () => {
      const area = document.createElement('textarea');
      area.value = text;
      document.body.append(area);
      area.select();
      document.execCommand('copy');
      area.remove();
      mark();
    };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(mark, fallback);
    else fallback();
  });
});
</script>` : '';
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(spec.title)}</title>
<style>${CSS}</style></head><body>
<h1>${escapeHtml(spec.title)}</h1>
${point}
${blocks}
${js}
</body></html>
`;
}

function checkMermaid(spec) {
  const sources = spec.blocks.filter((b) => b.kind === 'mermaid').map((b) => b.source);
  if (sources.length === 0) return '';
  const dir = mkdtempSync(join(tmpdir(), 'sketch-it-'));
  const md = join(dir, 'blocks.md');
  writeFileSync(md, sources.map((s) => '```mermaid\n' + s + '\n```').join('\n\n'));
  const script = fileURLToPath(new URL('./diagram.mjs', import.meta.url));
  try {
    execFileSync('node', [script, 'check', md], { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
  } catch (e) {
    process.stderr.write(e.stdout || e.stderr || String(e));
    return null;
  }
  return readMermaidJs();
}

function main() {
  const argv = process.argv.slice(2);
  let out = null;
  const files = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '-o' || argv[i] === '--out') out = argv[++i];
    else files.push(argv[i]);
  }
  if (files.length !== 1) {
    process.stderr.write('usage: page.mjs SPEC.json [-o OUT.html]\n');
    return 2;
  }
  let spec;
  try {
    spec = JSON.parse(readFileSync(files[0], 'utf8'));
  } catch (e) {
    process.stderr.write(`spec: ${e.message}\n`);
    return 1;
  }
  const errs = problems(spec);
  if (errs.length) {
    process.stderr.write(errs.map((m) => `spec: ${m}\n`).join(''));
    return 1;
  }
  const mermaidJs = checkMermaid(spec);
  if (mermaidJs === null) return 1;
  const dest = resolve(out || files[0].replace(/\.json$/, '') + '.html');
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, renderPage(spec, mermaidJs));
  process.stdout.write(`wrote ${dest}\n${pathToFileURL(dest).href}\n`);
  return 0;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit(main());
}
