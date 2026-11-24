#!/usr/bin/env node
// Turn a short JSON spec into one offline HTML page.
//
//   node page.mjs SPEC.json [-o OUT.html]
//
// The spec is { "title", "point"?, "blocks": [ ... ] }. Block kinds:
//   tree     { "text" }            indented names, two spaces per level
//   pre      { "text" }            pseudocode or a file tree, monospaced
//   diff     { "text" }            lines starting with + or - are the change
//   columns  { "items": [ { "title", "text" } ] }
//   screen   { "regions": [ { "name", "note"? } ] }
//   mermaid  { "source" }          checked with diagram.mjs before the page is written
// Each block may also have "title". No network, no navigation. A mermaid block
// inlines Mermaid's own browser build so the file still opens offline.

import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readMermaidJs } from './diagram.mjs';

const KINDS = new Set(['tree', 'pre', 'diff', 'columns', 'screen', 'mermaid']);

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
  if (b.kind === 'tree' || b.kind === 'pre' || b.kind === 'diff') {
    if (typeof b.text !== 'string' || !b.text.trim()) return [`${at}.text is required`];
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

function renderNodes(nodes) {
  if (nodes.length === 0) return '';
  const items = nodes.map((n) => `<li>${escapeHtml(n.label)}${renderNodes(n.children)}</li>`).join('');
  return `<ul>${items}</ul>`;
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
  } else body = `<pre class="mermaid">${escapeHtml(b.source)}</pre>`;
  return `<section class="block">${heading}${body}</section>`;
}

const CSS = `
body { font: 16px/1.45 system-ui, sans-serif; margin: 24px auto; max-width: 880px; padding: 0 16px; background: Canvas; color: CanvasText; }
h1 { font-size: 1.25rem; margin: 0 0 8px; }
.point { margin: 0 0 24px; }
h2 { font-size: 1rem; margin: 0 0 8px; }
h3 { font-size: 0.95rem; margin: 0 0 4px; }
.block { margin: 0 0 28px; }
.tree ul { list-style: none; margin: 0; padding-left: 16px; border-left: 1px solid color-mix(in srgb, CanvasText 35%, transparent); }
.tree > ul { padding-left: 0; border-left: 0; }
.tree li { margin: 4px 0; }
pre, .diff { font: 14px/1.4 ui-monospace, monospace; margin: 0; white-space: pre-wrap; }
.diff .add { background: color-mix(in srgb, green 22%, transparent); }
.diff .del { background: color-mix(in srgb, red 22%, transparent); }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.cols section, .region { border: 1px solid color-mix(in srgb, CanvasText 35%, transparent); padding: 12px; }
.cols p, .region p { margin: 4px 0 0; white-space: pre-wrap; }
.screen { display: grid; gap: 8px; max-width: 360px; }
.mermaid { margin: 0; }
`;

export function renderPage(spec, mermaidJs = '') {
  const blocks = spec.blocks.map(renderBlock).join('\n');
  const point = spec.point ? `<p class="point">${escapeHtml(spec.point)}</p>` : '';
  const js = mermaidJs ? `<script>${mermaidJs.replace(/<\/script/gi, '<\\/script')}</script>
<script>mermaid.initialize({ startOnLoad: true, securityLevel: 'strict', theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default' });</script>` : '';
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
