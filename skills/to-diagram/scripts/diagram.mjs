#!/usr/bin/env node
// Check Mermaid diagrams with Mermaid's own parser, and write them to a standalone HTML file.
//
//   node diagram.mjs check FILE...            .mmd/.mermaid: whole file; .md: every ```mermaid fence; -: stdin
//   node diagram.mjs html FILE [-o OUT.html]  check, then write one offline HTML page; prints its path
//   options: --max-nodes N (default 20; 0 turns the cap off)
//
// Exit 0 when every diagram parses and fits the cap, 1 otherwise. Dependencies (mermaid, jsdom)
// install once into $TO_DIAGRAM_CACHE, else $XDG_CACHE_HOME/to-diagram, else ~/.cache/to-diagram.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const DEPS = { mermaid: '11.17.2', jsdom: '26.1.0' };
const CACHE = process.env.TO_DIAGRAM_CACHE
  || join(process.env.XDG_CACHE_HOME || join(homedir(), '.cache'), 'to-diagram');

function ensureDeps() {
  const req = createRequire(join(CACHE, 'package.json'));
  const has = (name) => { try { req.resolve(`${name}/package.json`); return true; } catch { return false; } };
  if (!Object.keys(DEPS).every(has)) {
    mkdirSync(CACHE, { recursive: true });
    if (!existsSync(join(CACHE, 'package.json'))) writeFileSync(join(CACHE, 'package.json'), '{"private":true}\n');
    const specs = Object.entries(DEPS).map(([n, v]) => `${n}@${v}`);
    process.stderr.write(`to-diagram: installing ${specs.join(' ')} into ${CACHE}\n`);
    execFileSync('npm', ['install', '--prefix', CACHE, '--no-audit', '--no-fund', '--silent', ...specs], { stdio: 'inherit' });
  }
  return req;
}

async function loadMermaid(req) {
  const { JSDOM } = req('jsdom');
  const dom = new JSDOM('<!doctype html><html><body></body></html>');
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  const entry = req.resolve('mermaid');
  const { default: mermaid } = await import(pathToFileURL(entry).href);
  return mermaid;
}

// Each block: { source, file, line } where line is the first line of the diagram text.
export function extractBlocks(text, file) {
  const ext = extname(file).toLowerCase();
  if (ext !== '.md' && ext !== '.markdown') return [{ source: text, file, line: 1 }];
  const blocks = [];
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const open = lines[i].match(/^(\s*)(`{3,}|~{3,})\s*mermaid\s*$/);
    if (!open) continue;
    const fence = open[2];
    const body = [];
    let j = i + 1;
    while (j < lines.length && !lines[j].trim().startsWith(fence)) body.push(lines[j++]);
    blocks.push({ source: body.join('\n'), file, line: i + 2 });
    i = j;
  }
  return blocks;
}

function countNodes(db) {
  const tries = [
    () => db.getVertices?.()?.size, () => db.getActors?.()?.size, () => db.getClasses?.()?.size,
    () => db.getEntities?.()?.size, () => db.getStates?.()?.size, () => db.getTasks?.()?.length,
  ];
  for (const t of tries) {
    try { const n = t(); if (typeof n === 'number') return n; } catch { /* next */ }
  }
  return null;
}

export async function checkBlocks(mermaid, blocks, maxNodes) {
  const results = [];
  const warn = console.warn; const error = console.error;
  console.warn = () => {}; console.error = () => {};
  try {
    for (const b of blocks) {
      const where = `${b.file}:${b.line}`;
      try {
        const { diagramType } = await mermaid.parse(b.source);
        const d = await mermaid.mermaidAPI.getDiagramFromText(b.source);
        const n = countNodes(d.db);
        if (maxNodes > 0 && n !== null && n > maxNodes) {
          results.push({ ok: false, where, msg: `${diagramType}: ${n} nodes, cap is ${maxNodes}; split by zoom level` });
        } else {
          results.push({ ok: true, where, msg: `${diagramType}${n === null ? '' : `, ${n} nodes`}` });
        }
      } catch (e) {
        const msg = String(e?.message || e).split('\n').filter(Boolean).slice(0, 3).join(' | ');
        results.push({ ok: false, where, msg });
      }
    }
  } finally {
    console.warn = warn; console.error = error;
  }
  return results;
}

const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export function toHtml(sources, title, mermaidJs) {
  const body = sources.map((s) => `<pre class="mermaid">\n${escapeHtml(s)}\n</pre>`).join('\n');
  const js = mermaidJs.replace(/<\/script/gi, '<\\/script');
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,sans-serif;margin:2rem;background:Canvas;color:CanvasText}pre.mermaid{margin:0 0 3rem}</style>
</head><body>
<h1 style="font-size:1rem;font-weight:600">${escapeHtml(title)}</h1>
${body}
<script>${js}</script>
<script>
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
mermaid.initialize({ startOnLoad: true, securityLevel: 'strict', theme: dark ? 'dark' : 'default' });
</script>
</body></html>
`;
}

function parseArgs(argv) {
  const args = { cmd: argv[0], files: [], out: null, maxNodes: 20 };
  for (let i = 1; i < argv.length; i++) {
    if (argv[i] === '-o' || argv[i] === '--out') args.out = argv[++i];
    else if (argv[i] === '--max-nodes') args.maxNodes = Number(argv[++i]);
    else args.files.push(argv[i]);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!['check', 'html'].includes(args.cmd) || args.files.length === 0 || (args.cmd === 'html' && args.files.length !== 1)) {
    process.stderr.write('usage: diagram.mjs check FILE... | diagram.mjs html FILE [-o OUT.html] [--max-nodes N]\n');
    return 2;
  }
  const blocks = [];
  for (const f of args.files) {
    const text = f === '-' ? readFileSync(0, 'utf8') : readFileSync(f, 'utf8');
    const found = extractBlocks(text, f === '-' ? 'stdin.mmd' : f);
    if (found.length === 0) process.stderr.write(`${f}: no mermaid blocks\n`);
    blocks.push(...found);
  }
  if (blocks.length === 0) return 1;
  const req = ensureDeps();
  const mermaid = await loadMermaid(req);
  const results = await checkBlocks(mermaid, blocks, args.maxNodes);
  for (const r of results) process.stdout.write(`${r.where}: ${r.ok ? 'OK' : 'ERROR'} ${r.msg}\n`);
  if (results.some((r) => !r.ok)) return 1;
  if (args.cmd === 'html') {
    const src = args.files[0];
    const out = resolve(args.out || (src === '-' ? 'diagram.html' : src.replace(/\.[^./]+$/, '') + '.html'));
    mkdirSync(dirname(out), { recursive: true });
    const js = readFileSync(join(dirname(req.resolve('mermaid/package.json')), 'dist', 'mermaid.min.js'), 'utf8');
    const title = src === '-' ? 'Diagram' : src.split('/').pop().replace(/\.[^.]+$/, '');
    writeFileSync(out, toHtml(blocks.map((b) => b.source), title, js));
    process.stdout.write(`wrote ${out}\n${pathToFileURL(out).href}\n`);
  }
  return 0;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then((code) => process.exit(code), (e) => { process.stderr.write(`${e.stack || e}\n`); process.exit(2); });
}
