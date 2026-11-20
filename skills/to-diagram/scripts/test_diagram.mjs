// Tests for diagram.mjs: fence extraction, parse errors, the node cap, and the HTML page.
// Run: node --test skills/to-diagram/scripts/test_diagram.mjs (installs mermaid on first run).
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { extractBlocks, toHtml } from './diagram.mjs';

const SCRIPT = fileURLToPath(new URL('./diagram.mjs', import.meta.url));
const dir = mkdtempSync(join(tmpdir(), 'to-diagram-'));
const run = (...args) => {
  try {
    return { code: 0, out: execFileSync('node', [SCRIPT, ...args], { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }) };
  } catch (e) {
    return { code: e.status, out: e.stdout };
  }
};
const file = (name, text) => { const p = join(dir, name); writeFileSync(p, text); return p; };

test('extracts each mermaid fence from markdown with its line', () => {
  const md = '# t\n\n```mermaid\nflowchart TD\n A-->B\n```\n\n```js\nx\n```\n\n~~~mermaid\npie\n "a": 1\n~~~\n';
  const blocks = extractBlocks(md, 'doc.md');
  assert.deepEqual(blocks.map((b) => [b.line, b.source]), [[4, 'flowchart TD\n A-->B'], [13, 'pie\n "a": 1']]);
  assert.equal(extractBlocks('flowchart TD\n A', 'x.mmd').length, 1);
});

test('valid diagrams of each type pass', () => {
  const p = file('ok.md', [
    'flowchart LR\n A["parse(x)"] -->|validates| B',
    'sequenceDiagram\n A->>B: hi',
    'stateDiagram-v2\n [*] --> S\n S --> [*]',
    'erDiagram\n A ||--o{ B : has',
    'classDiagram\n A <|-- B',
    'gantt\n title t\n section s\n a :a1, 2026-01-01, 3d',
    'xychart-beta\n x-axis [a, b]\n bar [1, 2]',
  ].map((s) => '```mermaid\n' + s + '\n```').join('\n\n'));
  const r = run('check', p);
  assert.equal(r.code, 0, r.out);
  assert.equal(r.out.trim().split('\n').length, 7);
});

test('a syntax error fails with its location', () => {
  const r = run('check', file('bad.mmd', 'flowchart TD\n A --> B -->\n'));
  assert.equal(r.code, 1);
  assert.match(r.out, /bad\.mmd:1: ERROR Parse error/);
});

test('more nodes than the cap fails; --max-nodes 0 lifts it', () => {
  const big = 'flowchart TD\n' + Array.from({ length: 25 }, (_, i) => ` n${i} --> n${i + 1}`).join('\n');
  const p = file('big.mmd', big);
  assert.match(run('check', p).out, /26 nodes, cap is 20/);
  assert.equal(run('check', p, '--max-nodes', '0').code, 0);
});

test('html writes one offline page and refuses a broken diagram', () => {
  const out = join(dir, 'page.html');
  const r = run('html', file('flow.mmd', 'flowchart LR\n A --> B'), '-o', out);
  assert.equal(r.code, 0, r.out);
  const html = readFileSync(out, 'utf8');
  assert.match(html, /<pre class="mermaid">\nflowchart LR\n A --&gt; B\n<\/pre>/);
  assert.doesNotMatch(html, /<script src=/);
  assert.equal(run('html', file('broken.mmd', 'flowchart TD\n A -->'), '-o', join(dir, 'no.html')).code, 1);
});

test('toHtml escapes a closing script tag inside the library', () => {
  assert.doesNotMatch(toHtml(['pie'], 't', 'a="</script>"').split('<script>')[1], /<\/script>"/);
});
