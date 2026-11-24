// Tests for page.mjs: the spec, the layout, and a real Mermaid block.
// Run: node --test skills/sketch-it/scripts/test_page.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { problems, renderPage } from './page.mjs';

const SCRIPT = fileURLToPath(new URL('./page.mjs', import.meta.url));
const dir = mkdtempSync(join(tmpdir(), 'sketch-it-page-'));
const spec = (blocks, extra = {}) => ({ title: 'Token check', ...extra, blocks });

test('rejects a spec that cannot become a page', () => {
  assert.ok(problems({}).length > 0);
  assert.match(problems(spec([{ kind: 'widget' }])).join('\n'), /kind/);
  assert.match(problems(spec([{ kind: 'columns', items: [{ title: 'Only' }] }])).join('\n'), /items/);
  assert.deepEqual(problems(spec([{ kind: 'tree', text: 'root' }])), []);
});

test('puts the note inside the region and nests the call tree', () => {
  const html = renderPage(spec([
    { kind: 'screen', title: 'Settings', regions: [{ name: 'Toolbar', note: 'owns Run' }] },
    { kind: 'tree', text: 'handleRequest\n  authenticate\n    loadSession' },
    { kind: 'diff', text: ' dispatch\n+    authorize' },
  ], { point: 'Where the token is checked' }));
  assert.match(html, /<title>Token check<\/title>/);
  assert.match(html, /Where the token is checked/);
  assert.match(html, /<div class="region"><strong>Toolbar<\/strong><p>owns Run<\/p><\/div>/);
  assert.match(html, /<li>handleRequest<ul><li>authenticate<ul><li>loadSession/);
  assert.match(html, /class="add"/);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /max-width: 360px/);
  assert.match(html, /minmax\(220px/);
});

test('escapes text that would break the page', () => {
  const html = renderPage(spec([{ kind: 'pre', text: '<script>alert(1)</script>' }]));
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>alert/);
});

test('a mermaid block is checked and inlined; a broken one is refused', () => {
  const good = join(dir, 'good.json');
  writeFileSync(good, JSON.stringify(spec([
    { kind: 'mermaid', title: 'Order', source: 'sequenceDiagram\n  participant User\n  participant Page\n  User->>Page: open' },
  ])));
  const out = join(dir, 'good.html');
  const r = execFileSync('node', [SCRIPT, good, '-o', out], { encoding: 'utf8' });
  assert.match(r, /wrote /);
  const html = readFileSync(out, 'utf8');
  assert.match(html, /class="mermaid"/);
  assert.match(html, /mermaid\.initialize/);
  assert.ok(html.length > 100_000);

  const bad = join(dir, 'bad.json');
  writeFileSync(bad, JSON.stringify(spec([{ kind: 'mermaid', source: 'sequenceDiagram\n  this is not mermaid' }])));
  assert.throws(() => execFileSync('node', [SCRIPT, bad, '-o', join(dir, 'bad.html')], { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }));
});
