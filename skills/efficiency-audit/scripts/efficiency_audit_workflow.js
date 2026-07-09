export const meta = {
  name: 'efficiency-audit',
  description: 'Ruthlessly audit a codebase for 1-3 high-impact efficiency problems on hot paths, verify each with a concrete cost analysis, stop early once confirmed',
  phases: [
    { title: 'Recon', detail: 'map the hot paths where cost scales with data/traffic', model: 'sonnet' },
    { title: 'Hunt', detail: 'parallel finders dig for concrete complexity / N+1 / I/O / memory waste', model: 'sonnet' },
    { title: 'Verify', detail: 'adversarially confirm the path is hot and quantify the cost', model: 'sonnet' },
  ],
}

// args: { root?: string, target?: number, sentry?: string }
const ROOT = (args && args.root) || '.'
const TARGET = (args && args.target) || 3   // stop once this many findings are CONFIRMED (1-3)
const SENTRY = (args && args.sentry) || ''  // optional: a Sentry URL/project/DSN-context to pull aggregate perf stats from

const sentryBlock = SENTRY
  ? `\n\nPRODUCTION SIGNAL — a Sentry reference was provided: ${SENTRY}
Use it to ground your work in real aggregate data instead of guessing. If a Sentry MCP tool is available (mcp__sentry__*), query it for the slowest/most-frequent transactions, N+1 spans, and endpoints with the highest total time; otherwise fetch the URL with WebFetch. Prioritize the paths production actually shows as hot. If you cannot access it, proceed with static analysis and say so.`
  : ''

const CANDIDATE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['candidates'],
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'severity', 'file', 'line', 'why', 'evidence', 'hotness'],
        properties: {
          title: { type: 'string' },
          severity: { enum: ['critical', 'high', 'medium', 'low'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          why: { type: 'string', description: 'one sentence: the inefficiency and its cost' },
          hotness: { type: 'string', description: 'why this path is hot: per-request / per-row / over-large-input / high-frequency cron' },
          evidence: { type: 'string', description: 'the offending code snippet + surrounding context' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confirmed', 'severity', 'summary', 'cost', 'reasoning', 'fix'],
  properties: {
    confirmed: { type: 'boolean', description: 'true ONLY if the path is genuinely hot AND its cost scales with data/traffic (not a one-off or already-bounded path)' },
    severity: { enum: ['critical', 'high', 'medium', 'low'], description: 'your corrected severity after verification' },
    summary: { type: 'string', description: 'one-paragraph summary for the final report: the hot path and why it matters' },
    cost: { type: 'string', description: 'the CONCRETE cost analysis: complexity class, query count as f(N), allocation, or growth path — tied to realistic scale where possible' },
    reasoning: { type: 'string', description: 'why it is or is not real; how you confirmed the path is hot; any existing limit/cache/index you ruled out' },
    fix: { type: 'string', description: 'the one-line fix, idiomatic for the stack' },
  },
}

// ---- Phase 1: Recon -------------------------------------------------------
phase('Recon')
const recon = await agent(
  `You are doing recon for a ruthless EFFICIENCY audit of the codebase at "${ROOT}".
Do NOT fix anything. Map the HOT PATHS — the code that runs often or over large input, where cost scales with users/rows/requests/time.
Identify: the languages/frameworks; request handlers and the query patterns behind them (look for list/collection endpoints prone to N+1); loops over collections whose size scales with data; batch jobs / cron tasks; serialization of large result sets; and anywhere data volume grows unbounded.
Explicitly note what is NOT hot (one-shot startup/config code) so finders don't waste effort there.${sentryBlock}
Return a concise prioritized list of the 4-6 hottest files or areas, each with a one-line reason the cost scales.`,
  { label: 'recon', phase: 'Recon', model: 'sonnet' }
)

// ---- Phase 2: Hunt (parallel finders, distinct efficiency lenses) ---------
phase('Hunt')
const FINDERS = [
  { key: 'complexity', lens: 'algorithmic complexity: O(n^2)+ hot loops, N+1 database queries, repeated work inside loops, quadratic joins/lookups, unbounded in-memory growth. Report the cost as a function of N.' },
  { key: 'io-resource', lens: 'I/O & resource waste: synchronous/blocking I/O on hot request paths, missing pagination or streaming on large result sets, connection/cursor/file-handle leaks, absent caching/memoization on expensive repeated calls, over-fetching columns or rows.' },
]

const found = (await parallel(FINDERS.map(f => () =>
  agent(
    `You are a finder in a ruthless EFFICIENCY audit of "${ROOT}".
Your lens: ${f.lens}

Recon prioritized these hot areas (and noted what is NOT hot):
${recon}

Read the ACTUAL code. Find CONCRETE, high-impact inefficiencies whose cost SCALES with data or traffic — not micro-optimizations, and not slow code that runs once on fixed-size input. For each, capture the exact file, line, the offending snippet with surrounding context, a one-sentence cost, and WHY the path is hot (per-request / per-row / over-large-input / high-frequency). Prefer 1-3 of your strongest candidates over a long shallow list. If you find nothing whose cost genuinely scales, return an empty list — do not manufacture findings.`,
    { label: `hunt:${f.key}`, phase: 'Hunt', model: 'sonnet', schema: CANDIDATE_SCHEMA }
  )
))).filter(Boolean).flatMap(r => r.candidates)

// Dedup by file+line, rank by severity, keep the strongest handful for verification.
const RANK = { critical: 0, high: 1, medium: 2, low: 3 }
const seen = new Set()
const candidates = found
  .filter(c => { const k = `${c.file}:${c.line}`; if (seen.has(k)) return false; seen.add(k); return true })
  .sort((a, b) => (RANK[a.severity] - RANK[b.severity]))
  .slice(0, Math.max(TARGET * 2, 4))   // verify a few extra since some won't survive

log(`Recon done. ${found.length} raw candidates -> ${candidates.length} to verify (target ${TARGET} confirmed).`)

if (candidates.length === 0) {
  return { target: TARGET, confirmed_count: 0, findings: [], note: 'No high-impact efficiency issues found.' }
}

// ---- Phase 3: Verify (adversarial, sequential so we can stop early) --------
phase('Verify')
const confirmed = []
for (const c of candidates) {
  if (confirmed.length >= TARGET) break
  const verdict = await agent(
    `Adversarially verify this candidate EFFICIENCY finding from "${ROOT}". Your default stance is skeptical: only confirm if the path is genuinely hot AND its cost scales with data or traffic.

Candidate: ${c.title}
Claimed severity: ${c.severity}
Location: ${c.file}:${c.line}
Why this path is hot (claimed): ${c.hotness}
Why it matters (claimed): ${c.why}
Evidence:
${c.evidence}

Read the real code around ${c.file}:${c.line}. Confirm the path is actually reached often or over large input (trace callers if needed). Then QUANTIFY the cost: state the complexity class, the query count as a function of N, the allocation, or the growth path — tied to realistic production scale where you can.${sentryBlock}
Reject it (confirmed=false) if: the cost does not scale (fixed-size or one-off input), or an existing guard already bounds it (a LIMIT/pagination, a cache, a DB index, prefetch/select_related already present, a small hard cap).`,
    { label: `verify:${c.file.split('/').pop()}:${c.line}`, phase: 'Verify', model: 'sonnet', schema: VERDICT_SCHEMA }
  )
  if (verdict && verdict.confirmed) {
    confirmed.push({ title: c.title, file: c.file, line: c.line, hotness: c.hotness, ...verdict })
    log(`Confirmed ${confirmed.length}/${TARGET}: ${c.title}`)
  }
}

return {
  target: TARGET,
  confirmed_count: confirmed.length,
  findings: confirmed,
}
