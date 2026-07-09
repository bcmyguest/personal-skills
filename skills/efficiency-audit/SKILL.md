---
name: efficiency-audit
description: Ruthlessly audit a codebase for its 1-3 most serious EFFICIENCY problems — the hot paths that dominate latency, cost, or resource use — then verify each with a concrete cost analysis (complexity, query count, allocation, growth path) and a fix. Use whenever the user invokes /efficiency-audit or asks to "find the performance bottlenecks", "why is this slow", "audit this for efficiency", "what's burning CPU/memory/money here", "find the N+1 queries", or wants a sharp, high-signal performance review that stops at the top few offenders rather than an exhaustive list. Spins up a small, capped multi-agent workflow of Sonnet agents to trace hot paths and prove cost. For security vulnerabilities use security-audit instead.
---

# Efficiency Audit

Find the **1–3 most serious efficiency problems** in a codebase, prove they matter with a
**concrete cost analysis**, and give the fix. These are the defects that dominate latency,
cloud bill, or memory — the O(n²) inner loop, the N+1 query behind a list endpoint, the
unbounded result set loaded into memory, the missing cache on an expensive repeated call.

The whole point is **signal over volume**. A ruthless efficiency auditor doesn't flag
every micro-optimization; it finds the one path that grows with your data or traffic and
will fall over in production, quantifies the cost, and stops. Stopping early is a feature:
the caller asked for the top few, and every extra finding past the target dilutes attention
and burns the budget you were told to cap. **Cost must scale** — a slow path that runs once
at startup on fixed-size input is not a finding; one whose cost grows with users, rows,
requests, or time is.

## How it works

This skill runs a **capped multi-agent workflow** (the `Workflow` tool) so recon, hunting,
and verification happen in parallel across cheap **Sonnet** agents instead of serially in
your own context. You stay the orchestrator: you launch the workflow, read its structured
result, and write the report. The user invoking this skill is your explicit authorization
to call `Workflow`.

Three phases, hard-capped at roughly ten agents total:

1. **Recon** (1 agent) — map the *hot paths*: request handlers and their query patterns,
   loops over collections that scale with data, batch jobs and cron tasks, serialization
   of large result sets, and any place data volume grows with users/rows/time. This focuses
   finders on code that actually runs often or over large inputs, not one-shot setup code.
2. **Hunt** (2 parallel finders, distinct lenses) — one for **algorithmic complexity**
   (O(n²)+ hot loops, N+1 queries, repeated work inside loops, quadratic joins, unbounded
   memory growth), one for **I/O & resource waste** (sync I/O on hot paths, missing
   pagination/streaming, connection/cursor leaks, absent caching on expensive repeated
   calls, over-fetching). Finders return only concrete, high-impact candidates whose cost
   scales — with exact file/line/snippet — or nothing.
3. **Verify** (1 agent per candidate, sequential, stops early) — default stance skeptical.
   Re-reads the real code to confirm the path is actually hot (reached per-request or over
   large input, not a rare one-off) and quantifies the cost: the complexity class, the
   query count as a function of N, the allocation, the growth path. Rejects anything whose
   cost doesn't scale or is already bounded by an existing limit/cache/index. Verification
   stops the instant the target count is reached.

## Running it

Run the bundled script. Point `root` at the code to audit.

```
Workflow({
  scriptPath: "<skill-dir>/scripts/efficiency_audit_workflow.js",
  args: { root: "<absolute-path-to-repo>", target: 3, sentry: "<optional Sentry URL/project>" }
})
```

- `root` — path to the codebase (repo root, or a subdirectory to scope the audit). Prefer
  an **absolute path** so agents resolve it regardless of their working directory.
- `target` — how many **confirmed** findings to stop at, 1–3 (default 3). If the user says
  "just find me the worst one", pass `1`.
- `sentry` — **optional**. A Sentry URL, project slug, or DSN-context string. When the user
  supplies one, pass it through: recon and verify agents use it to ground their work in real
  aggregate stats (slowest/most-frequent transactions, N+1 spans, highest-total-time
  endpoints) rather than guessing which paths are hot. Agents pull it via a Sentry MCP tool
  (`mcp__sentry__*`) if present, otherwise `WebFetch`. If the user mentions Sentry,
  performance dashboards, or "here's where prod is slow", capture that reference and pass it
  here. Omit it entirely for a purely static analysis.

The workflow returns `{ target, confirmed_count, findings: [...] }`. Each finding carries
`title`, `severity`, `file`, `line`, `summary`, `cost` (the quantified analysis), and `fix`.

If the `Workflow` tool isn't available, fall back to the same three phases inline with the
`Agent` tool (Sonnet subagents): one recon agent, two finders, then verify candidates one
at a time until you hit the target.

## Reporting the results

Write the report yourself from the workflow's structured output — this is the deliverable
the user reads, so lead with the verdict. Use this shape:

```
## Efficiency audit — <N> confirmed issue(s)

### 1. <title>  ·  <severity>
**Where:** `<file>:<line>`
**Issue:** <one-paragraph description of the hot path and why it matters>
**Cost:** <concrete analysis: complexity class, or query count as f(N), or allocation/growth — e.g. "1 + N queries per list request; ~500 rows in prod = 501 round-trips">
**Fix:** <the one-line fix — e.g. prefetch_related, add index, paginate, memoize>
```

Guidance:
- Order findings by impact, worst first.
- The proof is always a **concrete cost statement**, never "this is slow". State the
  complexity (O(n²)), the query count as a function of N (1+N), the allocation, or the
  growth path (grows unbounded with row count). Tie it to realistic production scale where
  you can.
- Confirm the path is **actually hot** — reached per request, per row, or over large input.
  A quadratic loop over a fixed 3-element config list is not a finding.
- Prefer a fix that is standard for the stack (e.g. `prefetch_related`/`select_related`
  for Django N+1, an index for a slow filter, pagination/streaming for large result sets,
  memoization/caching for repeated expensive calls).
- If the workflow confirmed **zero** issues, say so plainly and name the paths checked.
  A clean result reported honestly beats a padded one.
- End with a one-line note that this is a focused top-N audit, not exhaustive coverage.

## Safety

This is a read-only analysis. Do not run destructive load tests or modify code as part of
the audit; report the findings and fixes so the developer can apply them.
