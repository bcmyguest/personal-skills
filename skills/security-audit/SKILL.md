---
name: security-audit
description: Ruthlessly audit a codebase for its 1-3 most serious SECURITY vulnerabilities, verify each one adversarially, and report it with a CVE reference (for known-vulnerable dependencies) or a minimal proof-of-concept (for novel bugs). Use whenever the user invokes /security-audit or asks to "security audit this", "find the worst vulnerabilities here", "is this exploitable", "find the scariest security holes", "what's the biggest security risk in this repo", or wants a sharp, high-signal security review that stops at the top few issues rather than an exhaustive checklist. Spins up a small, capped multi-agent workflow of Sonnet agents to hunt attack surface and prove exploitability. For performance/cost problems use efficiency-audit instead.
---

# Security Audit

Find the **1–3 most serious security vulnerabilities** in a codebase, prove they're real,
and report each with enough evidence to act on — a **CVE** for known-vulnerable
dependencies, or a **minimal proof-of-concept** for novel bugs.

The whole point is **signal over volume**. A ruthless security auditor doesn't hand back
forty style nits; it finds the thing an attacker actually reaches — the unauthenticated
RCE, the tenant-isolation bypass, the injection behind a customer-facing endpoint —
confirms it isn't a false alarm, and stops. Stopping early is a feature: the caller asked
for the top few, and every extra finding past the target dilutes attention and burns the
budget you were told to cap. A confirmed critical outranks three plausible-but-unproven
mediums.

## How it works

This skill runs a **capped multi-agent workflow** (the `Workflow` tool) so recon, hunting,
and adversarial verification happen in parallel across cheap **Sonnet** agents instead of
serially in your own context. You stay the orchestrator: you launch the workflow, read its
structured result, and write the report. The user invoking this skill is your explicit
authorization to call `Workflow` for a **defensive** audit.

Three phases, hard-capped at roughly ten agents total:

1. **Recon** (1 agent) — map the *attack surface*: entry points that handle untrusted
   input (HTTP/GraphQL/DRF handlers, deserializers, query builders, file/path handling,
   subprocess/`eval`, template rendering), the authentication/authorization model, secret
   handling, and dependency manifests for CVE lookup. This focuses finders where an
   attacker actually reaches, not on dead internal code.
2. **Hunt** (2 parallel finders, distinct lenses) — one for **injection & unsafe input**
   (SQLi, command injection, path traversal, SSRF, deserialization, template/eval
   injection), one for **authz & secrets** (broken/missing access control, tenant
   isolation, hardcoded secrets, weak crypto, known-vulnerable dependencies with
   `package@version` recorded for CVE lookup). Finders return only concrete, reachable,
   high-severity candidates with exact file/line/snippet — or nothing.
3. **Verify** (1 adversarial agent per candidate, sequential, stops early) — default
   stance skeptical. Re-reads the real code to confirm the untrusted-input→sink data flow
   is reachable and that no existing guard (ORM parameterization, framework escaping,
   permission decorator, auth middleware) already neutralizes it. Produces the **CVE id**
   for vulnerable deps or a **minimal PoC** for novel bugs. Verification stops the instant
   the target count is reached, so unproven extras cost nothing.

## Running it

Run the bundled script. Point `root` at the code to audit.

```
Workflow({
  scriptPath: "<skill-dir>/scripts/security_audit_workflow.js",
  args: { root: "<absolute-path-to-repo>", target: 3 }
})
```

- `root` — path to the codebase (repo root, or a subdirectory to scope the audit). Prefer
  an **absolute path** so agents resolve it regardless of their working directory.
- `target` — how many **confirmed** findings to stop at, 1–3 (default 3). If the user says
  "just find me the worst one", pass `1`.

The workflow returns `{ target, confirmed_count, findings: [...] }`. Each finding carries
`title`, `severity`, `file`, `line`, `summary`, `cve`, `poc`, and `fix`.

If the `Workflow` tool isn't available, fall back to the same three phases inline with the
`Agent` tool (Sonnet subagents): one recon agent, two finders, then verify candidates one
at a time until you hit the target.

## Reporting the results

Write the report yourself from the workflow's structured output — this is the deliverable
the user reads, so lead with the verdict. Use this shape:

```
## Security audit — <N> confirmed vulnerability(ies)

### 1. <title>  ·  <severity>
**Where:** `<file>:<line>`
**Vulnerability:** <one-paragraph description of the bug and its impact — who can reach it, what they gain>
**Proof:** <CVE-XXXX-NNNNN — one-line description — affected package@version> OR <minimal PoC: the exact request/input/code that triggers it>
**Fix:** <the one-line fix>
```

Guidance:
- Order findings by severity, worst first.
- For a **known-vulnerable dependency**, the proof is the **CVE** (id + one-line
  description + affected `package@version`). Look it up with web/CVE tools — never guess an id.
- For a **novel bug**, the proof is a **minimal, non-destructive PoC**: the smallest
  concrete request/input/snippet that triggers it. Show the exploit path, don't run a
  destructive one.
- State who the attacker is and what they reach (unauthenticated? another tenant?
  low-privilege user?) — reachability is what separates a real finding from a lint nit.
- If the workflow confirmed **zero** vulnerabilities, say so plainly and name the surfaces
  checked. A clean result reported honestly beats a padded one.
- End with a one-line note that this is a focused top-N audit, not exhaustive coverage.

## Safety

This is a **defensive** audit. Report vulnerabilities and minimal PoCs so they can be
fixed; never weaponize them, exfiltrate data, or run destructive proofs. If the audit
surfaces a live secret in code, flag it and recommend rotation — do **not** echo the
secret value.
