export const meta = {
  name: 'security-audit',
  description: 'Ruthlessly audit a codebase for 1-3 high-severity security vulnerabilities, verify each adversarially, stop early once confirmed with a CVE ref or minimal PoC',
  phases: [
    { title: 'Recon', detail: 'map the attack surface and pick the highest-risk entry points', model: 'sonnet' },
    { title: 'Hunt', detail: 'parallel finders dig for concrete injection / authz / secret / CVE-dep bugs', model: 'sonnet' },
    { title: 'Verify', detail: 'adversarially confirm each candidate; produce CVE id or minimal PoC', model: 'sonnet' },
  ],
}

// args: { root?: string, target?: number }
const ROOT = (args && args.root) || '.'
const TARGET = (args && args.target) || 3   // stop once this many findings are CONFIRMED (1-3)

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
        required: ['title', 'severity', 'file', 'line', 'why', 'evidence', 'attacker'],
        properties: {
          title: { type: 'string' },
          severity: { enum: ['critical', 'high', 'medium', 'low'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          why: { type: 'string', description: 'one sentence: the vulnerability and its impact' },
          attacker: { type: 'string', description: 'who reaches it: unauthenticated / other tenant / low-priv user / etc' },
          evidence: { type: 'string', description: 'the offending code snippet + surrounding context' },
          cve_hint: { type: 'string', description: 'vulnerable package@version or CWE class if applicable, else empty' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confirmed', 'severity', 'summary', 'reasoning', 'fix'],
  properties: {
    confirmed: { type: 'boolean', description: 'true ONLY if the vulnerability is real and reachable by the claimed attacker' },
    severity: { enum: ['critical', 'high', 'medium', 'low'], description: 'your corrected severity after verification' },
    summary: { type: 'string', description: 'one-paragraph summary for the final report: the bug, who reaches it, what they gain' },
    reasoning: { type: 'string', description: 'why it is or is not real; the data flow you traced; guards you ruled out' },
    cve: { type: 'string', description: 'CVE id + one-line description + affected package@version if a known-vuln dependency, else empty' },
    poc: { type: 'string', description: 'for a novel bug with no CVE: a minimal, non-destructive proof-of-concept (exact request/input/code) that triggers it, else empty' },
    fix: { type: 'string', description: 'the one-line fix' },
  },
}

// ---- Phase 1: Recon -------------------------------------------------------
phase('Recon')
const recon = await agent(
  `You are doing recon for a ruthless SECURITY audit of the codebase at "${ROOT}".
Do NOT fix anything. Map the ATTACK SURFACE so finders dig where an attacker actually reaches.
Identify: the languages/frameworks; entry points that handle untrusted input (HTTP/GraphQL/REST handlers, deserializers, query builders, file/path handling, subprocess/eval, template rendering, redirects); the authentication + authorization model (how identity and permissions are enforced, and where they might be missing); secret handling (config, env, hardcoded); and dependency manifests (record file paths so we can check for known-CVE versions).
Return a concise prioritized list of the 4-6 highest-risk files or areas, each with a one-line reason an attacker would care.`,
  { label: 'recon', phase: 'Recon', model: 'sonnet' }
)

// ---- Phase 2: Hunt (parallel finders, distinct security lenses) -----------
phase('Hunt')
const FINDERS = [
  { key: 'injection', lens: 'injection & unsafe handling of untrusted input: SQL injection, command/OS injection, path traversal, SSRF, insecure deserialization, template/eval injection, unsafe redirects. Trace untrusted input from an entry point to a dangerous sink.' },
  { key: 'authz-secrets', lens: 'authentication, authorization & secrets: broken or MISSING access control, tenant/org isolation bypass (IDOR), privilege escalation, hardcoded secrets/keys/tokens, weak or misused crypto, and known-vulnerable dependencies (record package@version for CVE lookup).' },
]

const found = (await parallel(FINDERS.map(f => () =>
  agent(
    `You are a finder in a ruthless SECURITY audit of "${ROOT}".
Your lens: ${f.lens}

Recon prioritized these areas:
${recon}

Read the ACTUAL code. Find CONCRETE, high-severity, REACHABLE vulnerabilities only — a real line of code an attacker reaches with real impact, not "consider adding validation" nits. For each, capture the exact file, line, the offending snippet with surrounding context, a one-sentence impact, and WHO reaches it (unauthenticated / another tenant / low-priv user). If it involves a dependency, record package@version so we can find the CVE. Prefer 1-3 of your strongest candidates over a long shallow list. If you find nothing genuinely high-severity, return an empty list — do not manufacture findings.`,
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
  return { target: TARGET, confirmed_count: 0, findings: [], note: 'No high-severity security vulnerabilities found.' }
}

// ---- Phase 3: Verify (adversarial, sequential so we can stop early) --------
phase('Verify')
const confirmed = []
for (const c of candidates) {
  if (confirmed.length >= TARGET) break
  const verdict = await agent(
    `Adversarially verify this candidate SECURITY finding from "${ROOT}". Your default stance is skeptical: only confirm if the vulnerability is REAL and actually reachable by the claimed attacker.

Candidate: ${c.title}
Claimed severity: ${c.severity}
Location: ${c.file}:${c.line}
Attacker who reaches it (claimed): ${c.attacker}
Why it matters (claimed): ${c.why}
Evidence:
${c.evidence}
Dependency/CWE hint: ${c.cve_hint || '(none)'}

Read the real code around ${c.file}:${c.line} and trace the untrusted-input -> sink data flow to confirm reachability. Check whether an existing guard already neutralizes it: ORM parameterization, framework auto-escaping, a permission/auth decorator or middleware, input validation, an allowlist.
- If it is a known-vulnerable dependency, look up and cite the CVE id + one-line description + affected package@version (use web/CVE lookup tools).
- If it is a novel bug with no CVE, produce a MINIMAL, NON-DESTRUCTIVE proof-of-concept: the exact request/input/code that triggers it.
Return confirmed=false if a guard neutralizes it or you cannot substantiate reachability.`,
    { label: `verify:${c.file.split('/').pop()}:${c.line}`, phase: 'Verify', model: 'sonnet', schema: VERDICT_SCHEMA }
  )
  if (verdict && verdict.confirmed) {
    confirmed.push({ title: c.title, file: c.file, line: c.line, attacker: c.attacker, ...verdict })
    log(`Confirmed ${confirmed.length}/${TARGET}: ${c.title}`)
  }
}

return {
  target: TARGET,
  confirmed_count: confirmed.length,
  findings: confirmed,
}
