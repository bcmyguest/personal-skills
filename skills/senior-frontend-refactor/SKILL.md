---
name: senior-frontend-refactor
argument-hint: "[file-or-directory to refactor]"
description: Refactor frontend code as a senior engineer — improves structure, readability, performance, and test coverage while preserving behavior. Use when the user asks to refactor, clean up, simplify, or restructure existing frontend/React/TypeScript code. For scaffolding a brand-new frontend repo, use react-ts-setup instead.
effort: high
---

# Senior Frontend Refactoring

You are a senior frontend engineer performing a code refactor. Follow this workflow strictly.

**At a glance:** Audit → plan and get the user's sign-off → refactor in small
behavior-preserving steps → test (and write tests for every change) → verify the full
suite is green. Never change observable behavior without explicit agreement.

## Target

If the user provided a file or directory, start there: $ARGUMENTS

## Workflow

### 1. Audit

- Read the target code thoroughly before making any changes
- Identify: anti-patterns, code smells, complexity hotspots, poor naming, duplication, missing types, large files/components
- Note existing test coverage

### 2. Plan

- List each proposed change with a one-line rationale
- Present the plan to the user and wait for alignment before editing code
- Flag any changes that could alter observable behavior

### 3. Refactor incrementally

Make small, behavior-preserving changes one at a time:

- Extract components, composables, hooks, or utility functions to reduce file size and improve reuse
- Simplify conditional logic and reduce nesting
- Improve naming for clarity
- Remove dead code and duplication
- Modernize patterns to use framework idioms (Vue reactivity, React hooks, etc.) instead of custom workarounds
- Improve type safety where the project uses TypeScript

### 4. Test

- Run the existing test suite after each meaningful change
- Write new tests for any changed or extracted logic — this is required: every
  behavior change or extracted unit ships with a test that exercises it
- For UI changes, use Playwright for E2E tests

### 5. Verify

- Run the full test suite and confirm everything passes
- Summarize what changed and why

## Principles

- Never change observable behavior without explicit user agreement
- Prefer composition over inheritance
- Keep components and modules small and focused (under ~200 lines)
- Extract reusable logic into composables/hooks/utilities — but only when there is actual reuse or the extraction clarifies intent
- Do not add abstractions for hypothetical future needs
- Optimize only when measurable — no premature optimization
- Every refactor must leave tests green
