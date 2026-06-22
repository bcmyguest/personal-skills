---
name: react-ts-setup
description: Scaffold a new React + TypeScript frontend repo the right way with Vite — asks the key decisions up front (package manager, Node manager + version, UI framework + theme colours, CI provider, dev Docker), then wires Vite + React Compiler, ESLint (flat) + Prettier, Vitest + Testing Library, husky + lint-staged + pre-commit hooks, .vscode settings, env templates, a dev Dockerfile, CI pipelines, and a README — keeping the Node version in lockstep across every file and leaving lint/typecheck/test/build green. Use this whenever starting a new frontend, React, or TypeScript web project, scaffolding a single-page app, or asked to "set up", "bootstrap", or "initialize" a React/Vite repo. For refactoring existing frontend code, use senior-frontend-refactor instead.
---

# Setting up a React + TypeScript project

Scaffold every new React SPA with **Vite** (it owns dev server, build, and pairs
natively with Vitest). Work top to bottom; confirm the decisions in step 0 before
scaffolding, because they change dependencies, config, and which later steps apply.

The whole point of this skill is a repo where **one `git clone` + install gets a
contributor to green** — lint, typecheck, test, and build all pass, the editor
formats on save, and commits are guarded. The recurring failure mode is a Node
version that drifts between `.nvmrc`, CI, and Docker; the steps below keep it in
**one** place on purpose.

> Versions move fast in this ecosystem. Don't hard-code dependency versions in
> `package.json` by hand and don't copy version numbers from this skill — let
> `create-vite` and `pnpm add` install the newest releases, and trust the lockfile.

**Steps:** (0) ask the decisions → (1) scaffold with Vite → (2) pin Node everywhere →
(3) ESLint + Prettier → (4) React Compiler + strict TS → (5) Vitest + Testing Library →
(6) UI framework + theme → (7) husky + lint-staged + pre-commit hooks → (8) `.vscode`
settings → (9) env files → (10) `.gitignore` check → (11) dev Docker (if chosen) →
(12) CI (if chosen) → (13) README → (14) verify lint/typecheck/test/build all green.
Steps 6, 11, and 12 depend on the step-0 answers.

## 0. Ask the decisions up front

Don't assume — ask, then use the answers to decide which steps apply:

| Decision           | Options                                              | Default                          |
| ------------------ | --------------------------------------------------- | -------------------------------- |
| Package manager    | pnpm · npm · bun                                     | **pnpm**                         |
| Node manager       | nvm · mise · asdf · Volta                            | ask; all write a pinned version  |
| Node version       | current Active LTS                                   | **24** (2026); 22 if conservative |
| UI framework       | MUI · Tailwind · Chakra · Mantine · none             | ask                              |
| Theme colours      | primary / secondary / error / background            | **ask for the basic set**        |
| Test framework     | Vitest                                               | **Vitest** (fixed)               |
| CI provider        | none · GitHub · GitLab · Bitbucket                   | ask                              |
| Dev Docker image   | yes · no                                             | ask                              |

The canonical path below is **pnpm + Vite + MUI**. Package-manager and UI-framework
substitutions are called out inline and in the reference files.

## 1. Scaffold with Vite

```bash
pnpm create vite@latest <app-name> --template react-ts
cd <app-name>
pnpm install
```

Package-manager variants: `npm create vite@latest <app> -- --template react-ts` ·
`bun create vite <app> --template react-ts`. The rest of this skill uses `pnpm`;
swap the prefix (`npm run` / `bun run`) for the others.

Then record the toolchain in `package.json` so CI and contributors match:

```jsonc
{
  "packageManager": "pnpm@<your installed version>", // exact; Corepack reads this. `pnpm --version`
  "engines": { "node": ">=22" } // a FLOOR, not the pin (the pin is step 2)
}
```

## 2. Pin Node and keep every file in lockstep

The Node version is the thing most likely to drift. It appears in up to four
places — make them agree, and prefer a **single source of truth** so they can't
diverge:

- **Node-manager file** (the real pin) — one of, matching the user's choice:
  - nvm: `.nvmrc` → `24`
  - mise / asdf: `.tool-versions` → `nodejs 24` (mise also reads `mise.toml`)
  - Volta: a `volta` block in `package.json`
- **`engines.node`** in `package.json` — a floor (`>=22`), not the exact pin.
- **CI** — read the pin instead of repeating it: `node-version-file: .nvmrc`
  (step 12). This is the trick that keeps CI from drifting.
- **Dockerfile** — `FROM node:24-alpine`, the same major (step 11). The one place
  that can't auto-read the pin, so bump it deliberately when you bump `.nvmrc`.

Commit the node-manager file. Default to the current **Active LTS** (Node 24 in
2026; 22 is the conservative choice and `engines` covers both).

## 3. ESLint (flat) + Prettier — set up so they never fight

`create-vite` ships a flat-config `eslint.config.js`. Add Prettier as the formatter,
`eslint-config-prettier` to switch off every ESLint rule that overlaps with
formatting (that's what stops the two tools from undoing each other), and
`eslint-plugin-perfectionist` to sort imports:

```bash
pnpm add -D prettier eslint-config-prettier eslint-plugin-perfectionist
```

Edit `eslint.config.js`: put `eslintConfigPrettier` **last** in `extends`, and add a
second config object for import ordering:

```js
import perfectionist from 'eslint-plugin-perfectionist'
import eslintConfigPrettier from 'eslint-config-prettier'
// ...
export default defineConfig([
  globalIgnores(['dist', 'coverage']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended, // includes the React Compiler rule (step 4)
      reactRefresh.configs.vite,
      eslintConfigPrettier, // MUST be last: disables formatting rules
    ],
    languageOptions: { globals: globals.browser },
  },
  {
    // Import ordering. Scope to the import rules — not perfectionist's full
    // recommended set, which also sorts objects/props/enums. `eslint --fix` sorts
    // imports; Prettier doesn't reorder them, so the two never conflict. Side-effect
    // imports (CSS, polyfills) keep their relative order by default.
    files: ['**/*.{ts,tsx}'],
    plugins: { perfectionist },
    rules: {
      'perfectionist/sort-imports': ['error', { type: 'natural' }],
      'perfectionist/sort-named-imports': ['error', { type: 'natural' }],
    },
  },
])
```

`.prettierrc.json` — match `create-vite`'s style (no semicolons, single quotes) so
the first format pass doesn't churn every file:

```json
{ "semi": false, "singleQuote": true, "trailingComma": "all", "printWidth": 80 }
```

`.prettierignore`:

```
dist
coverage
pnpm-lock.yaml
```

Add scripts (full set in step 5): `"lint": "eslint ."`, `"lint:fix": "eslint . --fix"`,
`"format": "prettier --write ."`, `"format:check": "prettier --check ."`.

## 4. React Compiler + incremental TypeScript

**React Compiler** auto-memoizes components, so you rarely hand-write
`useMemo`/`useCallback`. In Vite 8, `@vitejs/plugin-react` transforms with oxc, so
the compiler (a Babel plugin) is opted in via the plugin's `reactCompilerPreset`
helper plus `@rolldown/plugin-babel`:

```bash
pnpm add -D babel-plugin-react-compiler @rolldown/plugin-babel @babel/core @types/babel__core
```

`vite.config.ts`:

```ts
import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config' // typed `test` block, no /// <reference>

export default defineConfig({
  plugins: [react(), babel({ presets: [reactCompilerPreset()] })],
  test: {
    /* step 5 */
  },
})
```

Import `defineConfig` from `vitest/config` (not `vite`) so the `test` block is typed
without a `/// <reference types="vitest/config" />` directive — the import sorter
(step 3) would move that directive below the imports and silently disable it,
breaking the typecheck. The compiler's lint rule ships inside
`eslint-plugin-react-hooks`, which the flat config above extends. Confirm it's active
after build via React DevTools (components show a "Memo ✨" badge).

**Incremental TypeScript**: `create-vite` already uses project-references **build
mode** (`tsc -b`) with a `tsBuildInfoFile` in each `tsconfig.*.json`, so rebuilds
only recheck what changed. Keep `tsc -b` in the build and typecheck scripts:

```jsonc
"build": "tsc -b && vite build",
"typecheck": "tsc -b"
```

## 5. Testing with Vitest + Testing Library

```bash
pnpm add -D vitest jsdom @vitest/coverage-v8 \
  @testing-library/react @testing-library/jest-dom @testing-library/user-event
```

Add the `test` block to `vite.config.ts` (shares Vite's transform pipeline):

```ts
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'html'] },
  },
```

`src/test/setup.ts` — register jest-dom matchers and clean up between tests:

```ts
import '@testing-library/jest-dom/vitest'
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
})
```

Write one real sample test (`src/App.test.tsx`) that renders through the app's
provider and asserts behavior, so `pnpm test` proves the wiring end to end:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider } from '@mui/material/styles'
import App from './App'
import { theme } from './theme'

describe('App', () => {
  it('increments the counter on click', async () => {
    const user = userEvent.setup()
    render(
      <ThemeProvider theme={theme}>
        <App />
      </ThemeProvider>,
    )
    await user.click(screen.getByRole('button', { name: /count is 0/i }))
    expect(
      screen.getByRole('button', { name: /count is 1/i }),
    ).toBeInTheDocument()
  })
})
```

Scripts: `"test": "vitest run"`, `"test:watch": "vitest"`,
`"test:coverage": "vitest run --coverage"`.

## 6. UI framework + theme

**Ask the user for a basic colour set first** — at least primary and
secondary/accent; ideally error and background too. Put them in one theme module so
a rebrand is a one-file change.

Canonical path (MUI):

```bash
pnpm add @mui/material @emotion/react @emotion/styled
```

`src/theme.ts` — barebones, colours from the user:

```ts
import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#1f6feb' },
    secondary: { main: '#a371f7' },
    error: { main: '#d1242f' },
    background: { default: '#ffffff', paper: '#f6f8fa' },
  },
})
```

Wrap the app once in `src/main.tsx`:

```tsx
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { theme } from './theme'
// ...
<ThemeProvider theme={theme}>
  <CssBaseline />
  <App />
</ThemeProvider>
```

Not MUI? Read **only** the one file for the chosen framework — each is
self-contained, so you never load the other frameworks' context:

- Tailwind → [`references/ui-frameworks/tailwind.md`](references/ui-frameworks/tailwind.md)
- Chakra → [`references/ui-frameworks/chakra.md`](references/ui-frameworks/chakra.md)
- Mantine → [`references/ui-frameworks/mantine.md`](references/ui-frameworks/mantine.md)
- none (plain CSS) → [`references/ui-frameworks/none.md`](references/ui-frameworks/none.md)

## 7. Pre-commit hooks — two layers, husky owns the git hook

The repo gets **both** baseline file-hygiene hooks (the language-agnostic
`pre-commit` framework) **and** husky + lint-staged for ESLint/Prettier. The trap: the `pre-commit`
framework and husky both want to own `.git/hooks/pre-commit`. **Resolve it by
letting husky own the hook and calling the pre-commit framework from inside it — so
never run `pre-commit install`** (that would clobber husky's hook).

```bash
pnpm add -D husky lint-staged
pnpm exec husky init      # creates .husky/ and a prepare script
```

Add the `prepare` script (husky v9 sets this up): `"prepare": "husky"`.

`lint-staged` config in `package.json` — **order matters**: ESLint autofix first
(it may reorder imports, etc.), then Prettier as the final formatting authority:

```jsonc
"lint-staged": {
  "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
  "*.{js,jsx,json,css,md,yml,yaml,html}": ["prettier --write"]
}
```

`.husky/pre-commit` — run the two checkers in order (JS/TS first, then hygiene):

```sh
pnpm exec lint-staged

# The pre-commit framework needs Python; skip gracefully if it's absent.
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit run
fi
```

For the hygiene layer, create `.pre-commit-config.yaml` with the standard
file-hygiene hooks, pinning `rev` to the repo's latest tagged release (don't guess —
check the tags). The one JS-repo adjustment: `tsconfig*.json` and `.vscode/*.json`
are **JSONC** (they contain comments), which the strict `check-json` hook rejects —
`exclude` them:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0          # use the latest tagged release
    hooks:
      - id: trailing-whitespace      # strip trailing whitespace
      - id: end-of-file-fixer        # exactly one trailing newline
      - id: mixed-line-ending        # normalize line endings (LF)
        args: [--fix=lf]
      - id: check-yaml               # validate YAML
      - id: check-toml               # validate TOML
      - id: check-json               # validate JSON…
        exclude: ^(tsconfig.*\.json|\.vscode/.*\.json)$   # …but skip JSONC
      - id: check-added-large-files  # block accidental large blobs
      - id: check-merge-conflict     # block leftover conflict markers
      - id: check-case-conflict      # case-insensitive filename clashes
```

The hygiene layer is optional and needs the `pre-commit` tool installed **outside**
the repo (`pipx install pre-commit` or `brew install pre-commit`) — a JS repo has no
Python dep manager to track it, and `.husky/pre-commit` already skips the step when
the binary is absent. Don't run `pre-commit install` — husky owns the git hook.

The first commit (or `pre-commit run --all-files`) will modify files — fixing
trailing newlines on scaffolded SVGs, etc. That's expected; re-stage and commit.

## 8. Editor settings (`.vscode/`)

Format on save with Prettier, autofix ESLint on save:

```jsonc
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": { "source.fixAll.eslint": "explicit" },
  "eslint.useFlatConfig": true,
  "[typescript]": { "editor.defaultFormatter": "esbenp.prettier-vscode" },
  "[typescriptreact]": { "editor.defaultFormatter": "esbenp.prettier-vscode" },
  "typescript.tsdk": "node_modules/typescript/lib"
}
```

```jsonc
// .vscode/extensions.json
{ "recommendations": ["dbaeumer.vscode-eslint", "esbenp.prettier-vscode"] }
```

These files must be **committed** — fix `.gitignore` in step 10 so they aren't
swallowed by the scaffold's `.vscode/*` ignore.

## 9. Env files

Vite exposes only `VITE_`-prefixed vars to the browser via `import.meta.env`. Commit
a template, never the real `.env`:

```bash
# .env.example
VITE_API_BASE_URL=http://localhost:3000
```

Tell the user to `cp .env.example .env`. Never put server-side secrets in a
`VITE_` var — they ship to every client.

## 10. Verify `.gitignore` does the right thing

`create-vite`'s `.gitignore` has two gaps for this setup. Fix both:

- It ignores `.vscode/*` except `extensions.json` — also un-ignore **settings.json**
  so step 8's config ships.
- It does **not** ignore `.env` (only `*.local`) — add `.env` rules but keep the
  example tracked.

```gitignore
coverage

# Env files — keep secrets out, commit the template.
.env
.env.*
!.env.example

# Editor: commit the shared settings + recommendations.
!.vscode/extensions.json
!.vscode/settings.json
```

Verify with `git check-ignore` (it prints ignored paths; exits non-zero for
tracked ones):

```bash
git check-ignore -q .env && echo ".env ignored (good)"
git check-ignore -q .env.example || echo ".env.example tracked (good)"
git check-ignore -q .vscode/settings.json || echo "settings.json tracked (good)"
git check-ignore -q node_modules dist coverage && echo "build dirs ignored (good)"
```

## 11. Dev Docker image (if chosen)

A dev-only image running the Vite dev server with HMR. Keep `FROM` in lockstep with
`.nvmrc` (step 2):

```dockerfile
# Dockerfile.dev
FROM node:24-alpine
RUN corepack enable
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
EXPOSE 5173
CMD ["pnpm", "dev", "--host"]   # --host binds 0.0.0.0 so it's reachable
```

```yaml
# docker-compose.yml
services:
  web:
    build: { context: ., dockerfile: Dockerfile.dev }
    ports: ['5173:5173']
    volumes: ['.:/app', '/app/node_modules'] # live reload, keep container deps
    environment: ['CHOKIDAR_USEPOLLING=true'] # reliable watching on bind mounts
```

`.dockerignore`: `node_modules`, `dist`, `coverage`, `.git`, `.env`, `.env.*`
(but `!.env.example`), `*.log`.

## 12. CI pipeline (if chosen)

Run the same gate everywhere: `install → lint → typecheck → test → build`. For
**GitHub Actions**, `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4 # reads pnpm version from package.json
      - uses: actions/setup-node@v4
        with:
          node-version-file: .nvmrc # single source of truth (step 2)
          cache: pnpm
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm typecheck
      - run: pnpm test
      - run: pnpm build
```

`node-version-file: .nvmrc` is what keeps CI from drifting from local. For
**GitLab** and **Bitbucket** templates (and npm/bun substitutions), read
[`references/ci.md`](references/ci.md). Check for newer action majors before
committing.

## 13. README

Document, so a new contributor needs nothing but the README:

- **Prerequisites** — the node manager + how to use the pinned version
  (`nvm use` / `mise install`), and the package manager (`corepack enable`).
- **Install** — `pnpm install`.
- **Run locally** — `pnpm dev` (and the Docker path if set up).
- **Test** — `pnpm test`, `pnpm test:coverage`.
- **Lint / format** — `pnpm lint`, `pnpm format`; note format-on-save is configured.
- **Env** — `cp .env.example .env`, and what each `VITE_` var does.
- **Build** — `pnpm build`, `pnpm preview`.

## 14. Verify everything is green

Before handing off, run the full gate and a real commit so the hooks are proven:

```bash
pnpm install
pnpm lint
pnpm typecheck
pnpm test
pnpm build
git add -A && git commit -m "chore: scaffold react-ts baseline"   # fires husky → lint-staged → pre-commit
```

If lint-staged or pre-commit modifies files on that first commit, re-stage and
commit again — then it's clean. Fix any red before declaring done.

## Setup checklist

- [ ] asked step 0 decisions (package manager, Node manager + version, UI framework + colours, CI, Docker)
- [ ] scaffolded with `create-vite` react-ts; `packageManager` + `engines` set
- [ ] Node version pinned in **one** node-manager file; CI reads `.nvmrc`; Docker `FROM` matches
- [ ] ESLint flat config + Prettier, `eslint-config-prettier` last; perfectionist sorts imports; `.prettierrc` matches scaffold style
- [ ] React Compiler wired via `reactCompilerPreset` + `@rolldown/plugin-babel`; `tsc -b` incremental build
- [ ] Vitest + Testing Library; `src/test/setup.ts`; one passing sample test
- [ ] UI framework installed; theme/tokens module holds the user's colours
- [ ] husky owns the hook; lint-staged runs `eslint --fix` then `prettier --write`; pre-commit hygiene called from husky (no `pre-commit install`); `check-json` excludes JSONC
- [ ] `.vscode/settings.json` + `extensions.json` committed; format-on-save works
- [ ] `.env.example` committed, `.env` ignored; `VITE_` prefix documented
- [ ] `.gitignore` verified with `git check-ignore`
- [ ] dev Dockerfile + compose (if chosen), `FROM` in lockstep
- [ ] CI pipeline (if chosen) runs install → lint → typecheck → test → build
- [ ] README covers prerequisites / run / test / lint / env / build
- [ ] `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build` all green; test commit fires the hooks clean
