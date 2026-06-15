# CI pipeline templates

The **GitHub Actions** workflow is in SKILL.md (the common case). Use the matching
template below for GitLab or Bitbucket. Every template runs the same gate —
`install → lint → typecheck → test → build` — and every one pins the **same Node
major as `.nvmrc`** so CI can't drift from local. Bump the image tag here whenever
you bump `.nvmrc`.

All templates assume **pnpm**. Package-manager substitutions are at the bottom.

## GitLab CI — `.gitlab-ci.yml`

```yaml
default:
  image: node:24-alpine # keep in lockstep with .nvmrc
  before_script:
    - corepack enable
    - pnpm config set store-dir .pnpm-store
    - pnpm install --frozen-lockfile
  cache:
    key:
      files: [pnpm-lock.yaml]
    paths: [.pnpm-store]

stages: [verify]

lint:
  stage: verify
  script: [pnpm lint]

typecheck:
  stage: verify
  script: [pnpm typecheck]

test:
  stage: verify
  script: [pnpm test]

build:
  stage: verify
  script: [pnpm build]
  artifacts:
    paths: [dist]
    expire_in: 1 week
```

## Bitbucket Pipelines — `bitbucket-pipelines.yml`

```yaml
image: node:24 # keep in lockstep with .nvmrc

definitions:
  caches:
    pnpm: .pnpm-store

pipelines:
  default:
    - step:
        name: Verify
        caches: [pnpm]
        script:
          - corepack enable
          - pnpm config set store-dir .pnpm-store
          - pnpm install --frozen-lockfile
          - pnpm lint
          - pnpm typecheck
          - pnpm test
          - pnpm build
        artifacts:
          - dist/**
```

## Package-manager substitutions

The pipelines above call `pnpm <script>`. For another package manager, swap the
install command and the `pnpm run` prefix — the script names (`lint`, `typecheck`,
`test`, `build`) stay identical.

| Step    | pnpm                          | npm                  | bun                              |
| ------- | ----------------------------- | -------------------- | -------------------------------- |
| enable  | `corepack enable`             | (built in)           | use `oven-sh/setup-bun` / image  |
| install | `pnpm install --frozen-lockfile` | `npm ci`          | `bun install --frozen-lockfile`  |
| run     | `pnpm lint`                   | `npm run lint`       | `bun run lint`                   |

For **GitHub Actions** with npm, drop `pnpm/action-setup` and set
`cache: npm` on `actions/setup-node`; with bun, replace both with
`oven-sh/setup-bun@v2`. Keep `node-version-file: .nvmrc` so the Node pin stays in
one place regardless of package manager.
