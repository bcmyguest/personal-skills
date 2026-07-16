---
name: rust-setup
description: Scaffold a new Rust repo by copying shipped template files — CI (fmt, clippy -D warnings, locked tests), pre-commit + Conventional Commits, Dependabot, git-cliff auto-release on merge to main with static binaries, crates.io Trusted Publishing (never long-lived tokens), and a tested hello-world mock (CLI bin+lib or pure library). Use only when explicitly asked to scaffold, bootstrap, or set up a new Rust repo, crate, or CLI project — invoke deliberately, e.g. "/rust-setup" or "set up a new rust repo". Not for working in existing Rust projects, and not a trigger for casual Rust questions.
---

# Scaffolding a Rust repo

Everything ships as **real files under [`templates/`](templates/)** — copy them
verbatim, run one substitution pass, make the few marked edits. Do **not**
retype configs from memory or improvise structure: the templates are the
source of truth (lifted from a working repo,
[powerline-claude](https://github.com/bcmyguest/powerline-claude)), and the
decisions below only change *which* files get copied.

What the result looks like: logic in `src/lib.rs` (testable, hello-world mock
with unit + integration tests), CI failing on any warning, hygiene +
Conventional Commits enforced at commit time, Dependabot keeping deps and
actions current, and — if chosen — every merge to `main` auto-releasing via
git-cliff (conventional commits → semver → tag → GitHub release → crates.io).

## 0. Ask the decisions

| Decision            | Options                                        | Default              |
| ------------------- | ---------------------------------------------- | -------------------- |
| Crate name          | kebab-case; must be free on crates.io if publishing | ask             |
| GitHub repo slug    | `owner/repo`                                   | ask                  |
| One-line description| —                                              | ask                  |
| Crate type          | CLI (bin + lib) · library                      | **CLI**              |
| License             | `MIT OR Apache-2.0` · `AGPL-3.0-only` · other  | **MIT OR Apache-2.0** |
| Release automation  | auto-release every merge to `main` · none      | **auto-release**     |
| Publish to crates.io| yes (**Trusted Publishing only**) · no         | ask                  |
| Binary targets      | linux-musl x64/arm64 + macOS x64/arm64 (Windows row ships commented) | the 4 |

## 1. Copy the templates

`$SKILL` below is this skill's directory. Copy order matters — later layers
overlay earlier ones.

```bash
# Always — the base layer (note the /. so dotfiles come along):
cp -R "$SKILL/templates/base/." .

# Crate type — exactly one:
cp -R "$SKILL/templates/cli/." .                 # CLI: Cargo.toml, src/main.rs, tests/cli.rs
cp "$SKILL/templates/lib/Cargo.toml" .           # library: Cargo.toml only

# Release automation (skip all three lines entirely if "none"; $RS = the
# installed release-setup skill, a sibling of $SKILL — it owns the canonical
# cliff.toml and the release doctrine):
cp "$RS/templates/cliff.toml" .
mkdir -p .github/workflows
cp "$SKILL/templates/release/<variant>" .github/workflows/release.yml

# Pre-commit — assembled from the pre-commit-setup skill's shared fragments
# plus this skill's Rust layer ($PCS = the installed pre-commit-setup skill,
# a sibling of $SKILL; drop the conventional-commits line if releases: none):
{ echo 'repos:'; cat "$PCS/templates/hygiene.repos.yaml" \
    "$PCS/templates/conventional-commits.repos.yaml" \
    "$SKILL/templates/pre-commit-rust.repos.yaml"; } > .pre-commit-config.yaml
```

The pre-commit machinery (install, hook types, merge-into-existing) is the
**pre-commit-setup** skill's job — follow it with the assembled config, and
the release doctrine (commit types → semver, notes, no hand-bumped versions)
is the **release-setup** skill's. If either isn't installed alongside this
one, fetch the missing files from the same repo this skill came from.

Release variants — pick one:

| Crate type | crates.io | Copy                                            |
| ---------- | --------- | ----------------------------------------------- |
| CLI        | yes       | `release/release-cli-crates.yml`                |
| CLI        | no        | `release/release-cli-only.yml`                  |
| library    | yes       | `release/release-lib-crates.yml`                |
| library    | no        | `release-cli-only.yml`, then delete the `build` job and the binary/checksum steps in `release` |

## 2. Fill the placeholders

Five placeholders, one pass (adjust values; `{{crate_snake}}` is the crate
name with `-` → `_`):

```bash
grep -rlF '{{' --exclude-dir=target . | xargs sed -i \
  -e 's/{{crate_name}}/my-tool/g' \
  -e 's/{{crate_snake}}/my_tool/g' \
  -e 's|{{repo_slug}}|owner/my-tool|g' \
  -e 's/{{description}}/One-line description/g' \
  -e 's|{{license}}|MIT OR Apache-2.0|g'
```

(macOS/BSD sed: `sed -i ''`.) Afterwards `grep -rF '{{' .` must only hit
GitHub-Actions `${{ … }}` expressions and cliff.toml's Tera template.

## 3. LICENSE file(s) and marked edits

- `MIT OR Apache-2.0` → write **both** `LICENSE-MIT` and `LICENSE-APACHE`
  (canonical texts, copyright line = current year + author). Single license →
  one `LICENSE` file.
- Work through every `<!-- DELETE … -->` / `<!-- KEEP … -->` marker in
  `README.md` and `CLAUDE.md` — they mark the branch-dependent blocks (badges,
  install methods, Releases section wording).
- `Cargo.toml`: fill `keywords` / `categories` (or leave empty), and drop the
  `authors`-adjacent metadata you don't want.

## 4. Refresh the moving parts — don't trust shipped pins

The templates pin versions that were current when the skill was written; the
scaffold should start from *today's*:

- `pre-commit autoupdate --freeze` — the **pre-commit-setup** skill owns the
  hook-update / supply-chain policy (immutable SHAs, never plain
  `autoupdate`); follow its step 4.
- Check the `uses:` action majors in the copied workflows against upstream
  latest; bump if a new major exists. After the first push, **Dependabot owns
  this** (`.github/dependabot.yml` covers `cargo` + `github-actions`, weekly,
  grouped).
- `cargo generate-lockfile` (or the first `cargo test`) writes `Cargo.lock` —
  **commit it**, CI and releases run `--locked`.

## 5. Verify green, then first commit

```bash
git init -b main
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test          # unit (lib.rs) + integration (tests/) + binary e2e (CLI)
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
git add -A && git commit -m "feat: scaffold <crate-name>"
```

The commit message **must** be conventional — git-cliff derives every version
bump from commit types, starting with this one (`feat:` → the initial
`v0.1.0`). Fix any red before handing off; the mock `greet` code exists so all
three test layers demonstrably pass, and gets replaced by real code later.

## 6. Publishing to crates.io — Trusted Publishing ONLY

Never store a crates.io token in GitHub secrets: `release.yml` exchanges the
workflow run's OIDC identity for a short-lived token via
`rust-lang/crates-io-auth-action` (that's the `id-token: write` permission).
One crates.io limitation shapes the order: **a brand-new crate's first publish
must be manual** — trusted publishers can only be configured on a crate that
already exists. So:

1. **Bootstrap publish, once, locally:** on crates.io create a token scoped to
   *publish-new* with the shortest expiry offered, `cargo login`,
   `cargo publish`, then **revoke the token immediately**.
2. **Register the trusted publisher:** crates.io → the crate → *Settings* →
   *Trusted Publishing* → *GitHub*: repository owner + name, workflow filename
   `release.yml`, environment blank (unless you add one).
3. **Enforce it:** in the same crate settings, enable *Require Trusted
   Publishing* so token-based publishes are rejected outright.
4. **Now push to GitHub.** The release run tags `v0.1.0` and publishes the
   GitHub release; the crates.io step sees 0.1.0 already published and skips —
   every later merge publishes via OIDC only.

If a publish step ever fails (e.g. publisher not registered yet), the tag and
GitHub release have already landed — register the publisher and re-run the
workflow via `workflow_dispatch`, which republishes the latest tag idempotently.

## 7. After the push

- Confirm the `ci` and `release` runs are green and Dependabot is active
  (Insights → Dependency graph → Dependabot).
- The repo's `CLAUDE.md` ships as a skeleton — once real code replaces the
  mock, rewrite the description and add an Architecture section (`/init`
  drafts it); `AGENTS.md` just includes `CLAUDE.md`.

## Checklist

- [ ] decisions asked (name, slug, description, type, license, releases, crates.io)
- [ ] base layer + crate-type layer + release variant copied, nothing retyped; pre-commit config assembled from pre-commit-setup fragments + the Rust layer
- [ ] placeholder pass done; `grep -rF '{{'` shows only `${{ }}` / Tera hits
- [ ] LICENSE file(s) written; all `<!-- DELETE/KEEP -->` markers resolved
- [ ] `pre-commit autoupdate --freeze` run; action majors checked; `Cargo.lock` committed
- [ ] fmt + clippy + test green; hooks installed (pre-commit **and** commit-msg); first commit conventional
- [ ] crates.io (if publishing): bootstrap publish → token revoked → trusted publisher registered → *Require Trusted Publishing* enabled
- [ ] first release run green: tag + GitHub release (+ binaries), crates.io via OIDC only
