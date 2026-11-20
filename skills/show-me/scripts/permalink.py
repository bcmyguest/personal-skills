#!/usr/bin/env python3
"""Print a code reference: `path:start-end`, a permalink to those lines, and the lines themselves.

Usage: permalink.py PATH:START[-END] [--remote NAME] [--forge github|gitlab|azure|bitbucket|gitea]

The snippet is read from the file on disk, so it always matches. The permalink pins the
current commit and is printed only when the lines are committed and the commit is on a
remote; otherwise the reason is printed in its place. Exit 1 on a bad path or range.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

FORGES = ("github", "gitlab", "azure", "bitbucket", "gitea")
LANGS = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".rs": "rust", ".go": "go", ".sh": "bash", ".md": "markdown",
    ".json": "json", ".toml": "toml", ".yml": "yaml", ".yaml": "yaml", ".java": "java",
    ".cs": "csharp", ".rb": "ruby", ".sql": "sql", ".html": "html", ".css": "css",
}
RENDERED = {".md", ".markdown", ".rst", ".adoc", ".ipynb", ".org"}


class RefError(Exception):
    pass


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def parse_remote(url):
    """Return (forge, base_url, extra) from a git remote URL, forge None when unknown."""
    url = url.strip()
    m = re.match(r"(?:ssh://)?git@ssh\.dev\.azure\.com(?::v3)?/([^/]+)/([^/]+)/([^/]+?)/?$", url)
    if m:
        org, project, repo = m.groups()
        return "azure", f"https://dev.azure.com/{org}/{project}/_git/{repo}", None
    m = re.match(r"https?://(?:[^@/]+@)?dev\.azure\.com/([^/]+)/([^/]+)/_git/([^/]+?)/?$", url)
    if m:
        org, project, repo = m.groups()
        return "azure", f"https://dev.azure.com/{org}/{project}/_git/{repo}", None
    m = re.match(r"https?://(?:[^@/]+@)?([^./]+)\.visualstudio\.com/(?:DefaultCollection/)?([^/]+)/_git/([^/]+?)/?$", url)
    if m:
        org, project, repo = m.groups()
        return "azure", f"https://dev.azure.com/{org}/{project}/_git/{repo}", None
    m = re.match(r"(?:ssh://)?(?:[^@/]+@)?([^:/]+)(?::\d+)?[:/](.+?)(?:\.git)?/?$", url)
    if url.startswith(("http://", "https://")):
        m = re.match(r"https?://(?:[^@/]+@)?([^/]+)/(.+?)(?:\.git)?/?$", url)
    if not m:
        return None, None, None
    host, path = m.group(1).split(":")[0], m.group(2)
    lower = host.lower()
    forge = None
    if "github" in lower:
        forge = "github"
    elif "gitlab" in lower:
        forge = "gitlab"
    elif "bitbucket" in lower:
        forge = "bitbucket"
    elif any(k in lower for k in ("gitea", "forgejo", "codeberg")):
        forge = "gitea"
    return forge, f"https://{host}/{path}", None


def build_link(forge, base, sha, path, start, end):
    p = quote(path)
    rendered = Path(path).suffix.lower() in RENDERED
    if forge == "github":
        return f"{base}/blob/{sha}/{p}{'?plain=1' if rendered else ''}#L{start}" + (f"-L{end}" if end != start else "")
    if forge == "gitlab":
        return f"{base}/-/blob/{sha}/{p}{'?plain=1' if rendered else ''}#L{start}" + (f"-{end}" if end != start else "")
    if forge == "bitbucket":
        return f"{base}/src/{sha}/{p}#lines-{start}" + (f":{end}" if end != start else "")
    if forge == "gitea":
        return f"{base}/src/commit/{sha}/{p}{'?display=source' if rendered else ''}#L{start}" + (f"-L{end}" if end != start else "")
    if forge == "azure":
        return (f"{base}?path=/{p}&version=GC{sha}&line={start}&lineEnd={end + 1}"
                f"&lineStartColumn=1&lineEndColumn=1&lineStyle=plain&_a=contents")
    raise ValueError(forge)


def reference(spec, remote="origin", forge=None, cwd=None):
    """Return the markdown block for SPEC (`path:start[-end]`)."""
    m = re.match(r"^(.+?):(\d+)(?:-(\d+))?$", spec)
    if not m:
        raise RefError(f"expected PATH:START[-END], got {spec!r}")
    file = Path(cwd or ".", m.group(1)).resolve()
    start = int(m.group(2))
    end = int(m.group(3) or start)
    if not file.is_file():
        raise RefError(f"{m.group(1)}: no such file")
    lines = file.read_text(errors="replace").splitlines()
    if start < 1 or end < start or end > len(lines):
        raise RefError(f"{m.group(1)}: lines {start}-{end} outside 1-{len(lines)}")

    link, reason = None, None
    rc, top = git(file.parent, "rev-parse", "--show-toplevel")
    if rc != 0:
        reason = "not in a git repository"
        rel = m.group(1)
    else:
        rel = file.relative_to(Path(top).resolve()).as_posix()
        _, sha = git(top, "rev-parse", "HEAD")
        _, url = git(top, "remote", "get-url", remote)
        detected, base, _ = parse_remote(url) if url else (None, None, None)
        forge = forge or detected
        if not url:
            reason = f"no remote named {remote}"
        elif git(top, "ls-files", "--error-unmatch", rel)[0] != 0:
            reason = "file is not committed"
        elif git(top, "diff", "--quiet", "HEAD", "--", rel)[0] != 0:
            reason = "file has uncommitted changes, so the lines may differ from the commit"
        elif not git(top, "for-each-ref", "--contains", sha, "refs/remotes")[1]:
            reason = f"commit {sha[:12]} is not on any remote branch; push it first"
        elif not forge:
            reason = f"unknown forge for {url}; pass --forge {'|'.join(FORGES)}"
        else:
            link = build_link(forge, base, sha, rel, start, end)

    span = f"{start}" if start == end else f"{start}-{end}"
    head = f"`{rel}:{span}`" + (f" ({link})" if link else f" (no permalink: {reason})")
    lang = LANGS.get(file.suffix.lower(), "")
    body = "\n".join(lines[start - 1:end])
    fence = "````" if "```" in body else "```"
    return f"{head}\n\n{fence}{lang}\n{body}\n{fence}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", nargs="+", help="PATH:START[-END]")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--forge", choices=FORGES)
    args = ap.parse_args(argv)
    status = 0
    for spec in args.spec:
        try:
            print(reference(spec, args.remote, args.forge))
            print()
        except RefError as e:
            print(f"error: {e}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
