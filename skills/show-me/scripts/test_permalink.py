#!/usr/bin/env python3
"""Tests for permalink.py: remote parsing per forge, link shapes, and the no-link reasons."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import permalink as pl  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class ParseRemote(unittest.TestCase):
    def test_forges(self):
        cases = {
            "git@github.com:o/r.git": ("github", "https://github.com/o/r"),
            "https://github.com/o/r": ("github", "https://github.com/o/r"),
            "git@gitlab.com:g/sub/p.git": ("gitlab", "https://gitlab.com/g/sub/p"),
            "https://gitlab.example.com/g/p.git": ("gitlab", "https://gitlab.example.com/g/p"),
            "ssh://git@gitlab.example.com:2222/g/p.git": ("gitlab", "https://gitlab.example.com/g/p"),
            "git@bitbucket.org:ws/r.git": ("bitbucket", "https://bitbucket.org/ws/r"),
            "https://codeberg.org/o/r.git": ("gitea", "https://codeberg.org/o/r"),
            "git@ssh.dev.azure.com:v3/org/proj/repo": ("azure", "https://dev.azure.com/org/proj/_git/repo"),
            "https://org@dev.azure.com/org/proj/_git/repo": ("azure", "https://dev.azure.com/org/proj/_git/repo"),
            "https://org.visualstudio.com/proj/_git/repo": ("azure", "https://dev.azure.com/org/proj/_git/repo"),
            "git@git.internal:team/r.git": (None, "https://git.internal/team/r"),
        }
        for url, want in cases.items():
            with self.subTest(url=url):
                self.assertEqual(pl.parse_remote(url)[:2], want)


class BuildLink(unittest.TestCase):
    def test_shapes(self):
        b = "https://h/o/r"
        self.assertEqual(pl.build_link("github", b, SHA, "a/b.py", 3, 5), f"{b}/blob/{SHA}/a/b.py#L3-L5")
        self.assertEqual(pl.build_link("github", b, SHA, "README.md", 4, 4), f"{b}/blob/{SHA}/README.md?plain=1#L4")
        self.assertEqual(pl.build_link("gitlab", b, SHA, "a.py", 3, 5), f"{b}/-/blob/{SHA}/a.py#L3-5")
        self.assertEqual(pl.build_link("bitbucket", b, SHA, "a.py", 3, 5), f"{b}/src/{SHA}/a.py#lines-3:5")
        self.assertEqual(pl.build_link("gitea", b, SHA, "a.py", 3, 5), f"{b}/src/commit/{SHA}/a.py#L3-L5")
        self.assertEqual(
            pl.build_link("azure", b, SHA, "src/a.py", 3, 5),
            f"{b}?path=/src/a.py&version=GC{SHA}&line=3&lineEnd=6&lineStartColumn=1&lineEndColumn=1&lineStyle=plain&_a=contents",
        )
        self.assertIn("a%20b.py", pl.build_link("github", b, SHA, "a b.py", 1, 1))


def sh(cwd, *cmd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


class Reference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bare, self.repo = root / "remote.git", root / "work"
        sh(root, "git", "init", "-q", "--bare", str(self.bare))
        sh(root, "git", "init", "-q", str(self.repo))
        sh(self.repo, "git", "config", "user.email", "t@t")
        sh(self.repo, "git", "config", "user.name", "t")
        (self.repo / "a.py").write_text("one\ntwo\nthree\nfour\n")
        sh(self.repo, "git", "add", ".")
        sh(self.repo, "git", "commit", "-qm", "init")
        sh(self.repo, "git", "remote", "add", "origin", str(self.bare))

    def tearDown(self):
        self.tmp.cleanup()

    def ref(self, spec, **kw):
        return pl.reference(spec, cwd=self.repo, **kw)

    def push(self):
        sh(self.repo, "git", "push", "-q", "origin", "HEAD:main")

    def test_snippet_comes_from_disk(self):
        out = self.ref("a.py:2-3", forge="github")
        self.assertTrue(out.endswith("```python\ntwo\nthree\n```"))

    def test_unpushed_commit_gets_no_link(self):
        self.assertIn("is not on any remote branch", self.ref("a.py:1", forge="github"))

    def test_pushed_commit_gets_link(self):
        self.push()
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True).stdout.strip()
        self.assertIn(f"/blob/{sha}/a.py#L1-L2)", self.ref("a.py:1-2", forge="github"))

    def test_pushed_commit_without_tracking_ref_gets_link(self):
        # Single-branch and shallow clones have no tracking ref for a pushed feature branch.
        sh(self.repo, "git", "push", "-q", "origin", "HEAD:feature")
        sh(self.repo, "git", "update-ref", "-d", "refs/remotes/origin/feature")
        self.assertIn("/blob/", self.ref("a.py:1", forge="github"))

    def test_older_pushed_commit_gets_link(self):
        self.push()
        (self.repo / "b.py").write_text("x\n")
        sh(self.repo, "git", "add", ".")
        sh(self.repo, "git", "commit", "-qm", "two")
        self.push()
        sh(self.repo, "git", "update-ref", "-d", "refs/remotes/origin/main")
        sh(self.repo, "git", "reset", "-q", "--hard", "HEAD~1")
        self.assertIn("/blob/", self.ref("a.py:1", forge="github"))

    def test_unknown_forge_asks_for_flag(self):
        self.push()
        self.assertIn("pass --forge", self.ref("a.py:1"))

    def test_dirty_file_gets_no_link(self):
        self.push()
        (self.repo / "a.py").write_text("ONE\ntwo\nthree\nfour\n")
        self.assertIn("uncommitted changes", self.ref("a.py:1", forge="github"))

    def test_bad_range_and_path_raise(self):
        with self.assertRaises(pl.RefError):
            self.ref("a.py:3-9")
        with self.assertRaises(pl.RefError):
            self.ref("missing.py:1")
        with self.assertRaises(pl.RefError):
            self.ref("a.py")


if __name__ == "__main__":
    unittest.main()
