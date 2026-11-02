#!/usr/bin/env python3
"""Each registry fault makes `chains.py --check` fail. Run: python3 scripts/test_chains.py"""

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY = ["chains", "scripts", "skills", "CHAINS.md", "README.md"]


class CheckMode(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in COPY:
            src = ROOT / name
            (shutil.copytree if src.is_dir() else shutil.copy)(src, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def check(self):
        return subprocess.run(
            [sys.executable, str(self.tmp / "scripts" / "chains.py"), "--check"],
            capture_output=True,
            text=True,
        )

    def edit(self, rel, old, new):
        p = self.tmp / rel
        text = p.read_text()
        self.assertIn(old, text)
        p.write_text(text.replace(old, new, 1))

    def assert_fails(self, needle):
        r = self.check()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(needle, r.stderr)

    def test_clean_copy_passes(self):
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_stale_output(self):
        self.edit("CHAINS.md", "## Legend", "## Legend (edited)")
        self.assert_fails("CHAINS.md is stale")

    def test_unknown_skill_id(self):
        self.edit("chains/skills.dot", '"land-change"              [type=skill]', '"land-it" [type=skill]')
        self.assert_fails("skills/land-it/SKILL.md does not exist")

    def test_unclassified_relationship(self):
        self.edit("chains/skills.dot", "rel=suggested, origin=inferred, artifact=\"brief\"", "origin=inferred, artifact=\"brief\"")
        self.assert_fails("rel=None")

    def test_missing_origin(self):
        self.edit("chains/skills.dot", "rel=suggested, origin=inferred, artifact=\"brief\"", "rel=suggested, artifact=\"brief\"")
        self.assert_fails("origin=None")

    def test_broken_source_anchor(self):
        self.edit("chains/skills.dot", "skills/run-experiment/SKILL.md#method", "skills/run-experiment/SKILL.md#nowhere")
        self.assert_fails("has no #nowhere heading")

    def test_next_section_out_of_sync(self):
        self.edit("skills/brief-agent/SKILL.md", "- `next-agent` (suggested)", "- `land-change` (suggested)")
        self.assert_fails("Next lists `land-change`")

    def test_team_step_without_edge(self):
        self.edit("chains/skills.dot", '"port-changes > audit-agent-diff"', '"port-changes > land-change"')
        self.assert_fails("team:port: no transition port-changes -> land-change")


class Routing(unittest.TestCase):
    def test_every_method_skill_has_examples_with_known_targets(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import chains

        graph = chains.parse_dot(chains.REGISTRY.read_text())
        skills = chains.load_skills()
        text = (ROOT / "chains" / "routing.md").read_text()
        sections = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", text, re.M | re.S))
        method = [s for s, v in skills.items() if chains.next_items(v["text"]) is not None]
        for sid in method:
            self.assertIn(sid, sections, f"routing.md has no section for {sid}")
            self.assertIn("Routes here:", sections[sid], sid)
            self.assertIn("Near-misses:", sections[sid], sid)
        known = set(skills) | set(graph["nodes"]) | {"none"}
        for target in re.findall(r"→ `([^`]+)`", text):
            self.assertIn(target, known, f"routing.md near-miss target {target} is unknown")


if __name__ == "__main__":
    unittest.main()
