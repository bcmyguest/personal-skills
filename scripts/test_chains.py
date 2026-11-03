#!/usr/bin/env python3
"""Each chain fault makes `chains.py --check` fail. Run: python3 scripts/test_chains.py"""

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY = ["chains", "scripts", "skills", "wteam", "CHAINS.md", "README.md"]


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
        self.edit("CHAINS.md", "## Teams", "## Teams (edited)")
        self.assert_fails("CHAINS.md is stale")

    def test_team_step_with_unknown_skill(self):
        self.edit("wteam/teams/port/team.dot", 'skill="port-changes"', 'skill="port-it"')
        self.assert_fails("node worker has skill='port-it', not a skill in skills/")

    def test_next_item_with_unknown_relationship(self):
        self.edit("skills/brief-agent/SKILL.md", "- `next-agent` (suggested)", "- `next-agent` (maybe)")
        self.assert_fails("Next `next-agent` has (maybe)")

    def test_next_item_with_unknown_target(self):
        self.edit("skills/brief-agent/SKILL.md", "- `next-agent` (suggested)", "- `nowhere` (suggested)")
        self.assert_fails("Next `nowhere` is neither a skill here nor a known external")

    def test_team_hand_off_no_next_section_states(self):
        self.edit("wteam/teams/port/team.dot", 'skill="audit-agent-diff"', 'skill="land-change"')
        self.assert_fails("worker (port-changes) hands to reviewer (land-change)")

    def test_later_step_may_rely_on_the_step_workers_next(self):
        # independent-review hands to run-experiment only through plan-experiment's Next.
        self.edit("skills/plan-experiment/SKILL.md", "- `run-experiment` (prerequisite): the approved plan.\n", "")
        self.assert_fails("plan_review (independent-review) hands to run (run-experiment)")


class Routing(unittest.TestCase):
    def test_every_method_skill_has_examples_with_known_targets(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import chains

        skills = chains.load_skills()
        text = (ROOT / "chains" / "routing.md").read_text()
        sections = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", text, re.M | re.S))
        method = [s for s, v in skills.items() if chains.next_items(v["text"]) is not None]
        for sid in method:
            self.assertIn(sid, sections, f"routing.md has no section for {sid}")
            self.assertIn("Routes here:", sections[sid], sid)
            self.assertIn("Near-misses:", sections[sid], sid)
        known = set(skills) | set(chains.EXTERNAL) | {"none"}
        for target in re.findall(r"→ `([^`]+)`", text):
            self.assertIn(target, known, f"routing.md near-miss target {target} is unknown")


if __name__ == "__main__":
    unittest.main()
