"""Guards for the defect-2 thread pinning.

The failure this pins against is not a crash -- it is a published DG number
that nobody else can reproduce, because the harness never recorded how many
threads torch happened to grab. So there are two things worth testing and they
are different: that `pin_threads` reports the count that is actually in force
(not the one that was requested), and that every harness in this review is
wired to call it at all.

The second is a source-level check on purpose. A DG-bearing script that quietly
loses its `--threads` flag would still run, still print, still produce
plausible numbers -- and reintroduce the whole defect silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from experiments.threads import DEFAULT_THREADS, add_threads_arg, pin_threads

# Every harness written or touched by the 2026-08 review that can reach the
# DG/SEPARATED key path, and so must record its thread count.
PINNED_SCRIPTS = (
    "separation_beta_sweep.py",
    "completion_check.py",
    "gist_check.py",
    "abstention_recheck.py",
    "kv_learned_readin.py",
    # Not new in this review, but they carry the DG-bearing tables that the
    # write-up publishes, so they are pinned on the same terms.
    "recall_check.py",
    "separation_check.py",
)


@pytest.fixture(autouse=True)
def restore_thread_count():
    """Put the process's thread count back.

    `torch.set_num_threads` is global and permanent for the process, so a test
    that pins to 1 and walks away would silently slow down every test that runs
    after it -- including the timing-sensitive ones.
    """
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def test_pin_threads_reports_the_count_actually_in_force():
    """The return value must come from torch, not from the argument.

    Returning the request would defeat the point: the number that gets
    published has to be the number that was used, even if torch clamps it.
    """
    returned = pin_threads(2)
    assert returned == torch.get_num_threads()
    assert returned == 2


def test_pin_threads_rejects_a_nonsense_count():
    with pytest.raises(ValueError):
        pin_threads(0)


def test_default_is_the_documented_count():
    parser = argparse.ArgumentParser()
    add_threads_arg(parser)
    assert parser.parse_args([]).threads == DEFAULT_THREADS
    assert parser.parse_args(["--threads", "8"]).threads == 8


@pytest.mark.parametrize("script", PINNED_SCRIPTS)
def test_every_dg_bearing_harness_pins_and_records(script):
    source = (Path(__file__).parent.parent / "experiments" / script).read_text()
    assert "add_threads_arg(parser)" in source, f"{script} has no --threads flag"
    assert "pin_threads(args.threads)" in source, f"{script} never pins"
    # The count has to reach the reader, not just the thread pool.
    assert "threads}" in source or '"threads": threads' in source, (
        f"{script} pins threads but never reports the count it used"
    )
