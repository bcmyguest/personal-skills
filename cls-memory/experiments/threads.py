"""Pin the CPU thread count so a DG/SEPARATED number is reproducible.

Defect 2 of the 2026-08 review. The hippocampal DG key thresholds on
`h.topk(sparsity_k).values[..., -1:]`, so a unit sitting within float noise of
that cut flips in or out when the matmul reduction order changes -- and the
reduction order changes with the intra-op thread count. The key itself is
therefore a function of how many threads torch happened to grab.

Measured, LoCoMo 3 conversations, dim=1024, key=SEPARATED, beta=128:

    6 threads -> hit@1 0.2976 / @5 0.3623 / @10 0.3988
    8 threads -> hit@1 0.2895 / @5 0.3502 / @10 0.4008

Deterministic at a fixed thread count, not portable across them. The spread
(0.008 at hit@1, n=494) sits inside the project's ~0.04 resolution limit, so it
overturns no conclusion -- but without a pinned count no third party can
reproduce a published DG figure at all.

Both EMBEDDING arms are immune: no top-k selection, nothing to flip.

This pins the experiment harnesses only. It is deliberately not a library
change: `cls_memory` does not set thread counts, and a caller who wants
reproducible DG keys is the one who has to pin.
"""

from __future__ import annotations

import argparse

import torch

DEFAULT_THREADS = 6


def add_threads_arg(parser: argparse.ArgumentParser) -> None:
    """Add the `--threads` flag every experiment in this review shares."""
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"intra-op torch threads (default: {DEFAULT_THREADS}). DG/"
             f"SEPARATED results are only reproducible at a fixed count; see "
             f"experiments/threads.py",
    )


def pin_threads(threads: int) -> int:
    """Pin intra-op threads and return the count torch actually settled on.

    Returns `torch.get_num_threads()` rather than the requested `threads`
    because torch may clamp it. Callers record the returned value, never the
    request -- publishing the number that was asked for instead of the one that
    was used is exactly the failure this module exists to prevent.
    """
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")
    torch.set_num_threads(threads)
    return torch.get_num_threads()
