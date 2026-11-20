"""Does pattern completion survive real text? (review ticket 10)

    PYTHONPATH=. uv run python -u experiments/completion_check.py --corpus locomo

RESULTS.md section 4 reports perfect completion from 26 of 205 active units on
the **synthetic** corpus. Part II already retracted the synthetic *retrieval*
numbers as artifacts of the generator; the completion number is the same class
of number and has never faced the same test. This runs the identical protocol
on real corpora, and adds the arm section 4 is missing.

The missing arm is the whole point
----------------------------------
Section 4 measured settling alone. It never asked what plain cosine does with
the *same degraded cue*. That comparison is the only thing that can turn
"completion works" into "completion beats nearest-neighbour", because masking
coordinates of a sparse key leaves a vector that still points mostly at its
source: a one-shot cosine lookup may well recover the memory with no attractor
dynamics at all. Both arms see byte-identical cues here.

Protocols, as in section 4
--------------------------
  (a) coordinates occluded    -- weak on sparse keys, where most coordinates
                                 are zero anyway, but it is the only protocol
                                 defined for dense embedding keys
  (b) active units occluded   -- the honest difficulty scale for SEPARATED
                                 keys (2048-d, ~256 active), and the protocol
                                 the published table's headline came from

Reported per cell: completion recall@1, the single-shot cosine arm on the same
cue, cue cosine to the source, iterations to settle, whether the clamp held,
and the mixture rate -- so "settled onto the right memory" is distinguishable
from "settled onto one global mixture", which at low beta it usually is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from cls_memory import HippocampalKey, MemoryRecord, Persistence
from experiments import locomo, qmsum
from experiments.recall_check import SEED
from experiments.threads import add_threads_arg, pin_threads
from experiments.separation_beta_sweep import build_embedder, build_system

ARMS: list[tuple[str, HippocampalKey, bool]] = [
    ("baseline (dense key)", HippocampalKey.EMBEDDING, False),
    ("whitened (dense key)", HippocampalKey.EMBEDDING, True),
    ("DG (sparse key)", HippocampalKey.SEPARATED, False),
]

# The published table's ladder, extended one step down. 0.10 is the cell the
# headline claim quotes (26 of 205 active units).
KEEPS: tuple[float, ...] = (0.80, 0.50, 0.30, 0.20, 0.10, 0.05)


def _occlude_coordinates(key: Tensor, keep: float, *, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(key.shape, generator=g) < keep


def _occlude_active_units(key: Tensor, keep: float, *, seed: int) -> Tensor:
    """Keep a random `keep` fraction of the NONZERO units; drop the rest.

    Zero coordinates stay 'known' (they are known -- they are zero), which is
    what makes this the harder scale: the cue loses actual signal rather than
    padding.
    """
    g = torch.Generator().manual_seed(seed)
    active = key.abs() > 0
    drop = (torch.rand(key.shape, generator=g) >= keep) & active
    return ~drop


def _blank() -> dict:
    return {"n": 0, "completed": 0, "cosine": 0, "cue_cos": 0.0, "iters": 0.0,
            "clamp_exact": 0, "mixture": 0, "units": 0.0, "converged": 0}


def measure_arm(
    conversations,
    *,
    dim: int,
    key_mode: HippocampalKey,
    whiten: bool,
    beta: float,
    keeps: tuple[float, ...],
    protocol: str,
    per_conv: int,
) -> dict:
    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = build_embedder(corpus, dim=dim, whiten=whiten)
    occlude = _occlude_coordinates if protocol == "coords" else _occlude_active_units
    cells = {k: _blank() for k in keeps}

    for conv in conversations:
        system = build_system(embedder, conv, key_mode=key_mode)
        store = system.store
        mhn = store.mhn
        for turn in conv.turns:
            embedding = embedder.encode([turn.memory_text])[0]
            if float(embedding.norm()) < 1e-8:
                continue
            latent = system.cortex.latent(embedding)
            store.add(
                MemoryRecord(
                    text=turn.memory_text,
                    embedding=embedding,
                    latent=latent,
                    key=system.key_encoder(embedding, latent),
                    persistence=Persistence.TEMPORAL,
                    created_at=turn.timestamp,
                    last_reinforced_at=turn.timestamp,
                ),
                now=turn.timestamp,
            )
        now = max(t.timestamp for t in conv.turns)
        store.refresh_priors(now)

        # A deterministic spread of source memories rather than the first N:
        # the head of a conversation is systematically different from its tail.
        n = len(store)
        stride = max(1, n // per_conv)
        targets = list(range(0, n, stride))[:per_conv]

        for target in targets:
            key = mhn.patterns[target]
            for keep in keeps:
                mask = occlude(key, keep, seed=SEED + target)
                partial = key * mask
                if float(partial.norm()) < 1e-8:
                    continue  # nothing left of the cue; not a completion test
                cell = cells[keep]
                cell["n"] += 1
                cell["units"] += int((partial.abs() > 0).sum())
                unit_cue = partial / partial.norm()
                cell["cue_cos"] += float(unit_cue @ key)

                # Single-shot nearest neighbour on the SAME cue: no settling,
                # no clamping, no iterations. The thing completion must beat.
                cell["cosine"] += int(int((partial @ mhn.patterns.T).argmax()) == target)

                result = system.complete(partial, mask, top_k=1, beta=beta)
                top = result.results[0]
                cell["completed"] += int(store.record_at(target).id == top.record.id)
                cell["iters"] += result.trace.iterations
                cell["converged"] += int(result.trace.converged)
                cell["mixture"] += int(result.trace.is_mixture)
                # The clamp is the definition of this protocol: known
                # coordinates must come back untouched, or it is not
                # completion, it is retrieval with extra steps.
                cell["clamp_exact"] += int(
                    torch.allclose(result.trace.state[mask], partial[mask], atol=1e-5)
                )

    out = {}
    for keep, cell in cells.items():
        n = max(cell["n"], 1)
        out[keep] = {
            "n": cell["n"],
            "completed": cell["completed"] / n,
            "cosine": cell["cosine"] / n,
            "cue_cos": cell["cue_cos"] / n,
            "units": cell["units"] / n,
            "iters": cell["iters"] / n,
            "converged": cell["converged"] / n,
            "clamp_exact": cell["clamp_exact"] / n,
            "mixture": cell["mixture"] / n,
        }
    return out


def _print_arm(label: str, protocol: str, cells: dict, keeps: tuple[float, ...]) -> None:
    print(f"\n  {label}  [{protocol}]")
    print(f"    {'kept':>6} {'n':>5}  {'complete@1':>10} {'cosine@1':>9} "
          f"{'delta':>7}  {'units':>7} {'cue cos':>8} {'iters':>6} "
          f"{'conv':>5} {'clamp':>6} {'mixture':>8}")
    for keep in keeps:
        c = cells[keep]
        if not c["n"]:
            continue
        print(f"    {keep:6.2f} {c['n']:5d}  {c['completed']:10.3f} "
              f"{c['cosine']:9.3f} {c['completed'] - c['cosine']:+7.3f}  "
              f"{c['units']:7.1f} {c['cue_cos']:8.3f} {c['iters']:6.1f} "
              f"{c['converged']:5.2f} {c['clamp_exact']:6.2f} {c['mixture']:8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("locomo", "qmsum"), default="locomo")
    parser.add_argument("--locomo", type=int, default=10)
    parser.add_argument("--qmsum", type=int, default=25)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=128.0)
    parser.add_argument("--per-conv", type=int, default=40,
                        help="source memories sampled per conversation")
    parser.add_argument("--json", type=Path, default=None)
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)
    torch.manual_seed(SEED)

    if args.corpus == "locomo":
        convs = locomo.load()[: args.locomo]
    else:
        convs = qmsum.load(max_meetings=args.qmsum)
    turns = sum(len(c.turns) for c in convs)
    print(f"{args.corpus}: {len(convs)} conversations, {turns} turns, "
          f"LSA-{args.dim}, beta={args.beta:g}, {args.per_conv} cues/"
          f"conversation, {threads} threads")
    print("!! single-corpus run; the both-corpora overfitting guard is NOT in "
          "force for these numbers on their own")

    results: dict[str, dict] = {}
    for label, key_mode, whiten in ARMS:
        # Active-unit occlusion is only defined where the key is sparse.
        protocols = ("coords", "units") if key_mode is HippocampalKey.SEPARATED \
            else ("coords",)
        for protocol in protocols:
            t0 = time.time()
            cells = measure_arm(
                convs, dim=args.dim, key_mode=key_mode, whiten=whiten,
                beta=args.beta, keeps=KEEPS, protocol=protocol,
                per_conv=args.per_conv,
            )
            results[f"{label} [{protocol}]"] = cells
            _print_arm(label, protocol, cells, KEEPS)
            print(f"    ({time.time() - t0:.0f}s)")
            if args.json:
                args.json.write_text(json.dumps(
                    {"corpus": args.corpus, "beta": args.beta, "dim": args.dim,
                     "conversations": len(convs), "turns": turns,
                     "threads": threads,
                     "arms": {k: {str(kk): vv for kk, vv in v.items()}
                              for k, v in results.items()}},
                    indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
