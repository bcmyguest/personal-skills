"""Schema absorption (ticket 11): does gist recall actually buy anything?

    PYTHONPATH=. uv run python -u experiments/gist_check.py --corpus locomo

This is the first measurement of the project's third headline claim: that
`OrganizationalMemory.gist()` -- deliberately low-beta recall, see
`cls_memory.retrieval.PatternCompleter.gist` -- settles into a metastable
mixture that summarises a *family* of related memories, and that this is
useful for something a single-episode recall or a plain average cannot do.
It was dismissed once already on an adversarial near-verbatim-paraphrase
question set, "a task that could not have shown a gist effect even if one
existed" (ticket 11). This script is an honest attempt at a task the effect
*could* show up on, plus the negative-control baseline that tells us whether
any advantage is really the attractor dynamics or just averaging.

THE TASK, STATED PLAINLY
-------------------------
LoCoMo and QMSum both ship a `Question.evidence` field: the set of dialogue
turns (dia_ids) that a correct answer to that question actually needs. Most
questions resolve to one turn. A minority resolve to several *scattered*
turns -- e.g. "how many pets does X have" needs every turn that mentions a
pet, not the nearest one. This script keeps only the questions whose evidence
set has at least `--min-evidence` members (default 2) and scores each arm on
how much of that SET it recovers in its top-k, not whether it found the
single best match.

Why a correct answer cannot come from identifying one memory: by
construction the gold set has >= `--min-evidence` DISTINCT turns carrying
distinct facts. A read that returns one attractor and reads out its
neighbourhood can still list several turns, but there is nothing in a
single-episode retrieval that causes those extra list entries to be the
OTHER members of the fact family rather than merely nearby text -- whereas a
mixture state that has genuinely blended several family members is, by the
theory this project is testing, supposed to do exactly that.

HOW HONEST THIS IS, AND WHAT IT IS NOT
---------------------------------------
This is a DERIVED task, not a synthetic one: the evidence sets come from the
corpora's own human annotation (LoCoMo / QMSum), not from a generated
template family, which is the failure mode RESULTS.md already retracted one
result for. But it is also not a clean test of classical CLS "schema
abstraction" (averaging over recurring regularities to produce a summary
statement, e.g. Bartlett's restaurant script). It is closer to multi-hop
retrieval: the library has no generation step, so "a correct answer" is
operationalised as "the evidence turns are present in the returned set",
never as an actual synthesised sentence. A system could pass this task by
having a wide, low-precision retrieval radius with no metastable dynamics at
all -- which is exactly why Arm 3 (centroid-of-top-k) exists: if a plain
average of the k nearest neighbours covers the evidence set just as well as
the settled gist state, the credit belongs to breadth, not to the attractor
dynamics, and the mixture diagnostics below are what tells the two apart.

`experiments/rulebook.py` was considered as the corpus instead: two of its
sixteen situations have genuinely multi-source gold sets (e.g. "a
contractor's engagement is ending" needs access, HR and equipment rules
together). That is a hand-built, honestly-labelled version of the same idea,
but at n=2 it carries no statistical weight on its own, and rules live in one
flat store rather than per-conversation, which does not match this script's
per-conversation harness. It is not used here for that reason, not because
it is a worse task.

THREE ARMS, IDENTICAL QUERIES
------------------------------
  gist      `system.gist(...)`     -- low-beta settle, swept over `--betas`
                                       plus the library's own default factor
                                       (`--factor`), so nothing is hardcoded
                                       to one temperature while the metastable
                                       band from ticket 09's sweep is still
                                       being located.
  recall    `system.recall(...)`   -- ordinary settle at the shipped default
                                       beta (single fixed reference row).
  centroid  no library call at all -- plain unweighted average of the
                                       cosine-top-k stored KEYS, re-ranked by
                                       cosine against the store. No `step`,
                                       no `retrieve`, no iteration: this is
                                       the dynamics-free control the ticket
                                       requires so any gist advantage cannot
                                       be attributed to averaging as such.

DIAGNOSTICS
-----------
Per row this prints the task metric (evidence coverage@top-k) plus three
numbers that distinguish "settled onto a real mixture" from "settled onto
one episode" from "collapsed to the global centroid":

  top1         top attention weight at the terminal state (low = spread out)
  effective_n  exp(entropy) of the attention distribution -- the "effective
               number of memories" the state is a mixture of (1.0 = a single
               point mass, N = uniform over everything)
  mixture      fraction of questions where top1 < mixture_threshold, i.e.
               `RetrievalTrace.is_mixture`

For the centroid arm, which never settles, these are still computed -- by
scoring the FIXED averaged vector through the network's own attention formula
(`mhn.attention(centroid, beta)`) at the same beta each gist row used
(`RecallResult.beta`, the resolved numeric value even when `--factor` chose
it). That is a read of the landscape at a fixed point, not an iteration, so
it stays dynamics-free while remaining comparable to the gist row it sits
next to: a plain average can "look like a mixture" under this formula without
ever having been produced by settling, and that is precisely the case this
comparison is built to catch.

If the numbers come out matching the `hippocampus.step` docstring's
measured result -- low beta pulls toward the global centroid, gist's
coverage tracks the centroid arm's rather than beating it, and there is no
beta where `is_mixture` is high AND coverage beats both baselines -- that is
the expected honest negative result, not a bug in this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from cls_memory import HippocampalKey, MemoryRecord, OrganizationalMemory, Persistence
from experiments import locomo, qmsum
from experiments.recall_check import SEED
from experiments.threads import add_threads_arg, pin_threads
from experiments.separation_beta_sweep import build_embedder, build_system

# Spans very-low through moderate beta: 2 and 8 are deep in the region
# RESULTS.md's step() docstring reports as "pulled to the global centroid",
# 32 sits closer to the shipped default (128) without being it. Ticket 09's
# sweep is what actually locates the metastable band; these are starting
# points, not a claim about where it is -- override with --betas.
DEFAULT_BETAS: tuple[float, ...] = (2.0, 8.0, 32.0)
DEFAULT_FACTOR = 0.15  # PatternCompleter.gist's own shipped default.


# --------------------------------------------------------------------- ingest


def _ingest_conversation(system: OrganizationalMemory, conv) -> dict[str, str]:
    """Write every turn into the store; return record.id -> dia_id.

    Mirrors `separation_beta_sweep.sweep_arm`'s ingestion loop exactly, since
    that is the reference shape for "how this repo builds a task corpus" per
    a real conversation loader.
    """
    dia_of: dict[str, str] = {}
    for turn in conv.turns:
        embedding = system.embedder.encode([turn.memory_text])[0]
        if float(embedding.norm()) < 1e-8:
            continue  # degenerate text; the store would reject it anyway
        latent = system.cortex.latent(embedding)
        key = system.key_encoder(embedding, latent)
        record = MemoryRecord(
            text=turn.memory_text,
            embedding=embedding,
            latent=latent,
            key=key,
            persistence=Persistence.TEMPORAL,
            created_at=turn.timestamp,
            last_reinforced_at=turn.timestamp,
        )
        system.store.add(record, now=turn.timestamp)
        dia_of[record.id] = turn.dia_id
    return dia_of


# ------------------------------------------------------------------ scoring


def _mixture_stats(weights: Tensor, threshold: float) -> dict:
    """top1 / effective_n / is_mixture from a softmax weight vector.

    effective_n = exp(entropy): 1.0 for a point mass, N for a uniform mixture
    over N patterns -- the standard participation-ratio-style read of "how
    many things is this actually a blend of".
    """
    if weights.numel() == 0:
        return {"top1": 0.0, "effective_n": 0.0, "is_mixture": False}
    top1 = float(weights.max())
    p = weights.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    return {
        "top1": top1,
        "effective_n": math.exp(entropy),
        "is_mixture": top1 < threshold,
    }


def _coverage(ranked_ids: list[str], evidence: set[str]) -> float:
    if not evidence:
        return 0.0
    return len(set(ranked_ids) & evidence) / len(evidence)


def _centroid_of_top_k(mhn, cue: Tensor, k: int) -> Tensor:
    """Arm 3: the plain unit-normalised mean of the k nearest stored keys.

    Deliberately NOT `mhn.step`/`mhn.retrieve`: no softmax, no beta, no
    iteration, nothing that touches `mhn.config` at all. This is the
    dynamics-free baseline the ticket requires -- it exists so that any
    advantage `gist()` shows can be attributed to the settling process
    itself, not to "averaging a neighbourhood" as such, which this function
    does and gist() also does as a side effect of settling.
    """
    sims = cue @ mhn.patterns.T
    idx = torch.topk(sims, min(k, len(mhn))).indices
    raw = mhn.patterns[idx].mean(dim=0)
    return raw / raw.norm().clamp_min(1e-12)


def _blank_cell() -> dict:
    # task_n and diag_n are separate counters, not one shared "n": the
    # centroid arm's diagnostics are recorded once per (question, beta) pair
    # without ever calling `_record_task` for that beta (its task metric is
    # beta-invariant and recorded exactly once, outside the beta loop), so a
    # single shared counter would silently under-report how many diagnostic
    # observations actually went into the mean.
    return {
        "coverage": [], "full": 0, "task_n": 0,
        "top1": [], "effective_n": [], "mixture": 0, "diag_n": 0,
    }


def _record_task(cell: dict, ranked_ids: list[str], evidence: set[str]) -> None:
    cell["coverage"].append(_coverage(ranked_ids, evidence))
    cell["full"] += int(evidence <= set(ranked_ids))
    cell["task_n"] += 1


def _record_diag(cell: dict, weights: Tensor, threshold: float) -> None:
    stats = _mixture_stats(weights, threshold)
    cell["top1"].append(stats["top1"])
    cell["effective_n"].append(stats["effective_n"])
    cell["mixture"] += int(stats["is_mixture"])
    cell["diag_n"] += 1


def _finish(cell: dict) -> dict:
    task_n = max(cell["task_n"], 1)
    diag_n = max(cell["diag_n"], 1)
    return {
        "n": cell["task_n"],
        "diag_n": cell["diag_n"],
        "coverage": sum(cell["coverage"]) / task_n if cell["coverage"] else float("nan"),
        "full_recall": cell["full"] / task_n if cell["coverage"] else float("nan"),
        "top1": sum(cell["top1"]) / diag_n if cell["top1"] else float("nan"),
        "effective_n": sum(cell["effective_n"]) / diag_n if cell["effective_n"] else float("nan"),
        "mixture_rate": cell["mixture"] / diag_n if cell["top1"] else float("nan"),
    }


# --------------------------------------------------------------------- run


def run(
    conversations,
    *,
    dim: int,
    betas: tuple[float, ...],
    factor: float,
    top_k: int,
    min_evidence: int,
) -> dict:
    """Evaluate the three arms on identical multi-evidence queries.

    One embedder fitted once across the whole corpus (as every other harness
    in this repo does), one system per conversation, beta is a read-time
    parameter so every gist row is measured against the same store.
    """
    corpus_text = [t.memory_text for c in conversations for t in c.turns]
    embedder = build_embedder(corpus_text, dim=dim, whiten=False)

    labels = [str(b) for b in betas] + [f"factor={factor:g}"]
    recall_cell = _blank_cell()
    centroid_task_cell = _blank_cell()
    gist_cells = {label: _blank_cell() for label in labels}
    centroid_diag_cells = {label: _blank_cell() for label in labels}

    asked = 0
    for conv in conversations:
        system = build_system(embedder, conv, key_mode=HippocampalKey.EMBEDDING)
        dia_of = _ingest_conversation(system, conv)
        if not dia_of:
            continue
        store = system.store
        mhn = store.mhn
        threshold = mhn.config.mixture_threshold
        present = set(dia_of.values())
        now = max(t.timestamp for t in conv.turns)
        seed_k = min(top_k, len(store))

        for question in conv.questions:
            evidence = {e for e in question.evidence if e in present}
            if len(evidence) < min_evidence:
                continue
            asked += 1
            # Priors refreshed once per question, identically for every arm:
            # nothing below reinforces, so the three arms and every beta row
            # see the exact same store state for this query.
            store.refresh_priors(now)

            # ---- Arm 1: ordinary single-episode recall, shipped default beta.
            # reinforce=False everywhere in this script (gist's own default):
            # reinforcement mutates priors as a side effect of the READ, which
            # would make later questions' results depend on the order the
            # arms happened to run in. The point is identical queries.
            rec = system.recall(
                question.question, top_k=top_k, beta=None, reinforce=False, now=now
            )
            rec_ids = [dia_of[r.record.id] for r in rec.results]
            _record_task(recall_cell, rec_ids, evidence)
            _record_diag(recall_cell, rec.trace.weights, threshold)

            # ---- Arm 3: centroid-of-top-k, no attractor dynamics at all.
            cue = system.retrieval.encode_cue(question.question)
            centroid = _centroid_of_top_k(mhn, cue, seed_k)
            final_sims = centroid @ mhn.patterns.T
            final_idx = torch.topk(final_sims, seed_k).indices
            centroid_ids = [dia_of[store.record_at(int(i)).id] for i in final_idx]
            _record_task(centroid_task_cell, centroid_ids, evidence)

            # ---- Arm 2: gist, once per swept beta plus the factor default.
            for label, beta_kwargs in zip(
                labels,
                [{"beta": b} for b in betas] + [{"factor": factor, "beta": None}],
            ):
                g = system.gist(
                    question.question, top_k=top_k, reinforce=False, now=now,
                    **beta_kwargs,
                )
                g_ids = [dia_of[r.record.id] for r in g.results]
                _record_task(gist_cells[label], g_ids, evidence)
                _record_diag(gist_cells[label], g.trace.weights, threshold)

                # Centroid arm's diagnostics, scored at the SAME resolved
                # beta this gist row used (`g.beta` -- the numeric value even
                # when `factor` picked it). A pure score, not an iteration:
                # `mhn.attention` never moves `centroid`, so this stays
                # dynamics-free while remaining comparable to the row it sits
                # next to in the printed table.
                cweights = mhn.attention(centroid, beta=g.beta)
                _record_diag(centroid_diag_cells[label], cweights, threshold)

    if asked == 0:
        raise RuntimeError(
            f"no questions had >= {min_evidence} resolvable evidence turns; "
            "lower --min-evidence or load more conversations"
        )

    return {
        "asked": asked,
        "labels": labels,
        "recall": _finish(recall_cell),
        "centroid_task": _finish(centroid_task_cell),
        "gist": {label: _finish(gist_cells[label]) for label in labels},
        "centroid_diag": {label: _finish(centroid_diag_cells[label]) for label in labels},
    }


# ------------------------------------------------------------------- report


def _print_report(result: dict) -> None:
    n = result["asked"]
    print(f"\n{n} multi-evidence questions")
    r = result["recall"]
    print(
        f"  {'arm':<16} {'coverage':>9} {'full':>6}  {'top1':>7} "
        f"{'eff_n':>7} {'mixture':>8}"
    )
    print(
        f"  {'recall (default)':<16} {r['coverage']:9.3f} {r['full_recall']:6.3f}  "
        f"{r['top1']:7.3f} {r['effective_n']:7.3f} {r['mixture_rate']:8.3f}"
    )
    c = result["centroid_task"]
    print(
        f"  {'centroid':<16} {c['coverage']:9.3f} {c['full_recall']:6.3f}  "
        f"{'--':>7} {'--':>7} {'--':>8}  (task metric is beta-invariant; see "
        f"below for its diagnostics at each swept beta)"
    )
    print("\n  gist, per beta (task metric + its own settled-state diagnostics):")
    for label in result["labels"]:
        g = result["gist"][label]
        print(
            f"    {label:<14} {g['coverage']:9.3f} {g['full_recall']:6.3f}  "
            f"{g['top1']:7.3f} {g['effective_n']:7.3f} {g['mixture_rate']:8.3f}"
        )
    print(
        "\n  centroid diagnostics AT THE SAME beta (task metric unchanged -- "
        "only the lens moves):"
    )
    for label in result["labels"]:
        d = result["centroid_diag"][label]
        print(
            f"    {label:<14} {'--':>9} {'--':>6}  "
            f"{d['top1']:7.3f} {d['effective_n']:7.3f} {d['mixture_rate']:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("locomo", "qmsum"), default="locomo")
    parser.add_argument("--locomo", type=int, default=10)
    parser.add_argument("--qmsum", type=int, default=25)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument(
        "--betas", type=float, nargs="+", default=list(DEFAULT_BETAS),
        help="explicit low-beta values to try for the gist arm",
    )
    parser.add_argument(
        "--factor", type=float, default=DEFAULT_FACTOR,
        help="also run gist with beta left to the library's own "
             "default_beta*factor scaling",
    )
    parser.add_argument("--json", type=Path, default=None)
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)
    torch.manual_seed(SEED)

    if args.corpus == "locomo":
        convs = locomo.load()[: args.locomo]
    else:
        convs = qmsum.load(max_meetings=args.qmsum)

    print(
        f"{args.corpus}: {len(convs)} conversations, LSA-{args.dim}, "
        f"top_k={args.top_k}, min_evidence={args.min_evidence}, "
        f"{threads} threads"
    )
    result = run(
        convs,
        dim=args.dim,
        betas=tuple(args.betas),
        factor=args.factor,
        top_k=args.top_k,
        min_evidence=args.min_evidence,
    )
    _print_report(result)
    if args.json:
        args.json.write_text(json.dumps({**result, "threads": threads},
                                        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
