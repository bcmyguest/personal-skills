"""Separation x inverse-temperature: does the Hopfield layer ever leave cosine? (ticket 09)

    PYTHONPATH=. uv run python -u experiments/separation_beta_sweep.py --corpus locomo

Ticket 08 turned the separation mechanisms on at one operating point. This maps
the whole surface: every separation arm crossed with every inverse temperature,
with two things reported per cell --

  * **recall** (hit@1/5/10), against a *cosine kNN baseline measured on the same
    store*, so "competitive" is a comparison and not an adjective; and
  * **rho_rank**, the mean per-question Spearman correlation between the
    Hopfield ranking (softmax weights at the settled state) and plain cosine
    similarity of the cue to the stored patterns.

rho_rank = 1.0 means the attractor dynamics reordered nothing: the layer is
cosine kNN in a costume. The hypothesis under test is that raising separation
opens a moderate-beta window where rho_rank drops meaningfully below 1.0 *and*
recall stays competitive. RESULTS.md IV.1 already showed the *abstention*
signal degenerates into top-1 cosine at high beta; the open question is whether
the same collapse governs *retrieval ranking*, and whether separation moves it.

Two readings of "energy versus cosine" are reported, because the ticket's
phrase covers both and they answer different questions:

    rho_rank    per question, over stored patterns: is the RANKING different?
    rho_signal  over questions, neg_depth_nats vs cos_top1: is the CONFIDENCE
                signal different? (the RESULTS.md IV.1 quantity, continued
                here so the two tables can be read against each other)

Cost note: beta is a *read-time* parameter. The write path -- LSA fit, whitener
fit, cortex fit, key encoding, store writes -- does not depend on it, so each
separation arm is built ONCE and every beta is swept against that one store.
The grid is the full cross-product; only the redundant rebuilds are gone. On
QMSum, where a single build is 720-2500s, this is the difference between hours
and a day. `test_separation_beta_sweep.py` pins the shortcut to
`recall_check.evaluate`, which does rebuild per cell.

DG (`key=SEPARATED`) is carried as a **negative control**, not a second lever:
it is measured to *lower* separation (anisotropy 0.041 -> 0.134 on LoCoMo), so
if the capacity story is right, it should not beat the arm that raises it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from cls_memory import (
    CortexConfig,
    HippocampalKey,
    HopfieldConfig,
    KeyConfig,
    LatentSemanticEmbedder,
    MemoryRecord,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from cls_memory.energy import basin_depth
from cls_memory.whitening import WhitenedEmbedder, anisotropy
from experiments import locomo, qmsum
from experiments.metrics import mcnemar_exact, spearman_rho
from experiments.recall_check import KS, SEED
from experiments.threads import add_threads_arg, pin_threads

# The four separation arms of ticket 08, in the same order and with the same
# meanings, so a cell at beta=128 is directly comparable to that table.
ARMS: list[tuple[str, HippocampalKey, bool]] = [
    ("baseline", HippocampalKey.EMBEDDING, False),
    ("whitened", HippocampalKey.EMBEDDING, True),
    ("DG (neg control)", HippocampalKey.SEPARATED, False),
    ("both", HippocampalKey.SEPARATED, True),
]

# Spans the regime boundary in both directions: 2 and 8 are the metastable /
# gist end (RESULTS.md IV.1 measured rho=0.24 for the abstention signal at 8),
# 128 is the shipped default, 512 is deep in the collapsed regime where the
# mixture is dominated by its nearest component.
BETAS: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0, 512.0)


def build_embedder(corpus: list[str], *, dim: int, whiten: bool):
    """The write-side embedder for one arm, fitted exactly as `evaluate` does.

    Fitted ONCE on the full multi-conversation corpus and shared across the
    per-conversation systems -- fitting a whitener per conversation is
    rank-deficient and is the mistake that inflated RESULTS.md V.5.
    """
    raw = LatentSemanticEmbedder(dim=dim, seed=SEED).fit(corpus)
    if not whiten:
        return raw
    embedder = WhitenedEmbedder(raw)
    embedder.fit(corpus)
    return embedder


def build_system(embedder, conv, *, key_mode: HippocampalKey) -> OrganizationalMemory:
    """One conversation's memory, built the way `recall_check.evaluate` builds it.

    `hopfield.beta` is left at the config default and never read: every call
    below passes an explicit `beta`, which is the whole point of the sweep.
    """
    system = OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=embedder.dim,
                hidden_dims=(192, 96),
                latent_dim=32,
                epochs=15,
                batch_size=64,
                learning_rate=1e-3,
                kl_weight=0.01,
            ),
            novelty=NoveltyConfig(quantile=0.95, warmup=32, window=8192),
            hopfield=HopfieldConfig(beta=128.0),
            key=KeyConfig(mode=key_mode, expansion_dim=2048, sparsity_k=256),
            seed=SEED,
        ),
        embedder=embedder,
    )
    system.cortex.fit(embedder.encode([t.memory_text for t in conv.turns]), epochs=15)
    return system


def _blank_cell() -> dict:
    return {
        "hits": {k: 0 for k in KS},
        "per_question": {k: [] for k in KS},
        "rho_rank": [],
        "top1_agree": 0,
        "ties": 0,
        "overlap": 0.0,
        "mixture": 0,
        "depth_nats": [],
        "cos_top1": [],
    }


def sweep_arm(
    conversations,
    *,
    dim: int,
    key_mode: HippocampalKey,
    whiten: bool,
    betas: tuple[float, ...],
) -> dict:
    """Build one separation arm once, then measure every beta against it.

    Returns per-beta Hopfield cells plus the beta-independent cosine kNN
    reference measured on the same store -- the honest comparator for
    "is recall competitive", since it sees exactly the same patterns.
    """
    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = build_embedder(corpus, dim=dim, whiten=whiten)

    cells = {b: _blank_cell() for b in betas}
    shipped = {b: _blank_cell() for b in betas}
    cosine = _blank_cell()
    asked = 0
    embeddings_seen: list[Tensor] = []
    keys_seen: list[Tensor] = []

    for conv in conversations:
        system = build_system(embedder, conv, key_mode=key_mode)
        store = system.store
        mhn = store.mhn

        dia_of = {}
        for turn in conv.turns:
            embedding = embedder.encode([turn.memory_text])[0]
            if float(embedding.norm()) < 1e-8:
                continue  # degenerate text; the store rejects these
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
            store.add(record, now=turn.timestamp)
            dia_of[record.id] = turn.dia_id
            embeddings_seen.append(embedding.detach())
            keys_seen.append(key.detach())

        now = max(t.timestamp for t in conv.turns)
        present = set(dia_of.values())
        top_k = min(max(KS), len(store))
        for question in conv.questions:
            evidence = {e for e in question.evidence if e in present}
            if not evidence:
                continue
            asked += 1
            # Time-accurate salience, applied exactly where `recall` applies
            # it. Nothing below reinforces, so the priors are identical for
            # every beta and the arms differ only in temperature.
            store.refresh_priors(now)
            cue = system.retrieval.encode_cue(question.question)

            # Pure cosine kNN over the same stored patterns: no settling, no
            # salience prior. This is the thing the Hopfield layer has to beat.
            sims = cue @ mhn.patterns.T
            cos_idx = torch.topk(sims, top_k).indices
            cos_ranked = [dia_of[store.record_at(int(i)).id] for i in cos_idx]
            _score(cosine, cos_ranked, evidence)

            for beta in betas:
                trace = mhn.retrieve(cue, beta=beta)
                logits = mhn.logits(trace.state, beta)
                idx = torch.topk(logits, top_k).indices
                ranked = [dia_of[store.record_at(int(i)).id] for i in idx]
                cell = cells[beta]
                _score(cell, ranked, evidence)
                # The SHIPPED read path ranks by `trace.weights`, which is the
                # same ordering in exact arithmetic and is not the same
                # ordering in float32: at beta=512 a cosine gap of 0.2 puts the
                # softmax below 1e-38, so most of the store underflows to
                # exactly 0.0 and `topk` falls back to index order. Measured
                # here rather than assumed -- see `shipped` in the output.
                w_idx = torch.topk(trace.weights, top_k).indices
                _score(shipped[beta], [dia_of[store.record_at(int(i)).id]
                                       for i in w_idx], evidence)
                shipped[beta]["ties"] += int(w_idx.tolist() != idx.tolist())
                # Correlated on the LOGITS at the settled state, not the
                # softmax weights. The two induce the identical ranking, but
                # at beta=512 the weights underflow to exactly 0.0 for all but
                # a handful of patterns, and a vector that is 99% one tied
                # block correlates with nothing -- measured rho_rank 0.085 in
                # a cell whose top-1 agreed with cosine on 98.5% of questions.
                # That number was float underflow, not reordering.
                #
                # Over ALL stored patterns, not just the top-k: a reordering
                # that never reaches the top 10 is still a reordering, and
                # restricting to the top-k would bake in the very agreement
                # the correlation is supposed to test.
                cell["rho_rank"].append(
                    spearman_rho(mhn.logits(trace.state, beta), sims)
                )
                cell["top1_agree"] += int(int(idx[0]) == int(cos_idx[0]))
                # What a user would actually notice: how much of the returned
                # page is the same page cosine would have returned. Immune to
                # the tail, where the ranking is real but nobody reads it.
                cell["overlap"] += len(
                    set(idx.tolist()) & set(cos_idx.tolist())
                ) / top_k
                cell["mixture"] += int(trace.is_mixture)
                unit = cue / cue.norm().clamp_min(1e-12)
                basin = basin_depth(mhn, unit, beta)
                cell["depth_nats"].append(-basin.depth_nats)
                cell["cos_top1"].append(basin.top_similarity)

    out = {
        "asked": asked,
        "aniso_emb": anisotropy(torch.stack(embeddings_seen)),
        "aniso_key": anisotropy(torch.stack(keys_seen)),
        "cosine": _finish(cosine, asked),
        "betas": {b: _finish(cells[b], asked) for b in betas},
        "shipped": {b: _finish(shipped[b], asked) for b in betas},
    }
    for b in betas:
        cell = out["betas"][b]
        for k in KS:
            m = mcnemar_exact(out["cosine"]["per_question"][k], cell["per_question"][k])
            cell[f"mcnemar@{k}"] = m
    return out


def _score(cell: dict, ranked: list[str], evidence: set[str]) -> None:
    for k in KS:
        hit = int(any(e in ranked[:k] for e in evidence))
        cell["hits"][k] += hit
        cell["per_question"][k].append(hit)


def _finish(cell: dict, asked: int) -> dict:
    out = dict(cell)
    for k in KS:
        out[k] = cell["hits"][k] / max(asked, 1)
    rhos = [r for r in cell["rho_rank"] if r == r]  # drop NaN (constant vectors)
    out["rho_rank"] = sum(rhos) / len(rhos) if rhos else float("nan")
    out["rho_rank_n"] = len(rhos)
    out["top1_agree"] = cell["top1_agree"] / max(asked, 1)
    out["overlap"] = cell["overlap"] / max(asked, 1)
    out["ties"] = cell["ties"] / max(asked, 1)
    out["mixture"] = cell["mixture"] / max(asked, 1)
    if cell["depth_nats"]:
        out["rho_signal"] = spearman_rho(
            torch.tensor(cell["depth_nats"]), torch.tensor(cell["cos_top1"])
        )
    else:
        out["rho_signal"] = float("nan")
    del out["depth_nats"], out["cos_top1"], out["hits"]
    return out


def _print_arm(label: str, arm: dict, betas: tuple[float, ...]) -> None:
    n = arm["asked"]
    print(f"\n  {label}   (n={n} questions, aniso_emb {arm['aniso_emb']:+.3f}, "
          f"aniso_key {arm['aniso_key']:+.3f})")
    c = arm["cosine"]
    print(f"    {'beta':>6}  {'@1':>6} {'@5':>6} {'@10':>6}  {'rho_rank':>9} "
          f"{'top1=cos':>9} {'ovlp@10':>8} {'mixture':>8}  {'rho_signal':>10}  "
          f"{'d@1 vs cos':>11} {'p':>8}")
    print(f"    {'cosine':>6}  {c[1]:6.3f} {c[5]:6.3f} {c[10]:6.3f}  "
          f"{'--':>9} {'--':>9} {'--':>8} {'--':>8}  {'--':>10}  {'--':>11} "
          f"{'--':>8}")
    for b in betas:
        cell = arm["betas"][b]
        m = cell["mcnemar@1"]
        print(f"    {b:6.0f}  {cell[1]:6.3f} {cell[5]:6.3f} {cell[10]:6.3f}  "
              f"{cell['rho_rank']:9.3f} {cell['top1_agree']:9.3f} "
              f"{cell['overlap']:8.3f} {cell['mixture']:8.3f}  "
              f"{cell['rho_signal']:10.3f}  "
              f"{cell[1] - c[1]:+11.3f} {m['p']:8.4f}")

    # The read path as shipped, ranked by the underflowing softmax weights.
    # Identical to the table above wherever `ties` is 0.
    if any(arm["shipped"][b]["ties"] for b in betas):
        print("    shipped read path (ranks by trace.weights; ties broken by "
              "index order once the softmax underflows):")
        print(f"      {'beta':>6}  {'@1':>6} {'@5':>6} {'@10':>6}  "
              f"{'tied questions':>15}")
        for b in betas:
            sh = arm["shipped"][b]
            print(f"      {b:6.0f}  {sh[1]:6.3f} {sh[5]:6.3f} {sh[10]:6.3f}  "
                  f"{sh['ties']:15.3f}")


def _cell_json(cell: dict) -> dict:
    """One cell, JSON-safe: the per-question indicators dropped and every key a
    string. The hit@k entries are keyed by `int` and everything else by `str`,
    which `json.dumps(sort_keys=True)` cannot order -- it raised TypeError
    after the first arm and took a 532s LoCoMo run's results with it."""
    return {str(k): v for k, v in cell.items() if k != "per_question"}


def _jsonable(arm: dict) -> dict:
    """Drop the per-question indicator lists; keep everything summarised."""
    out = {k: v for k, v in arm.items()
           if k not in ("cosine", "betas", "shipped")}
    out["cosine"] = _cell_json(arm["cosine"])
    for section in ("betas", "shipped"):
        out[section] = {
            str(b): _cell_json(cell) for b, cell in arm[section].items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("locomo", "qmsum"), default="locomo")
    parser.add_argument("--locomo", type=int, default=10)
    parser.add_argument("--qmsum", type=int, default=25)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    parser.add_argument("--arms", type=str, nargs="+", default=None,
                        help="subset of arm labels to run (default: all four)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write results here after EVERY arm, so a long "
                             "run is recoverable if it is killed")
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)
    torch.manual_seed(SEED)
    betas = tuple(args.betas)

    if args.corpus == "locomo":
        convs = locomo.load()[: args.locomo]
    else:
        convs = qmsum.load(max_meetings=args.qmsum)
    turns = sum(len(c.turns) for c in convs)
    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]

    print(f"{args.corpus}: {len(convs)} conversations, {turns} turns, "
          f"LSA-{args.dim}, {threads} threads")
    print(f"grid: {len(arms)} separation arms x {len(betas)} betas = "
          f"{len(arms) * len(betas)} cells")
    print("!! single-corpus run; the both-corpora overfitting guard is NOT in "
          "force for these numbers on their own")

    results = {}
    for label, key_mode, whiten in arms:
        t0 = time.time()
        arm = sweep_arm(convs, dim=args.dim, key_mode=key_mode, whiten=whiten,
                        betas=betas)
        results[label] = arm
        _print_arm(label, arm, betas)
        print(f"    ({time.time() - t0:.0f}s)")
        if args.json:
            args.json.write_text(json.dumps(
                {"corpus": args.corpus, "conversations": len(convs),
                 "turns": turns, "dim": args.dim, "betas": list(betas),
                 "threads": threads,
                 "arms": {k: _jsonable(v) for k, v in results.items()}},
                indent=2, sort_keys=True))

    print("\n  rho_rank across the grid (1.000 = the Hopfield ranking IS "
          "cosine kNN):")
    print(f"    {'arm':<18} " + "  ".join(f"b={b:g}".rjust(9) for b in betas))
    for label in results:
        row = "  ".join(
            f"{results[label]['betas'][b]['rho_rank']:9.3f}" for b in betas
        )
        print(f"    {label:<18} {row}")


if __name__ == "__main__":
    main()
