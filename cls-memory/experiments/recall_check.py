"""Did the recall changes actually help, and do they hold on a second corpus?

    PYTHONPATH=. .venv/bin/python experiments/recall_check.py

Applies each default change cumulatively and measures on **two** unrelated
corpora — LoCoMo (two-person personal dialogue) and QMSum (many-speaker
organisational meetings). A change that helps one and hurts the other is
overfitting to a corpus, and the point of running both is to catch that.

Rows are cumulative, so each line isolates one decision. Two further axes --
`whiten` and `key_mode` -- are swept independently of that cumulative history
(ticket 07): they are ordinary keyword arguments to `evaluate`, not baked into
the label sequence, so any row can turn either on without disturbing the rows
above it. `evaluate` reports the separation *actually achieved* alongside
hit@k -- `anisotropy()` over the embeddings and hippocampal keys that row
really stored, not the configuration that was requested. A system that enables
whitening and never fits it would show up here as an unwhitened row, because
the number is measured, not assumed.
"""

from __future__ import annotations

import argparse
import time

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
from cls_memory.whitening import WhitenedEmbedder, anisotropy
from experiments import locomo, qmsum
from experiments.threads import add_threads_arg, pin_threads

SEED = 0
KS = (1, 5, 10)


def evaluate(
    conversations,
    *,
    dim: int,
    key_mode: HippocampalKey,
    beta: float,
    whiten: bool = False,
) -> dict:
    corpus = [t.memory_text for c in conversations for t in c.turns]
    raw_embedder = LatentSemanticEmbedder(dim=dim, seed=SEED).fit(corpus)
    if whiten:
        # Fitted ONCE on the full multi-conversation corpus, then shared
        # across every per-conversation system below -- exactly how the raw
        # LSA embedder is already shared. Fitting per-conversation instead
        # (a few hundred turns in `dim` dimensions) is rank-deficient and was
        # the exact mistake that inflated the whitening loss in RESULTS.md
        # V.5; see WhiteningConfig and cls_memory.whitening.
        embedder = WhitenedEmbedder(raw_embedder)
        embedder.fit(corpus)
    else:
        embedder = raw_embedder

    hits = {k: 0 for k in KS}
    # Per-question 0/1 outcomes, kept so arms can be compared with the exact
    # paired McNemar test in experiments/metrics.py rather than by eyeballing
    # two rates. At n=494 a difference below ~0.04 hit@1 is not resolvable, so
    # the aggregate alone cannot say whether an arm actually moved anything.
    per_question = {k: [] for k in KS}
    asked = 0
    embeddings_seen: list[Tensor] = []
    keys_seen: list[Tensor] = []
    for conv in conversations:
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
                hopfield=HopfieldConfig(beta=beta),
                key=KeyConfig(mode=key_mode, expansion_dim=2048, sparsity_k=256),
                seed=SEED,
            ),
            # `whitening.enabled` is deliberately left at its default (False):
            # this harness fits (and, when `whiten`, whitens) the embedder
            # itself, once, outside the per-conversation loop, and hands the
            # already-fitted object in -- exactly as it already does for the
            # plain LSA embedder. Setting `whitening.enabled=True` here too
            # would make `OrganizationalMemory.__init__` wrap it a *second*
            # time in a fresh, unfitted `WhitenedEmbedder`.
            embedder=embedder,
        )
        system.cortex.fit(
            embedder.encode([t.memory_text for t in conv.turns]), epochs=15
        )

        dia_of = {}
        for turn in conv.turns:
            embedding = embedder.encode([turn.memory_text])[0]
            if float(embedding.norm()) < 1e-8:
                continue  # degenerate text; the store now rejects these
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
            embeddings_seen.append(embedding.detach())
            keys_seen.append(key.detach())

        now = max(t.timestamp for t in conv.turns)
        present = set(dia_of.values())
        for question in conv.questions:
            evidence = {e for e in question.evidence if e in present}
            if not evidence:
                continue
            asked += 1
            result = system.recall(
                question.question, top_k=max(KS), reinforce=False, now=now
            )
            ranked = [dia_of[r.record.id] for r in result.results]
            for k in KS:
                hit = int(any(e in ranked[:k] for e in evidence))
                hits[k] += hit
                per_question[k].append(hit)

    out = {k: hits[k] / max(asked, 1) for k in KS}
    out["asked"] = asked
    out["per_question"] = per_question
    # The measured quantity, not the requested one: anisotropy over the
    # embeddings and hippocampal keys this row actually wrote to the store,
    # not over "whiten" or "key_mode" as booleans. 0.0 is isotropic (see
    # cls_memory.whitening.anisotropy for reference points).
    out["aniso_emb"] = anisotropy(torch.stack(embeddings_seen))
    out["aniso_key"] = anisotropy(torch.stack(keys_seen))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", type=int, default=3)
    parser.add_argument("--qmsum", type=int, default=25)
    # QMSum is ~10x LoCoMo's turn count (14431 vs 1451) and the LSA-1024 rows
    # cost hours there, so the two corpora are separable at the command line.
    # Running one at a time is a cost control, NOT a way to skip the
    # overfitting guard -- a change is only supported when both agree, per
    # this module's docstring.
    parser.add_argument("--corpus", choices=("both", "locomo", "qmsum"),
                        default="both")
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)
    torch.manual_seed(SEED)

    datasets = []
    if args.corpus in ("both", "locomo"):
        datasets.append(("LoCoMo", locomo.load()[: args.locomo]))
    if args.corpus in ("both", "qmsum"):
        datasets.append(("QMSum", qmsum.load(max_meetings=args.qmsum)))
    if args.corpus != "both":
        print(f"!! single-corpus run ({args.corpus}); the both-corpora "
              "overfitting guard is NOT in force for these numbers")
    print(f"torch threads: {threads}")
    for name, convs in datasets:
        print(f"{name}: {len(convs)} conversations, "
              f"{sum(len(c.turns) for c in convs)} turns")

    # (label, dim, key_mode, beta, whiten). The first four rows are the
    # original cumulative history and MUST reproduce the published numbers
    # unchanged (RESULTS.md Part III / HANDOFF.md §2) -- if a refactor moves
    # one of them, that is a bug in the refactor, not a new finding. The rows
    # below them exercise the two new axes (ticket 07) at the current best
    # operating point; they are new sweep points, not part of that history,
    # and are not interpreted here -- that is ticket 08's job.
    configs = [
        ("was:  LSA-256, separated, beta=8", 256, HippocampalKey.SEPARATED, 8.0, False),
        ("  +  beta=128", 256, HippocampalKey.SEPARATED, 128.0, False),
        ("  +  key=embedding", 256, HippocampalKey.EMBEDDING, 128.0, False),
        ("now:  +  LSA-1024", 1024, HippocampalKey.EMBEDDING, 128.0, False),
        ("  +  whitened", 1024, HippocampalKey.EMBEDDING, 128.0, True),
        ("  +  key=separated", 1024, HippocampalKey.SEPARATED, 128.0, False),
        ("  +  key=separated, whitened", 1024, HippocampalKey.SEPARATED, 128.0, True),
    ]

    for name, convs in datasets:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"  {'configuration':<32} " + "  ".join(f"@{k}".rjust(7) for k in KS)
              + "   aniso_emb  aniso_key")
        print("  " + "-" * 88)
        for label, dim, key_mode, beta, whiten in configs:
            t0 = time.time()
            r = evaluate(convs, dim=dim, key_mode=key_mode, beta=beta, whiten=whiten)
            print(f"  {label:<32} " + "  ".join(f"{r[k]:7.3f}" for k in KS)
                  + f"   {r['aniso_emb']:9.3f}  {r['aniso_key']:9.3f}"
                  + f"   ({r['asked']} q, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
