"""What should actually ship? Both corpora, corrected weighting, seed variance.

    PYTHONPATH=. .venv/bin/python experiments/final_recall.py

Written after an adversarial review invalidated three earlier conclusions. It
exists to avoid repeating their mistakes, so it differs from the earlier
ablations in four ways that matter:

  * **Seed variance on the hashed rows.** `HashedProjection` is a randomised
    projection; a single seed is a sample, not a measurement. The earlier
    headline ("RP passes the sparse ceiling") was seed 0 being the maximum of
    five draws.
  * **The hybrid is parameterised by effective weight `w`,** not by the scaling
    factor. The cosine is quadratic in that factor, so the earlier grid probed
    w = (0.50, 0.155, 0.059) while claiming to probe (0.5, 0.3, 0.2).
  * **Both corpora, always.** The previously proposed default (w=0.5) loses to
    pure lexical at hit@1 on QMSum while winning on LoCoMo.
  * **hit@k, named honestly** — 1 if any evidence turn is in the top k.

Differences below ~0.04 at hit@1 are not resolvable at LoCoMo's n=494, and
QMSum's n is far smaller still. Treat small gaps as ties.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from cls_memory import LatentSemanticEmbedder
from experiments import locomo, qmsum
from experiments.recall_ablation import (
    BGEEmbedder,
    HashedProjection,
    HybridEmbedder,
    dense_knn_ranker,
    score_ranking,
)

SEED = 0


def mean_over_seeds(conversations, build, label: str, seeds=(0, 1, 2)) -> None:
    """Report mean +- sd for a randomised embedder rather than one draw."""
    runs = []
    t0 = time.time()
    for seed in seeds:
        embedder = build(seed)
        hits = {k: 0 for k in (1, 5, 10)}
        asked = 0
        for conv in conversations:
            rank = dense_knn_ranker(conv, embedder)
            ids = {t.dia_id for t in conv.turns}
            for question in conv.questions:
                evidence = {e for e in question.evidence if e in ids}
                if not evidence:
                    continue
                asked += 1
                ranked = rank(question.question)
                for k in hits:
                    hits[k] += int(any(r in evidence for r in ranked[:k]))
        runs.append({k: v / max(asked, 1) for k, v in hits.items()})

    cells = []
    for k in (1, 5, 10):
        vals = [r[k] for r in runs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"@{k} {statistics.mean(vals):.3f}+-{sd:.3f}")
    print(f"  {label:<46} " + "  ".join(cells) + f"   ({time.time() - t0:.0f}s)")


def evaluate(name: str, conversations) -> None:
    corpus = [t.memory_text for c in conversations for t in c.turns]
    print(f"\n{'=' * 78}\n{name}: {len(conversations)} conversations, "
          f"{len(corpus)} turns\n{'=' * 78}")

    # The free win: same class, same dim, vocabulary cap removed.
    capped = LatentSemanticEmbedder(
        dim=1024, seed=SEED, max_features=20_000, min_df=2
    ).fit(corpus)
    score_ranking(conversations, lambda c: dense_knn_ranker(c, capped),
                  "LSA-1024, old defaults (min_df=2, 20k cap)")
    uncapped = LatentSemanticEmbedder(dim=1024, seed=SEED).fit(corpus)
    score_ranking(conversations, lambda c: dense_knn_ranker(c, uncapped),
                  "LSA-1024, NEW defaults (min_df=1, 100k cap)")

    mean_over_seeds(
        conversations,
        lambda s: HashedProjection(corpus, dim=4096, weighting="bm25", seed=s),
        "RP-4096 bm25 (3 seeds)",
    )

    try:
        bge = BGEEmbedder()
    except Exception as exc:
        print(f"  [BGE unavailable: {type(exc).__name__}: {exc}]")
        return
    score_ranking(conversations, lambda c: dense_knn_ranker(c, bge),
                  f"BGE-small-v1.5 ({bge.dim}d) alone")

    lexical = HashedProjection(corpus, dim=4096, weighting="bm25", seed=SEED)
    for w in (0.75, 0.6, 0.5):
        score_ranking(
            conversations,
            lambda c, ww=w: dense_knn_ranker(c, HybridEmbedder(lexical, bge, w=ww)),
            f"HYBRID RP-4096 + BGE (w={w:g} lexical)",
        )
    # The cheap pairing: no 4096-d lexical half, no extra fit.
    for w in (0.6, 0.5):
        score_ranking(
            conversations,
            lambda c, ww=w: dense_knn_ranker(c, HybridEmbedder(uncapped, bge, w=ww)),
            f"HYBRID LSA-1024 + BGE (w={w:g} lexical)",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", type=int, default=3)
    parser.add_argument("--qmsum", type=int, default=25)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    evaluate("LoCoMo", locomo.load()[: args.locomo])
    evaluate("QMSum", qmsum.load(max_meetings=args.qmsum))


if __name__ == "__main__":
    main()
