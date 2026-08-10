"""Re-runs the RP-4096-vs-exact-sparse retraction under one IDF convention.

    PYTHONPATH=. .venv/bin/python experiments/idf_retraction_check.py

Ticket 05 (`.scratch/cls-memory-review/issues/05-...md`): `TfidfIndex` and
`BM25Index` in the ceilings section of `recall_ablation.py` were fit
per-conversation while `HashedProjection` (the random projection) was fit on
the whole corpus -- two different IDF conventions compared against each other.
That mismatch is what HANDOFF.md §3.1d item 1's retraction rests on: "RP-4096
passes the sparse TF-IDF ceiling" was retracted because, across five seeds,
RP-4096-bm25 measured 0.319 +- 0.010 against a 0.320 exact-sparse figure it was
said to pass.

With the shared-`idf_corpus` hook wired through both `TfidfIndex` and
`BM25Index` (this ticket), this script re-runs exactly that comparison under
one convention, over several projection seeds, and reports the exact paired
McNemar test against the exact-sparse ceiling for every seed -- not just the
one that happened to look best.
"""

from __future__ import annotations

import statistics
import time

import torch

from experiments import locomo
from experiments.metrics import mcnemar_exact
from experiments.recall_ablation import (
    BM25Index,
    HashedProjection,
    TfidfIndex,
    dense_knn_ranker,
)

SEED = 0
SEEDS = (0, 1, 2, 3, 4)
KS = (1, 5, 10)


def indicators_for(conversations, rank_fn) -> dict[int, list[int]]:
    """Per-question hit indicators at every k, needed for paired McNemar."""
    out = {k: [] for k in KS}
    for conv in conversations:
        rank = rank_fn(conv)
        ids = {t.dia_id for t in conv.turns}
        for question in conv.questions:
            evidence = {e for e in question.evidence if e in ids}
            if not evidence:
                continue
            ranked = rank(question.question)
            for k in KS:
                out[k].append(int(any(r in evidence for r in ranked[:k])))
    return out


def summarize(label: str, indicators: dict[int, list[int]]) -> dict[int, float]:
    scores = {k: sum(v) / max(len(v), 1) for k, v in indicators.items()}
    print(f"  {label:<40} " + "  ".join(f"@{k} {scores[k]:.3f}" for k in KS))
    return scores


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    conversations = locomo.load()[:3]
    corpus = [t.memory_text for c in conversations for t in c.turns]
    idf_corpus = corpus
    n_asked = sum(
        1
        for c in conversations
        for q in c.questions
        if {e for e in q.evidence if e in {t.dia_id for t in c.turns}}
    )
    print(f"{len(conversations)} conversations, {len(corpus)} turns, "
          f"n={n_asked} questions")
    print(f"IDF corpus: shared, {len(idf_corpus)} docs "
          f"(whole {len(conversations)}-conversation corpus)\n")

    print("EXACT-SPARSE CEILING (shared IDF convention)")

    def tfidf_ceiling(c):
        idx = TfidfIndex([t.memory_text for t in c.turns], idf_corpus=idf_corpus)
        assert idx.n_idf_docs == len(idf_corpus)
        return lambda q: [c.turns[i].dia_id for i in idx.rank(q)]

    def bm25_ceiling(c):
        idx = BM25Index([t.memory_text for t in c.turns], idf_corpus=idf_corpus)
        assert idx.n_idf_docs == len(idf_corpus)
        return lambda q: [c.turns[i].dia_id for i in idx.rank(q)]

    tfidf_ind = indicators_for(conversations, tfidf_ceiling)
    summarize("TF-IDF cosine (exact-sparse)", tfidf_ind)
    bm25_ind = indicators_for(conversations, bm25_ceiling)
    summarize("BM25 (exact-sparse)", bm25_ind)

    print("\nRANDOM-PROJECTION RP-4096, BM25-weighted, several seeds "
          "(shared IDF convention)")
    rp_at1 = []
    rp_indicators_by_seed = {}
    for seed in SEEDS:
        rp = HashedProjection(idf_corpus, dim=4096, weighting="bm25", seed=seed)
        assert rp.n_idf_docs == len(idf_corpus)
        ind = indicators_for(conversations, lambda c, e=rp: dense_knn_ranker(c, e))
        scores = summarize(f"RP-4096-bm25 (seed {seed})", ind)
        rp_at1.append(scores[1])
        rp_indicators_by_seed[seed] = ind

    mean1 = statistics.mean(rp_at1)
    sd1 = statistics.stdev(rp_at1) if len(rp_at1) > 1 else 0.0
    print(f"\n  RP-4096-bm25 hit@1 across {len(SEEDS)} seeds: "
          f"mean={mean1:.3f} sd={sd1:.3f} (values: "
          f"{', '.join(f'{v:.3f}' for v in rp_at1)})")
    print(f"  exact-sparse TF-IDF hit@1: {sum(tfidf_ind[1]) / len(tfidf_ind[1]):.3f}")

    print("\nEXACT PAIRED MCNEMAR: RP-4096-bm25 (each seed) vs exact-sparse TF-IDF, hit@1")
    ps = []
    for seed in SEEDS:
        test = mcnemar_exact(tfidf_ind[1], rp_indicators_by_seed[seed][1])
        ps.append(test["p"])
        verdict = "significant" if test["p"] < 0.05 else "NOT significant"
        print(f"  seed {seed}: delta {test['delta']:+.4f}  b01={test['b01']} "
              f"b10={test['b10']}  p={test['p']:.4f}  ({verdict})")
    print(f"\n  p-value range across seeds: {min(ps):.4f} - {max(ps):.4f}")
    print(f"\n({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
