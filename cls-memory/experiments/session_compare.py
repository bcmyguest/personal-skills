"""Apples-to-apples: evaluate at the granularity published baselines use.

    PYTHONPATH=. .venv/bin/python experiments/session_compare.py

Published LoCoMo retrieval baselines retrieve **sessions** — BM25 top-3 out of
roughly 20-30 per conversation. Everything else in this repo retrieves
**turns** — top-1 out of ~590. Those are not the same task: the turn task has
~20x the candidates and no surrounding context, so this project's numbers have
never been comparable to anyone else's, in either direction.

This script runs both granularities through one harness so the comparison is
sound, and includes BM25 as a calibration point: if BM25 at session granularity
lands near published BM25 numbers, the harness is measuring what theirs does
and the other rows can be read against the literature. If it does not, the gap
is in the harness and no cross-paper claim should be made at all.

Metric is hit@k throughout — 1 if any evidence item is in the top k.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

from experiments import locomo
from experiments.metrics import mcnemar_exact
from experiments.recall_ablation import (
    BGEEmbedder,
    BM25Index,
    HashedProjection,
    HybridEmbedder,
)

SEED = 0


def session_documents(conv) -> tuple[list[str], list[str], dict[str, str]]:
    """One document per session, plus a map from dia_id to its session."""
    grouped: dict[str, list[str]] = defaultdict(list)
    session_of: dict[str, str] = {}
    for turn in conv.turns:
        grouped[turn.session].append(turn.memory_text)
        session_of[turn.dia_id] = turn.session
    names = sorted(grouped, key=lambda s: int(s.split("_")[1]))
    return names, ["\n".join(grouped[n]) for n in names], session_of


def evaluate(conversations, rank_builder, label: str, *, granularity: str,
             ks=(1, 3, 5, 10)) -> dict:
    """Returns per-question hit indicators at each k, for paired testing."""
    t0 = time.time()
    indicators = {k: [] for k in ks}
    candidates = []

    for conv in conversations:
        if granularity == "session":
            names, docs, session_of = session_documents(conv)
            targets = names
        else:
            docs = [t.memory_text for t in conv.turns]
            targets = [t.dia_id for t in conv.turns]
            session_of = None
        candidates.append(len(docs))
        rank = rank_builder(docs, targets)

        present = set(targets)
        for question in conv.questions:
            if session_of is not None:
                gold = {session_of[e] for e in question.evidence if e in session_of}
            else:
                gold = {e for e in question.evidence if e in present}
            if not gold:
                continue
            ranked = rank(question.question)
            for k in ks:
                indicators[k].append(int(any(r in gold for r in ranked[:k])))

    n = len(indicators[ks[0]])
    scores = {k: sum(v) / max(n, 1) for k, v in indicators.items()}
    mean_candidates = sum(candidates) / len(candidates)
    print(f"  {label:<40} " + "  ".join(f"@{k} {scores[k]:.3f}" for k in ks)
          + f"   (n={n}, {mean_candidates:.0f} candidates, {time.time() - t0:.0f}s)")
    return {"scores": scores, "indicators": indicators}


def bm25_builder(docs, targets):
    index = BM25Index(docs)
    return lambda q: [targets[i] for i in index.rank(q, top=10)]


def dense_builder(embedder_factory):
    def build(docs, targets):
        embedder = embedder_factory(docs)
        matrix = embedder.encode(docs)
        encode_query = getattr(embedder, "encode_query", embedder.encode)

        def rank(question: str):
            q = encode_query([question])[0]
            order = torch.topk(matrix @ q, min(10, len(docs))).indices.tolist()
            return [targets[i] for i in order]

        return rank

    return build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    conversations = locomo.load()[: args.conversations]
    print(f"{len(conversations)} conversations, "
          f"{sum(len(c.turns) for c in conversations)} turns\n")

    try:
        bge = BGEEmbedder()
    except Exception as exc:
        print(f"[BGE unavailable: {exc}]")
        bge = None

    def hybrid_factory(docs):
        lexical = HashedProjection(docs, dim=4096, weighting="bm25", seed=SEED)
        return HybridEmbedder(lexical, bge, w=0.5)

    for granularity in ("session", "turn"):
        print(f"{'=' * 78}\n{granularity.upper()} granularity"
              f"{'  <- what published baselines use' if granularity == 'session' else ''}"
              f"\n{'=' * 78}")
        results = {}
        results["BM25"] = evaluate(
            conversations, bm25_builder, "BM25 (calibration baseline)",
            granularity=granularity,
        )
        if bge is not None:
            results["BGE"] = evaluate(
                conversations, dense_builder(lambda d: bge), "BGE-small alone",
                granularity=granularity,
            )
            results["hybrid"] = evaluate(
                conversations, dense_builder(hybrid_factory),
                "HYBRID RP-4096 + BGE (w=0.5)", granularity=granularity,
            )
            for k in (1, 5):
                test = mcnemar_exact(
                    results["BM25"]["indicators"][k],
                    results["hybrid"]["indicators"][k],
                )
                verdict = "significant" if test["p"] < 0.05 else "NOT significant"
                print(f"    hybrid vs BM25 @{k}: delta {test['delta']:+.3f}  "
                      f"p={test['p']:.4f}  ({verdict})")
        print()


if __name__ == "__main__":
    main()
