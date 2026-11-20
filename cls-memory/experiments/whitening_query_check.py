"""Re-measures RESULTS.md Part VI.3 with `WhitenedEmbedder.encode_query` fixed.

    PYTHONPATH=. .venv/bin/python experiments/whitening_query_check.py

Ticket 02 (`.scratch/cls-memory-review/issues/02-...md`): `WhitenedEmbedder`
had no `encode_query`, so every call site of the form
`getattr(embedder, "encode_query", embedder.encode)` (`dense_knn_ranker` here,
and every production call site) silently fell back to `encode` for the
whitened arm while the unwhitened "BGE as shipped" arm above it kept BGE's
query instruction prefix. The VI.3 table -- and the numbers duplicated in
`WhiteningConfig`'s docstring -- were measured through that bug. This script
reproduces the same harness (3 conversations, dense_knn_ranker, `Whitener`
fitted once on the whole corpus) now that `WhitenedEmbedder.encode_query`
delegates correctly, so the "before" and "after" rows are directly comparable.
"""

from __future__ import annotations

import torch

from cls_memory.whitening import WhitenedEmbedder
from experiments import locomo
from experiments.recall_ablation import BGEEmbedder, dense_knn_ranker, score_ranking

SEED = 0


def main() -> None:
    torch.manual_seed(SEED)
    conversations = locomo.load()[:3]
    corpus = [t.memory_text for c in conversations for t in c.turns]
    print(f"{len(conversations)} conversations, {len(corpus)} turns\n")

    bge = BGEEmbedder()
    score_ranking(conversations, lambda c: dense_knn_ranker(c, bge),
                  "BGE as shipped")

    whitened = WhitenedEmbedder(BGEEmbedder()).fit(corpus)
    assert hasattr(whitened, "encode_query"), "fix not applied"
    # Sanity check the fix actually changed something observable: the query
    # path must differ from the document path for the same text once fitted.
    probe = ["did anything change?"]
    assert not torch.allclose(whitened.encode(probe), whitened.encode_query(probe)), (
        "encode_query is not distinguishable from encode -- the fix regressed"
    )
    score_ranking(
        conversations, lambda c: dense_knn_ranker(c, whitened),
        "whitened, fitted once on corpus, encode_query fixed",
    )


if __name__ == "__main__":
    main()
