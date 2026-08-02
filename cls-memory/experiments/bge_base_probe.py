"""Does the bigger BGE help, and does fastembed give us a route to it?

    PYTHONPATH=. .venv/bin/python experiments/bge_base_probe.py

`bge-small-en-v1.5` was loaded by hand from Qdrant's GCS tarball. fastembed can
reach four models by direct URL when HuggingFace is blocked -- it logs an HF
failure and falls back automatically -- and one of them is
**bge-base-en-v1.5**: 768 dimensions against small's 384, and stronger on MTEB.

This probe answers two things the main ablation does not:
  1. is fastembed a usable route here at all, or does it hard-depend on HF?
  2. does the larger encoder move recall enough to justify 2x the width?

Kept separate from `recall_ablation.py` so that file stays stable while it is
under review; fold the winning row in afterwards.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import torch

from experiments import locomo
from experiments.recall_ablation import (
    HashedProjection,
    HybridEmbedder,
    dense_knn_ranker,
    score_ranking,
)

SEED = 0
CACHE = str(pathlib.Path(__file__).parent / "data")


class FastEmbedEncoder:
    """BGE via fastembed, with the retrieval query prefix applied correctly.

    fastembed tries HuggingFace first and falls back to a direct URL for the
    handful of models that still carry one. That fallback is what makes it work
    in a network like this one; the HF error it logs on the way is expected and
    harmless.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, *, use_query_prefix: bool = True) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=CACHE)
        self.use_query_prefix = use_query_prefix
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def _encode(self, texts: list[str]) -> torch.Tensor:
        rows = [torch.tensor(v) for v in self._model.embed(texts)]
        out = torch.stack(rows)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(list(texts))

    def encode_query(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if self.use_query_prefix:
            texts = [self.QUERY_PREFIX + t for t in texts]
        return self._encode(list(texts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    conversations = locomo.load()[: args.conversations]
    corpus = [t.memory_text for c in conversations for t in c.turns]
    print(f"{len(conversations)} conversations, {len(corpus)} turns\n")

    lexical = HashedProjection(corpus, dim=4096, weighting="bm25", seed=SEED)
    score_ranking(conversations, lambda c: dense_knn_ranker(c, lexical),
                  "BM25-weighted RP-4096 (lexical reference)")

    t0 = time.time()
    base = FastEmbedEncoder("BAAI/bge-base-en-v1.5")
    print(f"  [fastembed loaded bge-base-en-v1.5, {base.dim}d, "
          f"{time.time() - t0:.0f}s]")
    score_ranking(conversations, lambda c: dense_knn_ranker(c, base),
                  f"BGE-base-v1.5 ({base.dim}d) alone")

    for alpha in (0.6, 0.5, 0.4, 0.3):
        score_ranking(
            conversations,
            lambda c, a=alpha: dense_knn_ranker(
                c, HybridEmbedder(lexical, base, alpha=a)
            ),
            f"HYBRID RP-4096 + BGE-base (alpha={alpha:g} lexical)",
        )


if __name__ == "__main__":
    main()
