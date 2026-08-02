"""Where is recall being lost? An ablation from raw text to settled attractor.

    PYTHONPATH=. .venv/bin/python experiments/recall_ablation.py

LoCoMo recall@1 is 0.121 against 1.000 on synthetic data. Before tuning
anything it is worth knowing which stage loses the signal, because the fixes are
completely different:

    text ──▶ [1] embedder ──▶ [2] VAE latent ──▶ [3] DG key ──▶ [4] MHN settling

Each row below adds one stage. A row that drops sharply against the one above it
is the culprit; rows that track each other are not worth tuning.

The top rows are *retrieval ceilings* — plain cosine kNN, no memory system at
all. If the full pipeline matches its ceiling, the pipeline is fine and the
embedding is the whole problem. If it falls well below, the memory system is
discarding signal the embedding had.

Baselines included on purpose:
  * **TF-IDF, no SVD.** Sparse lexical retrieval is a strong baseline for
    short-query QA. If 256-d LSA loses to it, the SVD is destroying the rare
    discriminative terms that make this task work, and the "semantic" embedding
    is a downgrade dressed as an upgrade.
  * **BM25.** The standard sparse retrieval baseline; saturating term frequency
    and normalising by document length matter when documents are ~15-word turns.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

import torch

from cls_memory import (
    CortexConfig,
    HashingEmbedder,
    HopfieldConfig,
    KeyConfig,
    HippocampalKey,
    LatentSemanticEmbedder,
    MemoryRecord,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from cls_memory.embeddings import _tokenize
from experiments import locomo

SEED = 0
KS = (1, 5, 10)


def recall_at_k(ranked_ids: list[str], evidence: set[str], k: int) -> int:
    return int(any(r in evidence for r in ranked_ids[:k]))


def score_ranking(conversations, rank_fn, label: str) -> dict:
    """rank_fn(conversation) -> callable(question_text) -> ranked list of dia_ids."""
    t0 = time.time()
    hits = {k: 0 for k in KS}
    asked = 0
    for conv in conversations:
        rank = rank_fn(conv)
        ids = {t.dia_id for t in conv.turns}
        for question in conv.questions:
            evidence = {e for e in question.evidence if e in ids}
            if not evidence:
                continue
            asked += 1
            ranked = rank(question.question)
            for k in KS:
                hits[k] += recall_at_k(ranked, evidence, k)
    out = {k: hits[k] / max(asked, 1) for k in KS}
    print(f"  {label:<44} " + "  ".join(f"@{k} {out[k]:.3f}" for k in KS)
          + f"   ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ baselines


class TfidfIndex:
    """Sparse TF-IDF cosine. No SVD, no learning -- the honest lexical baseline."""

    def __init__(self, docs: list[str], *, char_ngrams=(3, 5)) -> None:
        self.char_ngrams = char_ngrams
        self.docs = docs
        df: Counter = Counter()
        self.doc_terms = []
        for doc in docs:
            terms = Counter(self._terms(doc))
            self.doc_terms.append(terms)
            df.update(terms.keys())
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.doc_vecs = [self._weight(t) for t in self.doc_terms]

    def _terms(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        terms = list(tokens)
        terms += [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        if self.char_ngrams:
            lo, hi = self.char_ngrams
            padded = f" {text.lower().strip()} "
            for n in range(lo, hi + 1):
                terms += [padded[i : i + n] for i in range(len(padded) - n + 1)]
        return terms

    def _weight(self, counts: Counter) -> dict[str, float]:
        vec = {
            t: (1.0 + math.log(c)) * self.idf.get(t, 0.0)
            for t, c in counts.items()
            if t in self.idf
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def rank(self, query: str, top: int = 10) -> list[int]:
        q = self._weight(Counter(self._terms(query)))
        scores = []
        for i, doc in enumerate(self.doc_vecs):
            if len(q) < len(doc):
                s = sum(w * doc.get(t, 0.0) for t, w in q.items())
            else:
                s = sum(w * q.get(t, 0.0) for t, w in doc.items())
            scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:top]]


class BM25Index:
    """Okapi BM25 over word unigrams+bigrams -- the standard sparse baseline."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [self._terms(d) for d in docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df: Counter = Counter()
        self.tf = []
        for doc in self.docs:
            counts = Counter(doc)
            self.tf.append(counts)
            df.update(counts.keys())
        n = len(self.docs)
        self.idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for i, counts in enumerate(self.tf):
            for t in counts:
                self.postings.setdefault(t, []).append(i)

    @staticmethod
    def _terms(text: str) -> list[str]:
        tokens = _tokenize(text)
        return tokens + [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]

    def rank(self, query: str, top: int = 10) -> list[int]:
        scores: dict[int, float] = {}
        for term in set(self._terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1e-9)
                )
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [i for i, _ in ordered[:top]]


def dense_knn_ranker(conv, embedder):
    turns = conv.turns
    matrix = embedder.encode([t.memory_text for t in turns])

    def rank(question: str) -> list[str]:
        q = embedder.encode([question])[0]
        order = torch.topk(matrix @ q, min(10, len(turns))).indices.tolist()
        return [turns[i].dia_id for i in order]

    return rank


# ------------------------------------------------------------- full pipeline


def build_system(embedder, key_mode: HippocampalKey, beta: float, sparsity_k: int):
    return OrganizationalMemory(
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
            key=KeyConfig(mode=key_mode, expansion_dim=2048, sparsity_k=sparsity_k),
            seed=SEED,
        ),
        embedder=embedder,
    )


def pipeline_ranker(conv, embedder, key_mode, beta, sparsity_k):
    system = build_system(embedder, key_mode, beta, sparsity_k)
    system.cortex.fit(
        embedder.encode([t.memory_text for t in conv.turns]), epochs=15
    )
    id_of = {}
    for turn in conv.turns:
        embedding = embedder.encode([turn.memory_text])[0]
        latent = system.cortex.latent(embedding)
        record = MemoryRecord(
            text=turn.memory_text,
            embedding=embedding,
            latent=latent,
            key=system.key_encoder(embedding, latent),
            persistence=Persistence.TEMPORAL,
            created_at=turn.timestamp,
            last_reinforced_at=turn.timestamp,
        )
        system.store.add(record, now=turn.timestamp)
        id_of[record.id] = turn.dia_id
    now = max(t.timestamp for t in conv.turns)

    def rank(question: str) -> list[str]:
        result = system.recall(question, top_k=10, reinforce=False, now=now)
        return [id_of[r.record.id] for r in result.results]

    return rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    conversations = locomo.load()[: args.conversations]
    corpus = [t.memory_text for c in conversations for t in c.turns]
    n_turns = len(corpus)
    print("=" * 78)
    print("RECALL ABLATION — where does the signal go?")
    print("=" * 78)
    print(f"{len(conversations)} conversations, {n_turns} turns\n")

    print("RETRIEVAL CEILINGS (no memory system -- plain ranking over turns)")
    score_ranking(
        conversations,
        lambda c: (lambda idx: lambda q: [c.turns[i].dia_id for i in idx.rank(q)])(
            BM25Index([t.memory_text for t in c.turns])
        ),
        "BM25 (sparse, word 1-2 grams)",
    )
    score_ranking(
        conversations,
        lambda c: (lambda idx: lambda q: [c.turns[i].dia_id for i in idx.rank(q)])(
            TfidfIndex([t.memory_text for t in c.turns])
        ),
        "TF-IDF cosine (sparse, no SVD)",
    )
    hashing = HashingEmbedder(dim=256, seed=SEED)
    score_ranking(conversations, lambda c: dense_knn_ranker(c, hashing),
                  "dense kNN, hashing-256")
    for dim in (256, 1024):
        lsa = LatentSemanticEmbedder(dim=dim, seed=SEED).fit(corpus)
        score_ranking(conversations, lambda c, e=lsa: dense_knn_ranker(c, e),
                      f"dense kNN, LSA-{dim} (rank {lsa.rank})")

    print("\nFULL PIPELINE (VAE latent -> DG key -> Hopfield settling)")
    lsa256 = LatentSemanticEmbedder(dim=256, seed=SEED).fit(corpus)
    for key_mode in (HippocampalKey.SEPARATED, HippocampalKey.EMBEDDING):
        score_ranking(
            conversations,
            lambda c, m=key_mode: pipeline_ranker(c, lsa256, m, 32.0, 256),
            f"LSA-256, key={key_mode.value}, beta=32",
        )
    for beta in (8.0, 128.0):
        score_ranking(
            conversations,
            lambda c, b=beta: pipeline_ranker(
                c, lsa256, HippocampalKey.EMBEDDING, b, 256
            ),
            f"LSA-256, key=embedding, beta={beta:g}",
        )
    for k in (512, 1024):
        score_ranking(
            conversations,
            lambda c, kk=k: pipeline_ranker(
                c, lsa256, HippocampalKey.SEPARATED, 32.0, kk
            ),
            f"LSA-256, key=separated, sparsity_k={k}",
        )

    print("\n" + "=" * 78)
    print("Read the ceilings against the pipeline rows: a pipeline row close to")
    print("its embedder's ceiling means the memory system is not the problem.")
    print("=" * 78)


if __name__ == "__main__":
    main()
