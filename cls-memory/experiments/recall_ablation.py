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
import hashlib
import math
import pathlib
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
    """Hit@k, strictly: 1 if ANY evidence turn is in the top k.

    Not recall@k -- 93 of 494 LoCoMo questions have 2-6 evidence turns, so this
    sits systematically above true recall. Consistent across every row, so no
    comparison here is affected, but read the published figures as hit@k.
    """
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

    def __init__(
        self, docs: list[str], *, char_ngrams=(3, 5), idf_corpus: list[str] | None = None
    ) -> None:
        # `idf_corpus` exists so this shares an IDF convention with
        # HashedProjection, which fits on the whole corpus. Comparing a
        # per-conversation-IDF exact ranker against a corpus-IDF hashed one
        # confounded the projection with the fitting scope. This hook existed
        # but nothing called it until ticket 05 wired `main()`'s ceilings
        # section through it (RESULTS.md VI.5): measured, moving this class
        # from per-conversation to whole-corpus IDF shifts LoCoMo hit@1 by
        # +0.002 (0.320 -> 0.322, 3 conversations, n=494) -- small, but not
        # nothing next to the ~0.010 gap the RP-4096 retraction turns on.
        self.char_ngrams = char_ngrams
        self.docs = docs
        fit_docs = idf_corpus if idf_corpus is not None else docs
        df: Counter = Counter()
        for doc in fit_docs:
            df.update(set(self._terms(doc)))
        self.doc_terms = [Counter(self._terms(d)) for d in docs]
        n = len(fit_docs)
        self.n_idf_docs = n  # recorded so a run can assert every ceiling row
        # was fit on the same pool -- see main()'s shared `idf_corpus` guard.
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

    def __init__(
        self, docs: list[str], k1: float = 1.5, b: float = 0.75,
        *, idf_corpus: list[str] | None = None,
    ) -> None:
        # `idf_corpus` mirrors `TfidfIndex`'s hook (ticket 05): without it, BM25
        # estimated document frequency from whatever conversation it was
        # ranking -- a much smaller pool than `HashedProjection`, which fits on
        # the whole corpus. Comparing a per-conversation-IDF exact ranker
        # against a corpus-IDF hashed one confounds the "lexical ceiling vs
        # projection" comparison with the fitting scope, not just the method.
        self.k1, self.b = k1, b
        self.docs = [self._terms(d) for d in docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        self.postings: dict[str, list[int]] = {}
        for i, counts in enumerate(self.tf):
            for t in counts:
                self.postings.setdefault(t, []).append(i)

        fit_terms = (
            [self._terms(d) for d in idf_corpus] if idf_corpus is not None else self.docs
        )
        df: Counter = Counter()
        for doc in fit_terms:
            df.update(set(doc))
        n = len(fit_terms)
        self.n_idf_docs = n  # recorded so a run can assert every ceiling row
        # was fit on the same pool -- see main()'s shared `idf_corpus` guard.
        self.idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

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
            # A term can be in `idf` (fit on `idf_corpus`) but absent from this
            # conversation's own postings once the two corpora are allowed to
            # differ (ticket 05) -- `.get` rather than `[term]`, or a term seen
            # only elsewhere in the shared corpus raises a KeyError here.
            for i in self.postings.get(term, ()):
                f = self.tf[i][term]
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1e-9)
                )
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [i for i, _ in ordered[:top]]


class HashedProjection:
    """Signed feature hashing of TF-IDF into a dense vector — a JL projection.

    The alternative to truncated SVD, and the hypothesis this ablation exists
    to test. SVD keeps the top-k directions and *discards the tail entirely*;
    for short-query retrieval the discriminative signal lives in rare terms,
    which are exactly the tail. A Johnson-Lindenstrauss projection instead
    preserves every direction approximately, with distortion O(sqrt(log n / d)).

    Implemented as signed feature hashing rather than an explicit V x d matrix:
    each term hashes to `k` output coordinates with random signs. No large
    allocation, no fitting beyond the vocabulary/IDF pass, and it satisfies the
    `Embedder` protocol (`dim`, `encode`) so it drops into the pipeline
    unchanged if the numbers justify promoting it.

    `weighting="bm25"` swaps TF-IDF for BM25 term weights (saturating term
    frequency, length normalisation), which matter when documents are ~15-word
    dialogue turns.
    """

    def __init__(
        self,
        docs: list[str],
        *,
        dim: int = 1024,
        hashes: int = 2,
        weighting: str = "tfidf",
        k1: float = 1.5,
        b: float = 0.75,
        char_ngrams=(3, 5),
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.hashes = hashes
        self.weighting = weighting
        self.k1, self.b = k1, b
        self.char_ngrams = char_ngrams
        self.seed = seed
        self._cache: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}

        df: Counter = Counter()
        lengths = []
        for doc in docs:
            terms = self._terms(doc)
            lengths.append(len(terms))
            df.update(set(terms))
        n = len(docs)
        self.n_idf_docs = n  # recorded so a run can assert every ceiling row
        # was fit on the same pool -- see main()'s shared `idf_corpus` guard.
        self.avg_len = sum(lengths) / max(n, 1)
        if weighting == "bm25":
            self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        else:
            self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

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

    def _slots(self, term: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Deterministic (indices, signs) for a term. Cached — the same terms
        recur constantly across a corpus."""
        hit = self._cache.get(term)
        if hit is not None:
            return hit
        indices, signs = [], []
        for j in range(self.hashes):
            digest = hashlib.blake2b(
                f"{self.seed}:{j}:{term}".encode("utf-8"), digest_size=8
            ).digest()
            value = int.from_bytes(digest, "little")
            indices.append(value % self.dim)
            signs.append(1.0 if (value >> 63) & 1 else -1.0)
        hit = (tuple(indices), tuple(signs))
        self._cache[term] = hit
        return hit

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        out = torch.zeros(len(texts), self.dim)
        scale = 1.0 / math.sqrt(self.hashes)
        for row, text in enumerate(texts):
            counts = Counter(self._terms(text))
            length = sum(counts.values())
            for term, freq in counts.items():
                idf = self.idf.get(term)
                if idf is None:
                    continue
                if self.weighting == "bm25":
                    denom = freq + self.k1 * (
                        1 - self.b + self.b * length / max(self.avg_len, 1e-9)
                    )
                    weight = idf * freq * (self.k1 + 1) / denom
                else:
                    weight = (1.0 + math.log(freq)) * idf
                indices, signs = self._slots(term)
                for idx, sign in zip(indices, signs):
                    out[row, idx] += weight * sign * scale
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def assert_shared_idf(label: str, index, idf_corpus: list[str]) -> None:
    """Loud guard against ticket 05's confound: a lexical/hashed ceiling row
    fit on a different IDF pool than the rest of the ceilings section it is
    being compared against.

    `TfidfIndex`, `BM25Index` and `HashedProjection` all record `n_idf_docs`
    -- the size of the corpus their document frequencies were estimated from
    -- specifically so this can be checked. HANDOFF.md §3.1d item 1's
    "RP-4096 passes the sparse TF-IDF ceiling" retraction turned on exactly
    this mismatch (TF-IDF/BM25 fit per-conversation, HashedProjection fit on
    the whole corpus); this makes a future regression to that state a hard
    failure instead of a silent one.
    """
    if index.n_idf_docs != len(idf_corpus):
        raise AssertionError(
            f"{label}: fit on {index.n_idf_docs} docs, expected the shared "
            f"IDF corpus of {len(idf_corpus)} docs -- ceilings must share "
            "one IDF convention (ticket 05)."
        )


class SpacyVectorEmbedder:
    """Static word vectors, pooled properly.

    A naive mean of word vectors is a weak baseline and it is not what the
    literature means by "sentence embedding from word vectors". Three things
    matter, and skipping them understates the representation badly:

      * **L2-normalise each token vector before pooling.** Raw spaCy vector
        norms vary several-fold with frequency, so an unnormalised mean is
        dominated by whichever high-norm tokens happen to be present.
      * **Frequency weighting.** `sif` uses Arora et al. (2017) smooth inverse
        frequency, w = a / (a + p(word)), which downweights function words far
        more aggressively than IDF and is the standard choice. `idf` is the
        classic alternative.
      * **Remove the first principal component.** Averaged word vectors share a
        large common direction that carries no discriminative information;
        subtracting it is the other half of SIF and is usually worth more than
        the weighting.

    `model="en_core_web_lg"` has 343k unique vectors against md's 20k shared
    rows. Token *coverage* is 99.8% either way -- md maps 500k keys onto 20k
    rows, so `has_vector` is true but the vector is coarser.
    """

    def __init__(
        self,
        docs: list[str] | None = None,
        *,
        model: str = "en_core_web_lg",
        weighting: str = "sif",
        remove_pc: bool = True,
        normalize_tokens: bool = True,
        sif_a: float = 1e-3,
    ) -> None:
        import spacy

        self._nlp = spacy.load(
            model,
            exclude=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"],
        )
        self.dim = int(self._nlp.vocab.vectors.shape[1])
        self.weighting = weighting
        self.normalize_tokens = normalize_tokens
        self.sif_a = sif_a
        self.weights: dict[str, float] = {}
        self._pc: torch.Tensor | None = None

        if weighting in ("idf", "sif"):
            if docs is None:
                raise ValueError(f"weighting={weighting!r} needs the corpus")
            counts: Counter = Counter()
            doc_freq: Counter = Counter()
            for doc in docs:
                tokens = _tokenize(doc)
                counts.update(tokens)
                doc_freq.update(set(tokens))
            n_docs = len(docs)
            total = max(sum(counts.values()), 1)
            if weighting == "idf":
                self.weights = {
                    t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in doc_freq.items()
                }
            else:
                self.weights = {
                    t: sif_a / (sif_a + c / total) for t, c in counts.items()
                }

        if remove_pc:
            if docs is None:
                raise ValueError("remove_pc needs the corpus")
            matrix = self._pool(docs)
            # First right singular vector of the pooled corpus: the common
            # direction every sentence shares, which carries no signal.
            _, _, vh = torch.linalg.svd(
                matrix - matrix.mean(dim=0, keepdim=True), full_matrices=False
            )
            self._pc = vh[0]

    def _pool(self, texts) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        default = 1.0 if self.weighting == "mean" else self.sif_a
        for row, doc in enumerate(self._nlp.pipe(list(texts), batch_size=256)):
            total = torch.zeros(self.dim)
            weight_sum = 0.0
            for token in doc:
                if not token.has_vector or token.is_punct or token.is_space:
                    continue
                vector = torch.tensor(token.vector)
                if self.normalize_tokens:
                    vector = vector / vector.norm().clamp_min(1e-9)
                weight = (
                    1.0
                    if self.weighting == "mean"
                    else self.weights.get(token.lower_, default)
                )
                total += weight * vector
                weight_sum += weight
            if weight_sum:
                total /= weight_sum
            out[row] = total
        return out

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        out = self._pool(texts)
        if self._pc is not None:
            out = out - (out @ self._pc).unsqueeze(-1) * self._pc
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class HybridEmbedder:
    """Concatenation of a lexical and a semantic embedder, weighted by `w`.

    `w` is the **effective lexical share of the cosine**, which is what you
    actually want to reason about. Scaling the two halves by alpha and
    (1 - alpha) does *not* give that: with both halves unit-norm,

        cos_hybrid = [a^2 cos_lex + (1-a)^2 cos_sem] / (a^2 + (1-a)^2)

    -- quadratic, not linear. An earlier version swept `alpha` and labelled it
    "lexical share", so alpha=0.4 was reported as 40% lexical when the true
    share was 31%, and a grid of (0.5, 0.3, 0.2) probed w = (0.50, 0.155,
    0.059), never testing the lexical-dominant half at all. Scaling by sqrt(w)
    and sqrt(1-w) makes the label true.
    """

    def __init__(self, lexical, semantic, *, w: float = 0.5) -> None:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {w}")
        self.lexical = lexical
        self.semantic = semantic
        self.w = w
        self.dim = lexical.dim + semantic.dim

    def _combine(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        out = torch.cat(
            [left * math.sqrt(self.w), right * math.sqrt(1.0 - self.w)], dim=-1
        )
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return self._combine(
            self.lexical.encode(texts), self.semantic.encode(texts)
        )

    def encode_query(self, texts) -> torch.Tensor:
        """Forward to each half's query encoder where it has one.

        Without this, `dense_knn_ranker`'s getattr fallback silently used the
        *document* encoder for queries, so every hybrid row was measured with
        BGE's query instruction prefix missing while the BGE-alone row above it
        had it -- not an apples-to-apples comparison.
        """
        if isinstance(texts, str):
            texts = [texts]
        left = getattr(self.lexical, "encode_query", self.lexical.encode)(texts)
        right = getattr(self.semantic, "encode_query", self.semantic.encode)(texts)
        return self._combine(left, right)


class BGEEmbedder:
    """BAAI/bge-small-en-v1.5 — a real contextual sentence encoder.

    The first representation here with word order and context. Everything else
    in this file is a bag of tokens: lexical methods match strings, and averaged
    word vectors have no notion of order, so neither can distinguish "the
    contractor deleted the namespace" from "the namespace deleted the
    contractor".

    Two details that matter and are easy to get wrong:

      * **CLS pooling, not mean pooling.** BGE is trained with the CLS token as
        the sentence representation. Mean-pooling it silently degrades quality.
      * **Asymmetric encoding.** BGE v1.5 expects a query instruction prefix on
        *queries only*, never on documents. Omitting it is the single most
        common way to under-measure this model on retrieval, and it is exactly
        the query/document asymmetry the lexical rows here also suffer from.

    Weights come from Qdrant's fastembed mirror, which is reachable where
    HuggingFace is not. ONNX + tokenizers directly, so neither `fastembed` nor
    `transformers` is required at run time.
    """

    URL = "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        batch_size: int = 64,
        use_query_prefix: bool = True,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = pathlib.Path(
            model_dir or pathlib.Path(__file__).parent / "data" / "fast-bge-small-en-v1.5"
        )
        # Check a required file, not just the directory: a partial extraction
        # leaves the directory present and then raises a bare pyo3 Exception
        # from Tokenizer.from_file, which the caller's narrow except did not
        # catch, taking down the rest of the run.
        if not (path / "tokenizer.json").exists():
            self._download(path)

        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tok.enable_padding()
        self._tok.enable_truncation(512)
        onnx = path / "model_optimized.onnx"
        if not onnx.exists():
            onnx = path / "model.onnx"
        self._sess = ort.InferenceSession(
            str(onnx), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.dim = int(self._sess.get_outputs()[0].shape[-1])
        self.batch_size = batch_size
        self.use_query_prefix = use_query_prefix

    @classmethod
    def _download(cls, path: pathlib.Path) -> None:
        import tarfile
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.parent / "bge.tar.gz"
        print(f"  fetching BGE weights -> {path} ...")
        urllib.request.urlretrieve(cls.URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(path.parent)
        archive.unlink(missing_ok=True)

    def _forward(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            encoded = self._tok.encode_batch(chunk)
            feed = {
                "input_ids": torch.tensor(
                    [e.ids for e in encoded], dtype=torch.int64
                ).numpy(),
                "attention_mask": torch.tensor(
                    [e.attention_mask for e in encoded], dtype=torch.int64
                ).numpy(),
            }
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = feed["input_ids"] * 0
            out = self._sess.run(
                None, {k: v for k, v in feed.items() if k in self._inputs}
            )[0]
            rows.append(torch.tensor(out[:, 0]))  # CLS token
        stacked = torch.cat(rows, dim=0)
        return stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return self._forward(list(texts))

    def encode_query(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if self.use_query_prefix:
            texts = [self.QUERY_PREFIX + t for t in texts]
        return self._forward(list(texts))


def dense_knn_ranker(conv, embedder):
    turns = conv.turns
    matrix = embedder.encode([t.memory_text for t in turns])
    # Honour asymmetric encoders (BGE and friends) when they provide it.
    encode_query = getattr(embedder, "encode_query", embedder.encode)

    def rank(question: str) -> list[str]:
        q = encode_query([question])[0]
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
    # Every lexical/hashed ceiling row below is fit against THIS one corpus for
    # its IDF statistics -- ticket 05. TF-IDF and BM25 still *rank* only the
    # candidates of the conversation being scored (`c.turns`), but document
    # frequency comes from the same pool `HashedProjection` fits on. Before this
    # fix, TF-IDF/BM25 estimated IDF per-conversation while HashedProjection
    # used the whole corpus -- measured, that moves LoCoMo hit@1 by +0.002 for
    # TF-IDF and +0.006 for BM25 (RESULTS.md VI.5), which is what the RP-4096
    # "passes the ceiling" retraction (HANDOFF.md §3.1d item 1) rested on being
    # confounded with. `assert_shared_idf` makes a future regression to that
    # mismatch a hard failure instead of a silent one.
    idf_corpus = corpus
    print(f"  IDF corpus: shared, {len(idf_corpus)} docs "
          f"(whole {len(conversations)}-conversation corpus)\n")

    def bm25_ceiling(c):
        idx = BM25Index([t.memory_text for t in c.turns], idf_corpus=idf_corpus)
        assert_shared_idf("BM25", idx, idf_corpus)
        return lambda q: [c.turns[i].dia_id for i in idx.rank(q)]

    def tfidf_ceiling(c):
        idx = TfidfIndex([t.memory_text for t in c.turns], idf_corpus=idf_corpus)
        assert_shared_idf("TF-IDF", idx, idf_corpus)
        return lambda q: [c.turns[i].dia_id for i in idx.rank(q)]

    score_ranking(conversations, bm25_ceiling, "BM25 (sparse, word 1-2 grams)")
    score_ranking(conversations, tfidf_ceiling, "TF-IDF cosine (sparse, no SVD)")
    hashing = HashingEmbedder(dim=256, seed=SEED)
    score_ranking(conversations, lambda c: dense_knn_ranker(c, hashing),
                  "dense kNN, hashing-256")
    for dim in (256, 1024):
        lsa = LatentSemanticEmbedder(dim=dim, seed=SEED).fit(corpus)
        score_ranking(conversations, lambda c, e=lsa: dense_knn_ranker(c, e),
                      f"dense kNN, LSA-{dim} (rank {lsa.rank})")

    # The hypothesis under test: random projection should beat truncated SVD at
    # equal dim, because SVD discards the tail where the retrieval signal lives.
    for dim in (1024, 2048, 4096):
        rp = HashedProjection(idf_corpus, dim=dim, seed=SEED)
        assert_shared_idf(f"random-projection-{dim}", rp, idf_corpus)
        score_ranking(conversations, lambda c, e=rp: dense_knn_ranker(c, e),
                      f"dense kNN, random-projection-{dim}")
    for dim in (1024, 4096):
        rp = HashedProjection(idf_corpus, dim=dim, weighting="bm25", seed=SEED)
        assert_shared_idf(f"BM25-weighted RP-{dim}", rp, idf_corpus)
        score_ranking(conversations, lambda c, e=rp: dense_knn_ranker(c, e),
                      f"dense kNN, BM25-weighted RP-{dim}")

    # Beyond bag-of-words. Every row above matches strings and fails the same
    # way on paraphrase; these are the first that can bridge it. Pooling is
    # swept properly -- a naive mean is a strawman for word-vector embeddings.
    try:
        variants = [
            ("mean, raw", dict(weighting="mean", remove_pc=False, normalize_tokens=False)),
            ("mean, L2 tokens", dict(weighting="mean", remove_pc=False)),
            ("IDF, L2 tokens", dict(weighting="idf", remove_pc=False)),
            ("SIF, no PC removal", dict(weighting="sif", remove_pc=False)),
            ("SIF + PC removal", dict(weighting="sif", remove_pc=True)),
        ]
        best_semantic, best_score, best_label = None, -1.0, ""
        for label, kwargs in variants:
            emb = SpacyVectorEmbedder(corpus, model="en_core_web_lg", **kwargs)
            got = score_ranking(
                conversations, lambda c, e=emb: dense_knn_ranker(c, e),
                f"dense kNN, spaCy-lg {label}",
            )
            if got[1] > best_score:
                best_semantic, best_score, best_label = emb, got[1], label

        print(f"\n  best semantic pooling: {best_label} ({best_score:.3f}@1)")
        best_lexical = HashedProjection(corpus, dim=4096, weighting="bm25", seed=SEED)
        for alpha in (0.8, 0.7, 0.6, 0.5, 0.4):
            hybrid = HybridEmbedder(best_lexical, best_semantic, w=alpha)
            score_ranking(
                conversations, lambda c, e=hybrid: dense_knn_ranker(c, e),
                f"dense kNN, HYBRID RP-4096 + spaCy (w={alpha:g} lexical)",
            )
    except Exception as exc:
        print(f"  [skipped spaCy rows: {type(exc).__name__}: {exc}]")

    # A real contextual encoder -- the first row with word order.
    try:
        bge = BGEEmbedder()
        score_ranking(conversations, lambda c, e=bge: dense_knn_ranker(c, e),
                      f"dense kNN, BGE-small-v1.5 ({bge.dim}d, query prefix)")
        bare = BGEEmbedder(use_query_prefix=False)
        score_ranking(conversations, lambda c, e=bare: dense_knn_ranker(c, e),
                      "dense kNN, BGE-small-v1.5 (no query prefix)")
        lex = HashedProjection(corpus, dim=4096, weighting="bm25", seed=SEED)
        for alpha in (0.8, 0.7, 0.6, 0.5, 0.4):
            score_ranking(
                conversations,
                lambda c, e=HybridEmbedder(lex, bge, w=alpha): dense_knn_ranker(c, e),
                f"dense kNN, HYBRID RP-4096 + BGE (w={alpha:g} lexical)",
            )
    except Exception as exc:  # pyo3 raises bare Exception on a bad model dir
        print(f"  [skipped BGE rows: {type(exc).__name__}: {exc}]")

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
