"""Did the recall changes actually help, and do they hold on a second corpus?

    PYTHONPATH=. .venv/bin/python experiments/recall_check.py

Applies each default change cumulatively and measures on **two** unrelated
corpora — LoCoMo (two-person personal dialogue) and QMSum (many-speaker
organisational meetings). A change that helps one and hurts the other is
overfitting to a corpus, and the point of running both is to catch that.

Rows are cumulative, so each line isolates one decision.
"""

from __future__ import annotations

import argparse
import time

import torch

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
from experiments import locomo, qmsum

SEED = 0
KS = (1, 5, 10)


def evaluate(conversations, *, dim: int, key_mode: HippocampalKey, beta: float) -> dict:
    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = LatentSemanticEmbedder(dim=dim, seed=SEED).fit(corpus)

    hits = {k: 0 for k in KS}
    asked = 0
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
            dia_of[record.id] = turn.dia_id

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
                if any(e in ranked[:k] for e in evidence):
                    hits[k] += 1

    out = {k: hits[k] / max(asked, 1) for k in KS}
    out["asked"] = asked
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", type=int, default=3)
    parser.add_argument("--qmsum", type=int, default=25)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    datasets = [
        ("LoCoMo", locomo.load()[: args.locomo]),
        ("QMSum", qmsum.load(max_meetings=args.qmsum)),
    ]
    for name, convs in datasets:
        print(f"{name}: {len(convs)} conversations, "
              f"{sum(len(c.turns) for c in convs)} turns")

    configs = [
        ("was:  LSA-256, separated, beta=8", 256, HippocampalKey.SEPARATED, 8.0),
        ("  +  beta=128", 256, HippocampalKey.SEPARATED, 128.0),
        ("  +  key=embedding", 256, HippocampalKey.EMBEDDING, 128.0),
        ("now:  +  LSA-1024", 1024, HippocampalKey.EMBEDDING, 128.0),
    ]

    for name, convs in datasets:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"  {'configuration':<36} " + "  ".join(f"@{k}" .rjust(7) for k in KS))
        print("  " + "-" * 74)
        for label, dim, key_mode, beta in configs:
            t0 = time.time()
            r = evaluate(convs, dim=dim, key_mode=key_mode, beta=beta)
            print(f"  {label:<36} " + "  ".join(f"{r[k]:7.3f}" for k in KS)
                  + f"   ({r['asked']} q, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
