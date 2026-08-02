"""Benchmark the CLS memory on LoCoMo -- real conversations, ground-truth recall.

    PYTHONPATH=. .venv/bin/python experiments/benchmark_locomo.py [--conversations N]

The synthetic benchmark answered "does each mechanism work at all". This one
answers the question that decides whether the design is worth anything: **does
it survive real natural language?** LoCoMo supplies dialogue turns with real
timestamps and questions labelled with the exact turns that answer them, so
retrieval is scored against ground truth rather than string overlap.

Expect worse numbers than RESULTS.md. That is the point. Where they are worse
tells you which part of the design was carried by the generator.

Three measurements:

  A  retrieval    recall@k of ground-truth evidence turns, per embedder.
                  Isolated from the gate by storing every turn, so this scores
                  the hippocampus and the embedding geometry, nothing else.
  B  novelty gate what the gate does to real conversation, where there is no
                  clean routine/anomaly split. Held-out sessions measure whether
                  "surprise" tracks anything real or just tracks length.
  C  decay        the forgetting curve on real elapsed time (~231-day spans),
                  including whether it degrades recall of old-but-relevant turns.
"""

from __future__ import annotations

import argparse
import time

import torch

from cls_memory import (
    ConsolidationConfig,
    CortexConfig,
    DecayConfig,
    HashingEmbedder,
    HopfieldConfig,
    KeyConfig,
    LatentSemanticEmbedder,
    MemoryRecord,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from experiments import locomo

SEED = 0
EMBED_DIM = 256


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build(embedder, *, epochs: int = 40, quantile: float = 0.95) -> OrganizationalMemory:
    return OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=embedder.dim,
                hidden_dims=(192, 96),
                latent_dim=32,
                epochs=epochs,
                batch_size=64,
                learning_rate=1e-3,
                kl_weight=0.01,
            ),
            novelty=NoveltyConfig(quantile=quantile, warmup=32, window=8192),
            hopfield=HopfieldConfig(beta=32.0),
            key=KeyConfig(expansion_dim=2048, sparsity_k=256),
            decay=DecayConfig(half_life_days=30.0),
            consolidation=ConsolidationConfig(replay_batch=256),
            seed=SEED,
        ),
        embedder=embedder,
    )


def force_store(system: OrganizationalMemory, turn: locomo.Turn) -> MemoryRecord:
    """Write a turn straight into the store, bypassing the novelty gate.

    Measurement A is about retrieval, so every evidence turn must be present.
    Letting the gate drop turns would conflate "could not retrieve it" with
    "never stored it", and the two have completely different fixes.
    """
    embedding = system.embedder.encode([turn.memory_text])[0]
    latent = system.cortex.latent(embedding)
    record = MemoryRecord(
        text=turn.memory_text,
        embedding=embedding,
        latent=latent,
        key=system.key_encoder(embedding, latent),
        persistence=Persistence.TEMPORAL,
        created_at=turn.timestamp,
        last_reinforced_at=turn.timestamp,
        metadata={"dia_id": turn.dia_id, "session": turn.session},
    )
    return system.store.add(record, now=turn.timestamp)


def make_embedder(name: str, corpus: list[str]):
    if name == "hashing":
        return HashingEmbedder(dim=EMBED_DIM, seed=SEED)
    if name == "lsa":
        return LatentSemanticEmbedder(dim=EMBED_DIM, seed=SEED).fit(corpus)
    raise ValueError(name)


# --------------------------------------------------------------------------- A


def experiment_retrieval(conversations, embedder_names) -> dict:
    banner("A. Retrieval on real dialogue — recall@k against evidence turns")
    print("every turn is stored (gate bypassed), so this scores retrieval alone\n")

    results = {}
    for name in embedder_names:
        corpus = [t.memory_text for c in conversations for t in c.turns]
        t0 = time.time()
        embedder = make_embedder(name, corpus)
        hits = {1: 0, 5: 0, 10: 0}
        asked = 0
        mixtures = 0

        for conv in conversations:
            system = build(embedder)
            # The cortex only supplies latents here; a short fit is enough and
            # keeps the run affordable across embedders and conversations.
            system.cortex.fit(embedder.encode([t.memory_text for t in conv.turns]),
                              epochs=15)
            by_id = {}
            for turn in conv.turns:
                by_id[turn.dia_id] = force_store(system, turn).id

            now = max(t.timestamp for t in conv.turns)
            for question in conv.questions:
                evidence = [by_id[e] for e in question.evidence if e in by_id]
                if not evidence:
                    continue
                asked += 1
                result = system.recall(
                    question.question, top_k=10, reinforce=False, now=now
                )
                ranked = [r.record.id for r in result.results]
                mixtures += int(result.is_gist)
                for k in hits:
                    if any(e in ranked[:k] for e in evidence):
                        hits[k] += 1

        results[name] = {k: v / max(asked, 1) for k, v in hits.items()}
        results[name]["asked"] = asked
        print(f"  {name:<8} dim={embedder.dim:<4} "
              f"recall@1 {results[name][1]:.3f}  "
              f"recall@5 {results[name][5]:.3f}  "
              f"recall@10 {results[name][10]:.3f}  "
              f"({asked} questions, {time.time() - t0:.0f}s)")

    print("\nRandom baseline for scale: with ~590 turns stored, recall@10 by")
    print("chance is about 10/590 = 0.017.")
    return results


# --------------------------------------------------------------------------- B


def experiment_gate(conversations, embedder_name: str) -> None:
    banner("B. Novelty gate on real conversation")
    print("no clean routine/anomaly split exists here, so this reports what the")
    print("gate actually does and whether surprise tracks anything but length\n")

    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = make_embedder(embedder_name, corpus)

    conv = conversations[0]
    split = int(len(conv.turns) * 0.7)
    train = [t.memory_text for t in conv.turns[:split]]
    held = [t.memory_text for t in conv.turns[split:]]

    system = build(embedder, epochs=60)
    system.bootstrap(train)

    s_train = system.cortex.surprise(embedder.encode(train))
    s_held = system.cortex.surprise(embedder.encode(held))
    other = [t.memory_text for t in conversations[1].turns][: len(held)]
    s_other = system.cortex.surprise(embedder.encode(other))

    print(f"  {'population':<28} {'n':>5} {'mean':>8} {'median':>8}")
    for label, scores in (
        ("training turns", s_train),
        ("held-out turns, same convo", s_held),
        ("turns from another convo", s_other),
    ):
        print(f"  {label:<28} {scores.numel():>5} {scores.mean():>8.4f} "
              f"{scores.median():>8.4f}")

    # Does surprise just track sentence length? On a template corpus it could
    # not; on real dialogue it very well might, and that would make the whole
    # gate a length filter wearing a VAE.
    lengths = torch.tensor([float(len(t.split())) for t in held])
    s = s_held
    corr = float(
        ((lengths - lengths.mean()) * (s - s.mean())).mean()
        / (lengths.std() * s.std()).clamp_min(1e-9)
    )
    print(f"\n  correlation(surprise, turn length in words) = {corr:+.3f}")
    print("  a high value here means the gate is largely a length filter")

    stored = sum(1 for t in held if system.log_event(t).was_stored)
    print(f"  gate stored {stored}/{len(held)} held-out turns "
          f"({stored / len(held):.1%}) at quantile "
          f"{system.config.novelty.quantile}")


# --------------------------------------------------------------------------- C


def experiment_decay(conversations, embedder_name: str) -> None:
    banner("C. Forgetting curve on real elapsed time")

    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = make_embedder(embedder_name, corpus)
    conv = conversations[0]

    system = build(embedder)
    system.cortex.fit(embedder.encode([t.memory_text for t in conv.turns]), epochs=15)
    by_id = {t.dia_id: force_store(system, t).id for t in conv.turns}
    now = max(t.timestamp for t in conv.turns)

    print(f"  conversation spans {conv.span_days:.0f} days, "
          f"{len(conv.turns)} turns, half-life "
          f"{system.config.decay.half_life_days:.0f} days")

    salience = system.store.salience_vector(now)
    print(f"  salience at the newest timestamp: min {salience.min():.4f} "
          f"max {salience.max():.4f} mean {salience.mean():.4f}")
    doomed = int((salience < system.config.decay.prune_below).sum())
    print(f"  below the {system.config.decay.prune_below} prune floor: "
          f"{doomed}/{len(salience)} ({doomed / len(salience):.1%})")

    # Recall of old evidence, with and without decay applied.
    old_hits = fresh_hits = asked = 0
    for question in conv.questions:
        evidence = [by_id[e] for e in question.evidence if e in by_id]
        if not evidence:
            continue
        asked += 1
        decayed = system.recall(question.question, top_k=5, reinforce=False, now=now)
        if any(e in [r.record.id for r in decayed.results] for e in evidence):
            old_hits += 1
        # undecayed control: flatten every prior
        system.store.mhn.set_log_prior(torch.zeros(len(system.store)))
        flat = system.store.mhn.retrieve(system.retrieval.encode_cue(question.question))
        top = torch.topk(flat.weights, 5).indices.tolist()
        if any(
            system.store.record_at(i).id in evidence for i in top
        ):
            fresh_hits += 1

    print(f"\n  recall@5 with the forgetting curve applied: {old_hits / asked:.3f}")
    print(f"  recall@5 with all priors flat (no decay):   {fresh_hits / asked:.3f}")
    print("  a large gap means decay is discarding still-wanted memories -- on a")
    print("  231-day span with a 30-day half-life, most turns are far past it")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--embedders", default="hashing,lsa")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    conversations = locomo.load()[: args.conversations]

    print("=" * 78)
    print("CLS ORGANIZATIONAL MEMORY — LoCoMo BENCHMARK (real data)")
    print("=" * 78)
    for key, value in locomo.stats(conversations).items():
        print(f"  {key:<38} {value}")

    names = [n.strip() for n in args.embedders.split(",") if n.strip()]
    best = experiment_retrieval(conversations, names)
    preferred = max(names, key=lambda n: best[n][5])
    print(f"\nbest embedder by recall@5: {preferred}")

    experiment_gate(conversations, preferred)
    experiment_decay(conversations, preferred)

    print(f"\n{'=' * 78}\nDONE\n{'=' * 78}")


if __name__ == "__main__":
    main()
