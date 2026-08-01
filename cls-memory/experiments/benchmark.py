"""Train and evaluate the CLS memory system on a labelled synthetic corpus.

    PYTHONPATH=. .venv/bin/python experiments/benchmark.py [--quick]

Seven experiments, each answering a question the README asserts an answer to:

  1  novelty gate       Can VAE reconstruction error separate anomalies from
                        routine text? (ROC AUC + gate routing precision/recall)
  2  key representation Does the DG-separated key beat the VAE latent at scale?
  3  retrieval          Recall@1 from degraded natural-language cues.
  4  completion         Recall@1 vs fraction of the key that is masked out.
  5  capacity           Does retrieval degrade as the hippocampus fills?
  6  forgetting curve   Evergreen vs temporal survival and retrieval ordering.
  7  consolidation      Does interleaved replay actually reduce forgetting?

Everything is seeded. Numbers printed here are what the README should cite.
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from cls_memory import (
    ConsolidationConfig,
    CortexConfig,
    DecayConfig,
    HippocampalKey,
    HopfieldConfig,
    IngestionAction,
    KeyConfig,
    MemoryRecord,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
    Persistence,
)
from experiments import synthetic
from experiments.metrics import classification, roc_auc
from experiments.synthetic import Label

SEED = 0


def cue_is_unambiguous(cue: str, target: str, corpus: list[str]) -> bool:
    """True if the cue appears verbatim in exactly one stored text.

    Retrieval cannot be blamed for failing to pick between memories that both
    literally contain the query, so recall is reported both overall and
    restricted to cues that admit a single answer.
    """
    matches = [t for t in corpus if cue in t]
    return len(matches) == 1 and matches[0] == target


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nEXPERIMENT {n}: {title}\n{'=' * 78}")


def build_system(
    *,
    key_mode: HippocampalKey = HippocampalKey.SEPARATED,
    embed_dim: int = 256,
    latent_dim: int = 32,
    epochs: int = 60,
    quantile: float = 0.95,
    beta: float = 32.0,
) -> OrganizationalMemory:
    return OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=embed_dim,
                hidden_dims=(192, 96),
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=64,
                learning_rate=1e-3,
                kl_weight=0.01,
            ),
            novelty=NoveltyConfig(quantile=quantile, warmup=32, window=4096),
            hopfield=HopfieldConfig(beta=beta),
            key=KeyConfig(mode=key_mode, expansion_dim=2048, sparsity_k=256),
            decay=DecayConfig(half_life_days=30.0),
            consolidation=ConsolidationConfig(replay_batch=256, replay_steps=8),
            seed=SEED,
        )
    )


# --------------------------------------------------------------------------- 1


def experiment_novelty(routine, anomaly, rules, epochs: int) -> OrganizationalMemory:
    banner(1, "Novelty gate — can reconstruction error find the anomalies?")

    train = [i.text for i in routine[: int(len(routine) * 0.8)]]
    held_routine = [i.text for i in routine[int(len(routine) * 0.8) :]]
    anomaly_texts = [i.text for i in anomaly]

    system = build_system(epochs=epochs)
    t0 = time.time()
    report = system.bootstrap(train)
    print(f"trained on {len(train)} routine documents in {time.time() - t0:.1f}s")
    print(f"final VAE loss     {report.final_loss:.4f}")
    print(f"gate threshold     {report.novelty_threshold:.4f} "
          f"({system.config.novelty.quantile:.0%} quantile of training surprise)")

    s_train = system.cortex.surprise(system.embedder.encode(train))
    s_routine = system.cortex.surprise(system.embedder.encode(held_routine))
    s_anomaly = system.cortex.surprise(system.embedder.encode(anomaly_texts))
    s_rules = system.cortex.surprise(system.embedder.encode([i.text for i in rules]))

    print("\nsurprise distribution (reconstruction error)")
    for name, s in (
        ("train routine ", s_train),
        ("held-out routine", s_routine),
        ("anomaly       ", s_anomaly),
        ("evergreen rule", s_rules),
    ):
        print(f"  {name:<17} n={s.numel():<5} mean={s.mean():.4f} "
              f"median={s.median():.4f} p90={s.quantile(0.9):.4f}")

    auc = roc_auc(s_anomaly, s_routine)
    print(f"\nROC AUC (anomaly vs held-out routine): {auc:.4f}")

    # Route the held-out stream through the real gate, in a shuffled order.
    stream = [(t, False) for t in held_routine] + [(t, True) for t in anomaly_texts]
    random.Random(SEED).shuffle(stream)
    predicted, actual = [], []
    for text, is_anomaly in stream:
        result = system.log_event(text)
        predicted.append(result.was_stored)
        actual.append(is_anomaly)

    print(f"gate routing:  {classification(predicted, actual)}")
    print(f"hippocampus holds {len(system)} memories after a "
          f"{len(stream)}-item stream")

    # A high AUC with poor recall means the *ranking* is good and the
    # *threshold* is misplaced. Sweep the quantile to find the operating point,
    # scoring offline so the sweep does not mutate any memory.
    print("\noperating point sweep (offline, same trained cortex):")
    print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>7}")
    print("-" * 78)
    best = (0.0, None)
    for q in (0.999, 0.99, 0.975, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        threshold = float(torch.tensor(sorted(s_train.tolist())).quantile(q))
        pred = [float(x) > threshold for x in s_routine.tolist()] + [
            float(x) > threshold for x in s_anomaly.tolist()
        ]
        truth = [False] * s_routine.numel() + [True] * s_anomaly.numel()
        rep = classification(pred, truth)
        marker = ""
        if rep.f1 > best[0]:
            best = (rep.f1, q)
            marker = ""
        print(f"{q:>9.3f} {threshold:>10.4f} {rep.precision:>10.3f} "
              f"{rep.recall:>8.3f} {rep.f1:>7.3f}{marker}")
    print(f"best F1 at quantile {best[1]} (configured: "
          f"{system.config.novelty.quantile})")

    # CONFOUND CONTROL. Anomalies carry unique incident references, which are
    # unseen tokens -- so part of the measured novelty is "contains a new
    # number", not "violates the schema". Re-score against anomalies generated
    # without references to isolate structural novelty.
    plain = synthetic.generate(
        n_routine=1, n_anomaly=len(anomaly_texts), include_rules=False,
        unique_anomalies=False, seed=SEED + 1,
    )
    plain_texts = [i.text for i in plain if i.label is Label.ANOMALY]
    s_plain = system.cortex.surprise(system.embedder.encode(plain_texts))
    auc_plain = roc_auc(s_plain, s_routine)
    print("\ncontrol -- anomalies WITHOUT unique incident references:")
    print(f"  mean surprise {s_plain.mean():.4f} (with refs: {s_anomaly.mean():.4f})")
    print(f"  ROC AUC       {auc_plain:.4f} (with refs: {auc:.4f})")
    print("  the gap is the share of detection attributable to unseen tokens")
    return system


# --------------------------------------------------------------------------- 2


def experiment_key_representation(routine, anomaly, epochs: int) -> None:
    banner(2, "Key representation — DG-separated vs raw embedding vs VAE latent")

    train = [i.text for i in routine[: int(len(routine) * 0.8)]]
    targets = [i.text for i in anomaly]
    rng = random.Random(SEED)
    cues = [synthetic.partial_cue(t, 0.5, rng=rng) for t in targets]

    print("cues are 50% contiguous fragments; only unambiguous cues are scored")
    print(f"{'key mode':<12} {'dim':>5} {'recall@1':>9} {'recall@3':>9} "
          f"{'max offdiag':>12} {'mean offdiag':>13}")
    print("-" * 78)

    for mode in (HippocampalKey.SEPARATED, HippocampalKey.EMBEDDING, HippocampalKey.LATENT):
        system = build_system(key_mode=mode, epochs=epochs)
        system.bootstrap(train)

        records = []
        for text in targets:
            # Force-store every anomaly so all modes hold an identical set;
            # this isolates the key representation from gate behaviour.
            records.append(system.ingest(text, persistence=Persistence.TEMPORAL).record)
        stored = [(t, r) for t, r in zip(targets, records) if r is not None]

        all_texts = [t for t, _ in stored]
        hit1 = hit3 = 0
        evaluated = 0
        for (text, record), cue in zip(stored, cues):
            if not cue_is_unambiguous(cue, text, all_texts):
                continue  # the cue admits several correct answers
            evaluated += 1
            result = system.recall(cue, top_k=3, reinforce=False)
            ids = [h.record.id for h in result.results]
            hit1 += int(ids[:1] == [record.id])
            hit3 += int(record.id in ids)

        p = system.store.mhn.patterns
        gram = (p @ p.T - torch.eye(len(p))).abs()
        n = len(p)
        mean_off = float(gram.sum() / (n * (n - 1))) if n > 1 else 0.0
        print(f"{mode.value:<12} {system.store.mhn.dim:>5} "
              f"{hit1 / evaluated:>9.3f} {hit3 / evaluated:>9.3f} "
              f"{float(gram.max()):>12.3f} {mean_off:>13.3f}")


# --------------------------------------------------------------------------- 3


def experiment_retrieval(system: OrganizationalMemory, anomaly) -> None:
    banner(3, "Retrieval — recall@1 vs how much of the query survives")

    stored = [r for r in system.records if not r.is_evergreen]
    by_text = {r.text: r for r in stored}
    targets = [i.text for i in anomaly if i.text in by_text]
    print(f"{len(targets)} of {len(anomaly)} anomalies are in the hippocampus\n")

    all_texts = [r.text for r in stored]
    print(f"{'cue fraction':>13} {'recall@1':>9} {'recall@3':>9} "
          f"{'R@1 unambig':>12} {'n unambig':>10} {'mean conf':>10}")
    print("-" * 78)
    for fraction in (1.0, 0.75, 0.5, 0.35, 0.25):
        rng = random.Random(SEED)
        hit1 = hit3 = 0
        unambiguous = unambiguous_hits = 0
        confidence = 0.0
        for text in targets:
            cue = synthetic.partial_cue(text, fraction, rng=rng)
            result = system.recall(cue, top_k=3, reinforce=False)
            ids = [h.record.id for h in result.results]
            target_id = by_text[text].id
            correct = ids[:1] == [target_id]
            hit1 += int(correct)
            hit3 += int(target_id in ids)
            confidence += result.confidence
            if cue_is_unambiguous(cue, text, all_texts):
                unambiguous += 1
                unambiguous_hits += int(correct)
        n = len(targets)
        ua = unambiguous_hits / unambiguous if unambiguous else float("nan")
        print(f"{fraction:>13.2f} {hit1 / n:>9.3f} {hit3 / n:>9.3f} "
              f"{ua:>12.3f} {unambiguous:>10d} {confidence / n:>10.3f}")

    # Out-of-distribution control: nothing in memory should look like a match.
    print("\nconfabulation control (queries with no stored answer):")
    for query in [
        "quantum chromodynamics lattice gauge simulation",
        "the marmalade recipe calls for seville oranges",
    ]:
        result = system.recall(query, reinforce=False)
        print(f"  depth={result.basin.depth:6.3f} conf={result.confidence:.3f} "
              f"flagged={result.basin.is_confabulation}  {query[:40]!r}")


# --------------------------------------------------------------------------- 4


def experiment_completion(system: OrganizationalMemory, anomaly) -> None:
    banner(4, "Pattern completion — masked keys, clamped known coordinates")

    stored = {r.text: r for r in system.records if not r.is_evergreen}
    targets = [i.text for i in anomaly if i.text in stored][:40]
    sparsity = system.key_encoder.dg.sparsity() if system.key_encoder.dg else 1.0
    print(f"key is {system.store.mhn.dim}-d, {sparsity:.0%} of units active\n")

    print("(a) occluding raw COORDINATES — most of which are zero in a sparse code")
    print(f"{'kept':>6} {'recall@1':>9} {'active kept':>12} {'clamp ok':>9} {'iters':>7}")
    print("-" * 78)
    for keep in (0.8, 0.6, 0.4, 0.3, 0.2, 0.1):
        hits = clamped = 0
        active_kept = 0.0
        iterations = 0
        for seed, text in enumerate(targets):
            partial, mask = system.retrieval.occlude(text, keep, seed=seed)
            full = system.retrieval.encode_cue(text)
            n_active = int((full != 0).sum())
            active_kept += int(((partial != 0) & mask).sum()) / max(n_active, 1)
            result = system.complete(partial, mask, top_k=1, reinforce=False)
            hits += int(result.top.record.id == stored[text].id)
            clamped += int(
                torch.allclose(result.trace.state[mask], partial[mask], atol=1e-6)
            )
            iterations += result.trace.iterations
        n = len(targets)
        print(f"{keep:>6.2f} {hits / n:>9.3f} {active_kept / n:>12.3f} "
              f"{clamped / n:>9.3f} {iterations / n:>7.1f}")

    print("\n(b) occluding ACTIVE UNITS — the honest difficulty scale for a")
    print("    sparse code: keep a fraction of the non-zero units only")
    print(f"{'kept':>6} {'recall@1':>9} {'units kept':>11} {'cue cos':>9} {'iters':>7}")
    print("-" * 78)
    for keep in (0.8, 0.5, 0.3, 0.2, 0.1, 0.05):
        hits = 0
        units = 0.0
        cos = 0.0
        iterations = 0
        for seed, text in enumerate(targets):
            full = system.retrieval.encode_cue(text)
            active = (full != 0).nonzero(as_tuple=True)[0]
            g = torch.Generator().manual_seed(seed)
            n_keep = max(1, int(round(len(active) * keep)))
            chosen = active[torch.randperm(len(active), generator=g)[:n_keep]]
            mask = torch.zeros_like(full, dtype=torch.bool)
            mask[chosen] = True
            partial = full * mask
            units += n_keep
            cos += float(
                torch.nn.functional.cosine_similarity(partial, full, dim=0)
            )
            result = system.complete(partial, mask, top_k=1, reinforce=False)
            hits += int(result.top.record.id == stored[text].id)
            iterations += result.trace.iterations
        n = len(targets)
        print(f"{keep:>6.2f} {hits / n:>9.3f} {units / n:>11.1f} {cos / n:>9.3f} "
              f"{iterations / n:>7.1f}")


# --------------------------------------------------------------------------- 5


def experiment_capacity(routine, anomaly, epochs: int) -> None:
    banner(5, "Capacity — does recall degrade as the hippocampus fills?")

    train = [i.text for i in routine[: int(len(routine) * 0.8)]]
    system = build_system(epochs=epochs)
    system.bootstrap(train)

    probe_texts = [i.text for i in anomaly[:40]]
    probes: list = []
    for text in probe_texts:
        probes.append((text, system.ingest(text).record))
    probes = [(t, r) for t, r in probes if r is not None]

    # Filler must bypass the novelty gate: routine text is *supposed* to be
    # routed PREDICTED, so gated ingestion would never grow the store and the
    # capacity question would go unmeasured. Writing straight to the store is
    # the honest way to load it up -- we are testing the Hopfield layer's
    # behaviour under load, not the gate's.
    filler = [i.text for i in routine[int(len(routine) * 0.8) :]]

    def force_store(text: str) -> None:
        embedding = system.embedder.encode([text])[0]
        latent = system.cortex.latent(embedding)
        system.store.add(
            MemoryRecord(
                text=text,
                embedding=embedding,
                latent=latent,
                key=system.key_encoder(embedding, latent),
                persistence=Persistence.TEMPORAL,
            )
        )

    def evaluate() -> tuple[float, float, float]:
        hits = 0
        for text, record in probes:
            cue = synthetic.partial_cue(text, 0.5, rng=random.Random(SEED))
            result = system.recall(cue, top_k=1, reinforce=False)
            hits += int(result.top.record.id == record.id)
        seps = [system.store.mhn.separation(i) for i in range(len(system))]
        seps = [s for s in seps if s != float("inf")]
        return hits / len(probes), min(seps), sum(seps) / len(seps)

    print(f"{'stored':>8} {'recall@1':>9} {'min separation':>15} "
          f"{'mean separation':>16}")
    print("-" * 78)

    pending = [n for n in (50, 100, 200, 400, 800) if n >= len(system)]
    recall, min_sep, mean_sep = evaluate()
    print(f"{len(system):>8} {recall:>9.3f} {min_sep:>15.4f} {mean_sep:>16.4f}")

    for extra in filler:
        force_store(extra)
        if pending and len(system) >= pending[0]:
            pending.pop(0)
            recall, min_sep, mean_sep = evaluate()
            print(f"{len(system):>8} {recall:>9.3f} {min_sep:>15.4f} "
                  f"{mean_sep:>16.4f}")
        if not pending:
            break


# --------------------------------------------------------------------------- 6


def experiment_forgetting(system: OrganizationalMemory, rules) -> None:
    banner(6, "Forgetting curve — evergreen vs temporal over simulated time")

    for item in rules:
        system.remember_rule(item.text)

    print(f"{'day':>6} {'memories':>9} {'evergreen':>10} {'temporal':>9} "
          f"{'mean salience':>14}")
    print("-" * 78)
    elapsed = 0
    for target_day in (0, 30, 60, 90, 180, 365):
        step = target_day - elapsed
        if step:
            for record in system.records:
                record.age_by(step)
            elapsed = target_day
        system.sweep()  # maintenance first, so each row is post-sweep state
        stats = system.stats()
        print(f"{target_day:>6} {stats['total']:>9} {stats['evergreen']:>10} "
              f"{stats['temporal']:>9} {stats['mean_salience']:>14.4f}")

    stats = system.stats()
    print(f"\nafter one simulated year: {stats['evergreen']} evergreen and "
          f"{stats['temporal']} temporal memories survive")
    assert stats["evergreen"] == len(rules), "evergreen rules must be immortal"
    print("all evergreen rules retained (asserted)")


# --------------------------------------------------------------------------- 7


def experiment_consolidation(routine, anomaly, epochs: int) -> None:
    banner(7, "Consolidation — does interleaved replay reduce forgetting?")

    train = [i.text for i in routine[: int(len(routine) * 0.8)]]
    old = [i.text for i in routine[: 400]]
    intruder = [i.text for i in anomaly] * 4

    results = {}
    for label, interleave in (("no replay (naive)", False), ("interleaved replay", True)):
        system = build_system(epochs=epochs)
        system.bootstrap(train)
        for text in anomaly[:30]:
            system.ingest(text.text)

        before = float(system.cortex.surprise(system.embedder.encode(old)).mean())
        new_data = system.embedder.encode(intruder)
        system.consolidation.config.prune_predicted = False
        if interleave:
            system.consolidation.consolidate(new_data, epochs=30)
        else:
            system.cortex.fit(new_data, epochs=30)
        after = float(system.cortex.surprise(system.embedder.encode(old)).mean())
        results[label] = (before, after, after - before)
        print(f"{label:<20} schema surprise on old routine: "
              f"{before:.4f} -> {after:.4f}  (drift {after - before:+.4f})")

    naive_drift = results["no replay (naive)"][2]
    cls_drift = results["interleaved replay"][2]
    verdict = "replay reduces forgetting" if cls_drift < naive_drift else "NO BENEFIT"
    reduction = (
        (naive_drift - cls_drift) / abs(naive_drift) * 100 if naive_drift else 0.0
    )
    print(f"\nverdict: {verdict} ({reduction:.1f}% less drift)")

    # Pruning behaviour: what does consolidation release?
    system = build_system(epochs=epochs)
    system.bootstrap(train)
    for item in anomaly[:30]:
        system.ingest(item.text)
    for item in synthetic.RULE_TEMPLATES:
        system.remember_rule(item)
    before_n = len(system)
    report = system.sleep(epochs=20)
    print(f"\nconsolidation pass: {before_n} -> {len(system)} memories "
          f"({report.replayed} replayed, {report.pruned_predicted} pruned as learned, "
          f"loss {report.loss_before:.4f} -> {report.loss_after:.4f})")
    evergreen = sum(1 for r in system.records if r.is_evergreen)
    print(f"evergreen surviving: {evergreen}/{len(synthetic.RULE_TEMPLATES)}")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller/faster run")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    n_routine = 400 if args.quick else 2000
    n_anomaly = 30 if args.quick else 60
    epochs = 20 if args.quick else 60

    items = synthetic.generate(n_routine=n_routine, n_anomaly=n_anomaly, seed=SEED)
    routine, anomaly, rules = synthetic.split(items)

    print("=" * 78)
    print("CLS ORGANIZATIONAL MEMORY — SYNTHETIC BENCHMARK")
    print("=" * 78)
    print(f"routine  {len(routine):>5} items from {len(synthetic.ROUTINE_TEMPLATES)} templates")
    print(f"anomaly  {len(anomaly):>5} items from {len(synthetic.ANOMALY_TEMPLATES)} templates")
    print(f"rules    {len(rules):>5} evergreen policies")

    system = experiment_novelty(routine, anomaly, rules, epochs)
    experiment_key_representation(routine, anomaly, epochs)
    experiment_retrieval(system, anomaly)
    experiment_completion(system, anomaly)
    experiment_capacity(routine, anomaly, epochs)
    experiment_forgetting(system, rules)
    experiment_consolidation(routine, anomaly, epochs)

    print(f"\n{'=' * 78}\nBENCHMARK COMPLETE\n{'=' * 78}")


if __name__ == "__main__":
    main()
