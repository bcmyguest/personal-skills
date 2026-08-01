"""End-to-end walkthrough of the CLS organizational memory.

    python examples/demo.py

Runs offline: the default HashingEmbedder needs no model download.
"""

from __future__ import annotations

import torch

from cls_memory import MemorySystemConfig, OrganizationalMemory, Persistence
from cls_memory.config import CortexConfig, HopfieldConfig, NoveltyConfig

# Eighteen months of unremarkable operational chatter: the organisation's schema.
ROUTINE = [
    "deployment of the billing service completed without incident",
    "deployment of the checkout service completed without incident",
    "deployment of the search service completed without incident",
    "deployment of the notifications service completed without incident",
    "nightly batch job finished successfully in the eu-west region",
    "nightly batch job finished successfully in the us-east region",
    "nightly batch job finished successfully in the ap-south region",
    "weekly report generated and delivered to the finance team",
    "weekly report generated and delivered to the operations team",
    "customer support ticket resolved within the standard response window",
    "customer support ticket closed within the standard response window",
    "routine database backup completed for the primary cluster",
    "routine database backup completed for the replica cluster",
    "scheduled maintenance window opened for the staging environment",
    "scheduled maintenance window closed for the staging environment",
    "access review completed for the engineering team with no exceptions",
]

RULES = [
    "all refunds above 500 dollars require director approval",
    "production deploys are frozen during the december code freeze",
    "customer data may not leave the eu-west region under any circumstances",
]

INCIDENTS = [
    "the shard rebalancer corrupted the orders index during failover",
    "a contractor deleted the staging kubernetes namespace by hand",
    "the payment provider rotated credentials without notice and checkout failed",
    "an unfamiliar vendor invoice arrived from a shell company in belize",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> None:
    torch.manual_seed(0)

    config = MemorySystemConfig(
        cortex=CortexConfig(
            input_dim=128,
            hidden_dims=(96, 64),
            latent_dim=24,
            epochs=400,
            batch_size=8,
            learning_rate=3e-3,
        ),
        novelty=NoveltyConfig(quantile=0.8, warmup=4, window=256),
        hopfield=HopfieldConfig(beta=32.0),
        seed=0,
    )
    system = OrganizationalMemory(config)

    # -------------------------------------------------------------- bootstrap
    rule("1. Train the slow neocortex on the routine corpus")
    report = system.bootstrap(ROUTINE)
    print(f"corpus            {report.corpus_size} documents")
    print(f"final VAE loss    {report.final_loss:.4f}")
    print(f"novelty threshold {report.novelty_threshold:.4f}  "
          f"(the {config.novelty.quantile:.0%} quantile of corpus surprise)")

    # ------------------------------------------------------- synaptic gating
    rule("2. Synaptic ingestion: surprise decides what the hippocampus keeps")
    print(f"{'surprise':>9}  {'action':<12}  text")
    print("-" * 74)
    for text in ROUTINE[:3] + INCIDENTS:
        result = system.log_event(text)
        print(f"{result.novelty:9.4f}  {result.action.value:<12}  {text[:44]}")

    print("\nBusiness rules bypass the gate entirely:")
    for text in RULES:
        result = system.remember_rule(text)
        print(f"{result.novelty:9.4f}  {result.action.value:<12}  {text[:44]}")

    print("\nNote the first line: a quantile-calibrated gate admits (1 - q) of its")
    print("own calibration distribution by construction, so the worst-reconstructed")
    print("routine item still reads as novel. Widen the corpus or raise `quantile`")
    print("to tighten this; it is a property of the threshold, not a bug.")

    stats = system.stats()
    print(f"\nhippocampus: {stats['total']} memories "
          f"({stats['evergreen']} evergreen, {stats['temporal']} episodic) "
          f"from {len(ROUTINE[:3]) + len(INCIDENTS) + len(RULES)} submissions")

    # ---------------------------------------------------------------- recall
    rule("3. Pattern completion from a partial cue")
    for query in ["shard rebalancer orders index", "refunds director approval"]:
        result = system.recall(query, top_k=3, reinforce=False)
        print(f"\ncue: {query!r}")
        print(f"  settled in {result.trace.iterations} iteration(s), "
              f"energy {result.trace.energy:+.4f}, confidence {result.confidence:.3f}")
        for hit in result.results:
            print(f"    {hit.weight:6.3f}  [{hit.record.persistence.value:<9}] {hit.record.text[:46]}")

    rule("4. Strict pattern completion: 30% of the key, the rest filled in")
    target = INCIDENTS[0]
    partial, mask = system.retrieval.occlude(target, keep_fraction=0.3, seed=1)
    result = system.complete(partial, mask, top_k=3)
    print(f"target      {target}")
    print(f"cue         {int(mask.sum())}/{mask.numel()} key coordinates known")
    print(f"recovered   {result.top.record.text}")
    print(f"correct     {result.top.record.text == target}")

    rule("5. Confabulation check: energy at the cue, not the settled state")
    for query in ["shard rebalancer orders index", "quantum chromodynamics lattice gauge"]:
        basin = system.recall(query, reinforce=False).basin
        print(f"{query[:44]:<46} depth={basin.depth:7.4f}  "
              f"log p={basin.log_density:8.3f}  confabulation={basin.is_confabulation}")

    # ------------------------------------------------------- gist vs episode
    rule("6. Beta as a level-of-abstraction dial")
    for label, res in (
        ("episodic (high beta)", system.recall("deployment service", reinforce=False)),
        ("gist     (low beta) ", system.gist("deployment service")),
    ):
        top = res.top
        print(f"{label}  beta={res.beta:7.2f}  top weight={top.weight:.3f}  "
              f"mixture={res.is_gist}")

    # ------------------------------------------------------------------ time
    rule("7. Evergreen vs temporal under the 30-day forgetting curve")
    print(f"{'age (days)':>10}  {'evergreen':>10}  {'episodic':>10}")
    print("-" * 34)
    ever = next(r for r in system.records if r.is_evergreen)
    epis = next(r for r in system.records if not r.is_evergreen)
    for days in (0, 30, 60, 90, 180, 365):
        e = ever.salience(half_life_days=30.0)
        t = pow(2.0, -days / 30.0) * epis.strength
        print(f"{days:10d}  {e:10.4f}  {t:10.4f}")

    before = len(system)
    for record in system.records:
        record.age_by(200)
    sweep = system.sweep()
    print(f"\nafter 200 days: {before} -> {len(system)} memories "
          f"({sweep.pruned_decayed} episodic pruned, all evergreen retained)")
    for record in system.records:
        print(f"    [{record.persistence.value:<9}] {record.text[:52]}")

    # --------------------------------------------------------- consolidation
    rule("8. Consolidation: replay + interleaved training")
    system.log_event("the shard rebalancer corrupted the orders index again")
    consolidated = system.sleep(epochs=20)
    print(f"replayed samples  {consolidated.replayed}")
    print(f"loss              {consolidated.loss_before:.4f} -> {consolidated.loss_after:.4f}")
    print(f"pruned as learned {consolidated.pruned_predicted}")
    print(f"surviving         {len(system)} memories")

    print("\nEvergreen rules survive every mechanism -- decay, sweep, and")
    print("consolidation pruning:")
    for record in system.store.by_persistence(Persistence.EVERGREEN):
        print(f"    {record.text}")


if __name__ == "__main__":
    main()
