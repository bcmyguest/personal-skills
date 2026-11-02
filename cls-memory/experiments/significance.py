"""Statistical significance of the consolidation claims.

    PYTHONPATH=. .venv/bin/python experiments/significance.py [--seeds N]

The benchmark reports single-seed point estimates. Two of the claims resting on
them are the kind that motivated reasoning attaches to, so they get a proper
repeated-measures test here:

  H1  interleaved replay reduces schema drift versus naive training
  H2  raising `episodic_ratio` from 0 to 0.125 costs drift protection
      (reported as 55.3% -> 54.5%, i.e. "0.8 points, within noise")
  H3  raising `episodic_ratio` from 0 to 0.125 increases pruning
      (reported as 0 -> 6 of 30 memories released)

**Paired design.** Every condition in a given seed shares one bootstrapped
cortex, deep-copied before each arm. Cortex-to-cortex variance is large and
dominates the effects under test; pairing removes it. An earlier unpaired sweep
in this project produced a nonsense ordering (more training budget apparently
releasing *fewer* memories) purely from that confound.

**Test.** Exact paired permutation (sign-flipping) on the per-seed differences.
With n seeds there are 2^n sign assignments, all enumerated -- so the p-value is
exact, assumes nothing about normality, and needs no scipy (unavailable here).
Reported alongside the mean difference and a bootstrap 95% CI, because with a
handful of seeds the effect size and its spread matter more than the p-value.

Note H2 is a test for *harm*: failing to reject is not proof the cost is zero,
only that this many seeds cannot resolve it. The CI is the honest summary, and
it is reported either way.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import random

import torch

from cls_memory import (
    ConsolidationConfig,
    CortexConfig,
    DecayConfig,
    HopfieldConfig,
    KeyConfig,
    MemorySystemConfig,
    NoveltyConfig,
    OrganizationalMemory,
)
from experiments import synthetic
from experiments.synthetic import Label

# Smaller than the headline benchmark so that many seeds are affordable; the
# effects under test are properties of the consolidation loop, not of corpus
# size, and H1/H3 reproduce at the benchmark's scale.
N_ROUTINE = 600
N_ANOMALY = 30
EPOCHS = 40
CONSOLIDATE_EPOCHS = 40


# ----------------------------------------------------------------- statistics


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def exact_sign_permutation_p(diffs: list[float]) -> float:
    """Two-sided exact paired permutation test.

    Under the null "the pairing carries no signal", each difference's sign is
    exchangeable. Enumerate all 2^n sign assignments and count how many produce
    a mean at least as extreme as observed.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    if n > 20:  # 2^20 = 1e6, still fine; guard against silly inputs
        raise ValueError("too many seeds for exact enumeration")
    observed = abs(mean(diffs))
    extreme = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        flipped = [s * d for s, d in zip(signs, diffs)]
        if abs(mean(flipped)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def bootstrap_ci(
    xs: list[float], *, iterations: int = 20_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(iterations):
        means.append(mean([xs[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    return lo, hi


def cohens_dz(diffs: list[float]) -> float:
    """Paired effect size: mean difference over its own standard deviation."""
    s = stdev(diffs)
    return mean(diffs) / s if s > 0 else float("inf")


def report(name: str, diffs: list[float], unit: str = "") -> None:
    m = mean(diffs)
    lo, hi = bootstrap_ci(diffs)
    p = exact_sign_permutation_p(diffs)
    print(f"  {name}")
    print(f"    mean difference {m:+.4f}{unit}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    sd {stdev(diffs):.4f}   dz {cohens_dz(diffs):+.2f}   "
          f"exact permutation p = {p:.4f}   n = {len(diffs)}")
    verdict = "significant at 0.05" if p < 0.05 else "NOT resolvable at this n"
    print(f"    -> {verdict}")


# ------------------------------------------------------------------ the trial


def build(seed: int, episodic_ratio: float) -> OrganizationalMemory:
    return OrganizationalMemory(
        MemorySystemConfig(
            cortex=CortexConfig(
                input_dim=256,
                hidden_dims=(192, 96),
                latent_dim=32,
                epochs=EPOCHS,
                batch_size=64,
                learning_rate=1e-3,
                kl_weight=0.01,
            ),
            novelty=NoveltyConfig(quantile=0.95, warmup=32, window=4096),
            hopfield=HopfieldConfig(beta=32.0),
            key=KeyConfig(expansion_dim=2048, sparsity_k=256),
            decay=DecayConfig(half_life_days=30.0),
            consolidation=ConsolidationConfig(
                replay_batch=256, replay_steps=8, episodic_ratio=episodic_ratio
            ),
            seed=seed,
        )
    )


def run_seed(seed: int) -> dict:
    """One paired trial: naive vs replay@0.0 vs replay@0.125, shared cortex."""
    torch.manual_seed(seed)
    random.seed(seed)

    items = synthetic.generate(n_routine=N_ROUTINE, n_anomaly=N_ANOMALY, seed=seed)
    routine, anomaly, _ = synthetic.split(items)
    train = [i.text for i in routine[: int(len(routine) * 0.8)]]
    old = [i.text for i in routine[:200]]
    intruder_texts = [i.text for i in anomaly] * 4

    system = build(seed, episodic_ratio=0.0)
    system.bootstrap(train)
    for item in anomaly:
        system.ingest(item.text)

    old_x = system.embedder.encode(old)
    intruder = system.embedder.encode(intruder_texts)
    before = float(system.cortex.surprise(old_x).mean())
    baseline_cortex = copy.deepcopy(system.cortex.state_dict())

    def drift_after(fn) -> float:
        system.cortex.load_state_dict(copy.deepcopy(baseline_cortex))
        fn()
        return float(system.cortex.surprise(old_x).mean()) - before

    # arm 1: naive -- train on the new data alone, no replay
    d_naive = drift_after(
        lambda: system.cortex.fit(intruder, epochs=CONSOLIDATE_EPOCHS)
    )

    # arms 2 and 3: interleaved replay at two episodic ratios
    system.consolidation.config.prune_predicted = False
    results = {}
    for ratio in (0.0, 0.125):
        system.consolidation.config.episodic_ratio = ratio
        results[ratio] = drift_after(
            lambda: system.consolidation.consolidate(
                intruder, epochs=CONSOLIDATE_EPOCHS
            )
        )

    # pruning: rerun consolidation with pruning on, counting releases
    pruned = {}
    for ratio in (0.0, 0.125):
        system.cortex.load_state_dict(copy.deepcopy(baseline_cortex))
        system.consolidation.config.episodic_ratio = ratio
        system.consolidation.config.prune_predicted = False
        system.consolidation.consolidate(intruder, epochs=CONSOLIDATE_EPOCHS)
        pruned[ratio] = len(system.consolidation.prune_predicted())
        # restore the store for the next arm
        for item in anomaly:
            if not any(r.text == item.text for r in system.records):
                system.ingest(item.text)

    return {
        "seed": seed,
        "before": before,
        "drift_naive": d_naive,
        "drift_r000": results[0.0],
        "drift_r125": results[0.125],
        "pruned_r000": float(pruned[0.0]),
        "pruned_r125": float(pruned[0.125]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    print("=" * 78)
    print("SIGNIFICANCE OF THE CONSOLIDATION CLAIMS")
    print("=" * 78)
    print(f"paired design, {args.seeds} seeds, exact sign-permutation tests")
    print(f"corpus {N_ROUTINE} routine / {N_ANOMALY} anomaly, "
          f"{EPOCHS} bootstrap epochs, {CONSOLIDATE_EPOCHS} consolidation epochs\n")

    rows = []
    for seed in range(args.seeds):
        row = run_seed(seed)
        rows.append(row)
        print(f"  seed {seed}: naive {row['drift_naive']:+.4f}  "
              f"r=0 {row['drift_r000']:+.4f}  r=.125 {row['drift_r125']:+.4f}  "
              f"pruned {int(row['pruned_r000'])}/{int(row['pruned_r125'])}")

    def col(k: str) -> list[float]:
        return [r[k] for r in rows]

    def reduction(a: str, b: str) -> list[float]:
        """Per-seed drift reduction %, replay arm b against naive arm a."""
        return [
            100.0 * (r[a] - r[b]) / r[a] if r[a] != 0 else 0.0 for r in rows
        ]

    print("\n" + "=" * 78)
    print("H1  interleaved replay reduces drift versus naive training")
    print("=" * 78)
    report("drift(naive) - drift(replay, r=0.125)",
           [a - b for a, b in zip(col("drift_naive"), col("drift_r125"))])
    red = reduction("drift_naive", "drift_r125")
    lo, hi = bootstrap_ci(red)
    print(f"    drift reduction {mean(red):.1f}%  95% CI [{lo:.1f}%, {hi:.1f}%]  "
          f"sd {stdev(red):.1f}")

    print("\n" + "=" * 78)
    print("H2  episodic_ratio 0 -> 0.125 costs drift protection")
    print("=" * 78)
    print("  (positive difference = the fix drifts MORE, i.e. a real cost)")
    report("drift(r=0.125) - drift(r=0)",
           [b - a for a, b in zip(col("drift_r000"), col("drift_r125"))])
    r000 = reduction("drift_naive", "drift_r000")
    r125 = reduction("drift_naive", "drift_r125")
    print(f"    reduction at r=0     {mean(r000):.1f}%  sd {stdev(r000):.1f}")
    print(f"    reduction at r=0.125 {mean(r125):.1f}%  sd {stdev(r125):.1f}")
    report("reduction%(r=0) - reduction%(r=0.125)",
           [a - b for a, b in zip(r000, r125)], unit=" pp")

    print("\n" + "=" * 78)
    print("H3  episodic_ratio 0 -> 0.125 increases pruning")
    print("=" * 78)
    print(f"    memories released, r=0     mean {mean(col('pruned_r000')):.2f}  "
          f"sd {stdev(col('pruned_r000')):.2f}  "
          f"values {[int(x) for x in col('pruned_r000')]}")
    print(f"    memories released, r=0.125 mean {mean(col('pruned_r125')):.2f}  "
          f"sd {stdev(col('pruned_r125')):.2f}  "
          f"values {[int(x) for x in col('pruned_r125')]}")
    report("pruned(r=0.125) - pruned(r=0)",
           [b - a for a, b in zip(col("pruned_r000"), col("pruned_r125"))],
           unit=" memories")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
