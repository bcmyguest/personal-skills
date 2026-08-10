"""Do the separation mechanisms actually help retrieval? (review ticket 08)

    PYTHONPATH=. uv run python -u experiments/separation_check.py --corpus locomo

Four arms at one operating point (beta=128, LSA-1024): baseline, whitening on,
dentate-gyrus pattern separation on, and both. Each is compared against the
baseline with the **exact paired McNemar test**, because at LoCoMo's n=494 a
difference below ~0.04 hit@1 is not resolvable and this project has previously
published sub-threshold differences as findings (HANDOFF §3.1d).

Why the ticket's premise needed revising
----------------------------------------
Ticket 08 was written on the assumption that two mechanisms here raise
separation: whitening and dentate-gyrus pattern separation. Measured, only the
first does. `anisotropy` is the mean cosine between unrelated pairs, so LOWER
is better separated, and DG moves it the wrong way -- 0.041 -> 0.134 on LoCoMo
and 0.048 -> 0.139 on QMSum (RESULTS.md IV.3). The mechanism named "pattern
separation" is, on these embedders, a pattern *de*-separation.

So DG is carried here as a **negative control** rather than as a second lever:
if the Hopfield capacity story is right that separation is what matters, the
arm that lowers separation should not beat the arm that raises it. That is a
falsifiable prediction, which the original framing did not offer.
"""

from __future__ import annotations

import argparse
import time

import torch

from cls_memory import HippocampalKey
from experiments import locomo, qmsum
from experiments.metrics import mcnemar_exact
from experiments.recall_check import KS, SEED, evaluate

ARMS = [
    ("baseline (LSA-1024, key=embedding)", HippocampalKey.EMBEDDING, False),
    ("whitened  (raises separation)", HippocampalKey.EMBEDDING, True),
    ("DG key    (LOWERS separation)", HippocampalKey.SEPARATED, False),
    ("both", HippocampalKey.SEPARATED, True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("locomo", "qmsum"), default="locomo")
    parser.add_argument("--locomo", type=int, default=3)
    parser.add_argument("--qmsum", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    if args.corpus == "locomo":
        convs = locomo.load()[: args.locomo]
    else:
        convs = qmsum.load(max_meetings=args.qmsum)
    turns = sum(len(c.turns) for c in convs)
    print(f"{args.corpus}: {len(convs)} conversations, {turns} turns\n")

    results = []
    for label, key_mode, whiten in ARMS:
        t0 = time.time()
        r = evaluate(convs, dim=1024, key_mode=key_mode, beta=128.0, whiten=whiten)
        results.append((label, r))
        print(f"  {label:<36} " + "  ".join(f"{r[k]:6.3f}" for k in KS)
              + f"   aniso_emb {r['aniso_emb']:+.3f}  aniso_key {r['aniso_key']:+.3f}"
              + f"   ({r['asked']} q, {time.time() - t0:.0f}s)")

    base_label, base = results[0]
    # Both ends, not just @1: whitening is measured to help the top of the
    # ranking and hurt its depth, and a defaults decision that only looks at
    # hit@1 would miss half of that trade.
    for k in (1, 10):
        print(f"\n  exact paired McNemar vs baseline, hit@{k} (n={base['asked']}):")
        print(f"    {'arm':<36} {'delta':>8} {'b01':>5} {'b10':>5} {'p':>9}  verdict")
        for label, r in results[1:]:
            m = mcnemar_exact(base["per_question"][k], r["per_question"][k])
            delta = r[k] - base[k]
            resolvable = m["p"] < 0.05
            verdict = "RESOLVED" if resolvable else "tie (not resolvable)"
            print(f"    {label:<36} {delta:+8.3f} {m['b01']:5d} {m['b10']:5d} "
                  f"{m['p']:9.4f}  {verdict}")

    print("\n  separation actually achieved (lower anisotropy = better separated):")
    for label, r in results:
        print(f"    {label:<36} aniso_key {r['aniso_key']:+.3f}")


if __name__ == "__main__":
    main()
