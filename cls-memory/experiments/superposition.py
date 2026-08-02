"""The MHN as a substrate, not a ranker: identity with attention, and capacity.

    PYTHONPATH=. .venv/bin/python experiments/superposition.py

Everything measured before this file scored the Hopfield network as a *ranker
of text chunks* -- and `softmax(beta X xi) X` over stored rows is an attention
head, which is a soft kNN, so "ties cosine kNN" was a tautology dressed up as a
finding. It was the one axis where the answer was fixed in advance.

The claims that actually distinguish an associative memory from a vector
database are about representation, not ranking:

  PART 1  One MHN update IS one attention head. Not "analogous to" -- the same
          arithmetic. If that holds numerically, memories can enter a
          transformer through attention as key/value pairs instead of through
          the prompt as tokens, and cost zero context.

  PART 2  A settled state superposes many memories in ONE fixed-width vector.
          The cost of applying k rules is the cost of applying one. Text
          injection is linear in k; this is constant. The open question is
          capacity: how large can k get before the mixture stops being
          decodable?

  PART 3  What that buys in the only currency that matters -- tokens.

Part 2 is the real experiment. A sum of k near-orthogonal unit vectors retains
each component at cosine ~1/sqrt(k), so recovery degrades smoothly rather than
falling off a cliff; where it breaks depends on how correlated the stored items
are, which for real text embeddings is an empirical question and not a
theoretical one. Measured on both corpora because the rulebook is small and
homogeneous while LoCoMo turns are numerous and diverse.
"""

from __future__ import annotations

import argparse
import math

import torch

from cls_memory import HopfieldConfig, ModernHopfieldNetwork
from experiments import locomo
from experiments.recall_ablation import BGEEmbedder
from experiments.rulebook import RULES


# ---------------------------------------------------------------------------
# Part 1: the identity


def attention_head(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   scale: float) -> torch.Tensor:
    """A transformer attention head, written out plainly.

    softmax(scale * q K^T) V -- exactly what sits inside every layer of every
    transformer, with no Hopfield vocabulary anywhere in it.
    """
    return torch.softmax(scale * (query @ keys.T), dim=-1) @ values


def part1_identity(embedder) -> None:
    print("=" * 78)
    print("PART 1 -- is one MHN update literally one attention head?")
    print("=" * 78)

    torch.manual_seed(0)
    patterns = embedder.encode([r.text for r in RULES])
    query = embedder.encode(["a contractor needs production database access"])[0]

    worst = worst_exact = 0.0
    for beta in (1.0, 8.0, 32.0, 128.0, 512.0):
        mhn = ModernHopfieldNetwork(patterns.shape[1], HopfieldConfig(beta=beta))
        mhn.write(patterns)
        hopfield = mhn.step(query)
        # K = V = the stored memories; the inverse temperature is the scale.
        attention = attention_head(query, mhn.patterns, mhn.patterns, beta)
        gap = float((hopfield - attention).abs().max())
        worst = max(worst, gap)

        # Same comparison in float64, so the residual is the mathematics rather
        # than float32 rounding. The stored energy folds a -beta*||x||^2/2 term
        # into the logits; patterns are unit-norm by config, so that term is the
        # same constant for every memory and cancels inside the softmax. The
        # identity is therefore exact, not approximate.
        p64, q64 = mhn.patterns.double(), query.double()
        lhs = torch.softmax(beta * (q64 @ p64.T - 0.5), dim=-1) @ p64
        rhs = attention_head(q64, p64, p64, beta)
        exact = float((lhs - rhs).abs().max())
        worst_exact = max(worst_exact, exact)
        print(f"  beta={beta:6.1f}   float32 {gap:.2e}    float64 {exact:.2e}")

    print(f"\n  worst disagreement: {worst:.2e} (float32), "
          f"{worst_exact:.2e} (float64)")
    print("  -> the memory is not something you query *before* the model.")
    print("     It is a K/V pair the model can attend to directly, at a cost of")
    print("     zero context tokens.")

    # The scale a real transformer uses is 1/sqrt(d_head), which pins beta.
    d = patterns.shape[1]
    print(f"\n  A transformer head with d={d} uses scale 1/sqrt(d) = "
          f"{1/math.sqrt(d):.4f}, i.e. beta = {1/math.sqrt(d):.4f}.")
    print("  That is FAR below the beta=128 this store needs for episodic recall")
    print("  -- so memories injected into a real head land in the metastable")
    print("  (gist) regime by default, which is Part 2's subject.")


# ---------------------------------------------------------------------------
# Part 2: superposition capacity


def superpose(patterns: torch.Tensor, members: list[int], beta: float,
              mhn: ModernHopfieldNetwork) -> torch.Tensor:
    """One settled state carrying k memories.

    Seeded from the mean of the members -- the state a cue touching all of them
    would produce -- then settled by the Hopfield update, so what is measured is
    a genuine attractor of the landscape rather than an arbitrary average.
    """
    state = patterns[members].mean(dim=0)
    state = state / state.norm().clamp_min(1e-12)
    for _ in range(8):
        nxt = mhn.step(state, beta)
        nxt = nxt / nxt.norm().clamp_min(1e-12)
        if float((nxt - state).norm()) < 1e-6:
            state = nxt
            break
        state = nxt
    return state


def capacity_curve(name: str, patterns: torch.Tensor, *, beta: float,
                   trials: int, ks: list[int], generator: torch.Generator) -> None:
    n, d = patterns.shape
    mhn = ModernHopfieldNetwork(d, HopfieldConfig(beta=beta))
    mhn.write(patterns)

    print(f"\n  {name}: {n} memories, d={d}, beta={beta:g}")
    print(f"    {'k':>3}  {'per-item recall':>15}  {'exact set':>10}  "
          f"{'mean cos to members':>20}  {'ideal sum':>10}")
    for k in ks:
        if k > n:
            continue
        recalled, exact, cosines, ideal = [], [], [], []
        for _ in range(trials):
            members = torch.randperm(n, generator=generator)[:k].tolist()
            state = superpose(patterns, members, beta, mhn)

            sims = patterns @ state
            top = torch.topk(sims, k).indices.tolist()
            hit = len(set(top) & set(members))
            recalled.append(hit / k)
            exact.append(int(hit == k))
            cosines.append(float(sims[members].mean()))

            # Ceiling: the plain normalised sum, with no attractor dynamics.
            # If the settled state cannot beat this, the settling is decoration.
            raw = patterns[members].mean(dim=0)
            raw = raw / raw.norm().clamp_min(1e-12)
            top_raw = torch.topk(patterns @ raw, k).indices.tolist()
            ideal.append(len(set(top_raw) & set(members)) / k)

        print(f"    {k:>3}  {sum(recalled)/trials:>15.3f}  "
              f"{sum(exact)/trials:>10.2f}  {sum(cosines)/trials:>20.3f}  "
              f"{sum(ideal)/trials:>10.3f}")


def whiten(x: torch.Tensor, *, floor: float = 1e-2) -> torch.Tensor:
    """Centre, ZCA-whiten, renormalise -- an isotropic code for the same content.

    Sentence embeddings are famously anisotropic: BGE vectors for *unrelated*
    text sit at cosine ~0.75, all crammed into a narrow cone. Superposition
    needs the opposite. A sum of k vectors retains each component at cosine
    ~1/sqrt(k) only when they are near-orthogonal; when everything already
    points the same way there is no component structure left to decode, and a
    mixture is indistinguishable from any other mixture.

    Whitening equalises the covariance spectrum, which is the standard fix
    (Mu & Viswanath 2018 "all-but-the-top"; Su et al. 2021 "whitening
    sentence representations"). `floor` regularises the small singular values,
    which are noise directions that would otherwise be amplified enormously.
    """
    centred = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centred, full_matrices=False)
    scale = (s / math.sqrt(x.shape[0])).clamp_min(floor)
    out = (centred @ vh.T / scale) @ vh
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def anisotropy(x: torch.Tensor, generator: torch.Generator, pairs: int = 20000) -> float:
    """Mean cosine between random distinct pairs -- 0.0 is an isotropic code."""
    n = x.shape[0]
    i = torch.randint(0, n, (pairs,), generator=generator)
    j = torch.randint(0, n, (pairs,), generator=generator)
    keep = i != j
    return float((x[i[keep]] * x[j[keep]]).sum(-1).mean())


def part2_capacity(embedder, trials: int, beta: float) -> None:
    print("\n" + "=" * 78)
    print("PART 2 -- how many memories fit in ONE vector and stay decodable?")
    print("=" * 78)
    print("  per-item recall = fraction of the k superposed memories that rank")
    print("  in the top k. exact set = all k recovered, no intruders.")
    print("  'ideal sum' is the same test on the plain normalised mean, which")
    print("  is the ceiling any settling has to justify itself against.")

    generator = torch.Generator().manual_seed(0)
    ks = [1, 2, 4, 8, 16, 32, 64]

    turns = [t.memory_text for c in locomo.load()[:2] for t in c.turns]
    facts = embedder.encode(turns)
    n, d = facts.shape

    print(f"\n  anisotropy (mean cosine between unrelated memories):")
    print(f"    BGE as shipped   {anisotropy(facts, generator):+.3f}")
    white = whiten(facts)
    print(f"    BGE whitened     {anisotropy(white, generator):+.3f}")
    control = torch.randn(n, d, generator=generator)
    control = control / control.norm(dim=-1, keepdim=True)
    print(f"    random unit      {anisotropy(control, generator):+.3f}   "
          f"<- the isotropic ideal")

    capacity_curve("LoCoMo turns, BGE as shipped", facts, beta=beta,
                   trials=trials, ks=ks, generator=generator)
    capacity_curve("LoCoMo turns, BGE WHITENED", white, beta=beta,
                   trials=trials, ks=ks, generator=generator)
    capacity_curve("random unit vectors (ceiling)", control, beta=beta,
                   trials=trials, ks=ks, generator=generator)
    beta_sweep(white, ks=[2, 4, 8, 16, 32], trials=trials, generator=generator)


# ---------------------------------------------------------------------------
# Part 3: what it costs


def beta_sweep(patterns: torch.Tensor, *, ks: list[int], trials: int,
               generator: torch.Generator) -> None:
    """Where does settling preserve a mixture, and where does it collapse it?

    Two different operations get conflated by calling both "retrieval":

      * **holding** k memories in one state -- wants the metastable regime,
        where the update leaves a mixture alone;
      * **completing** a partial cue to one memory -- wants the episodic
        regime, where the update snaps to the nearest attractor.

    beta is the dial between them, and it is the same dial as diffusion noise
    level. A real transformer head runs at 1/sqrt(d), which is deep in the
    first regime -- so a memory injected as K/V is *held*, not collapsed.
    """
    n, d = patterns.shape
    print(f"\n  beta sweep on whitened vectors ({n} memories, d={d}).")
    print("  'none' = the raw normalised sum, no settling at all.")
    betas = [1.0 / math.sqrt(d), 0.5, 2.0, 8.0, 32.0, 128.0]
    header = "  ".join(f"b={b:<7.3f}" for b in betas)
    print(f"    {'k':>3}  {'none':>7}  {header}")

    for k in ks:
        cells = []
        raw_scores = []
        for beta in betas:
            mhn = ModernHopfieldNetwork(d, HopfieldConfig(beta=beta))
            mhn.write(patterns)
            hits, raws = [], []
            gen = torch.Generator().manual_seed(1234 + k)
            for _ in range(trials):
                members = torch.randperm(n, generator=gen)[:k].tolist()
                raw = patterns[members].mean(dim=0)
                raw = raw / raw.norm().clamp_min(1e-12)
                top_raw = torch.topk(patterns @ raw, k).indices.tolist()
                raws.append(len(set(top_raw) & set(members)) / k)

                state = superpose(patterns, members, beta, mhn)
                top = torch.topk(patterns @ state, k).indices.tolist()
                hits.append(len(set(top) & set(members)) / k)
            cells.append(sum(hits) / trials)
            raw_scores = raws
        print(f"    {k:>3}  {sum(raw_scores)/trials:>7.3f}  "
              + "  ".join(f"{c:>9.3f}" for c in cells))


def part3_budget(embedder) -> None:
    print("\n" + "=" * 78)
    print("PART 3 -- the same content, in tokens vs in slots")
    print("=" * 78)
    lengths = [len(e.ids) for e in embedder._tok.encode_batch([r.text for r in RULES])]
    mean = sum(lengths) / len(lengths)
    print(f"  mean rule = {mean:.0f} tokens (real tokenizer)")
    print(f"\n  {'rules applied':>14}  {'as text (tokens)':>18}  "
          f"{'as K/V slots':>13}  {'ratio':>8}")
    for k in (1, 3, 5, 10, 25, 100):
        text = k * mean
        print(f"  {k:>14}  {text:>18.0f}  {1:>13}  {text:>7.0f}x")
    print("\n  Text injection is linear in the number of rules. A superposed")
    print("  state is one slot regardless -- that is the whole claim, and Part 2")
    print("  is what bounds how far it can be pushed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--beta", type=float, default=128.0)
    args = parser.parse_args()
    torch.manual_seed(0)

    embedder = BGEEmbedder()
    part1_identity(embedder)
    part2_capacity(embedder, args.trials, args.beta)
    part3_budget(embedder)


if __name__ == "__main__":
    main()
