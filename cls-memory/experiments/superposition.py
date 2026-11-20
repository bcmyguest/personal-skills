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

  PART 4  Whitening as a RANKER, not a substrate -- RESULTS.md V.5/VI.3.
          Helps hit@1 (inside this project's ~0.04 resolution limit) and
          costs depth. Added here (review ticket 06) because those sections
          previously had no reproducible source in this file at all.

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
from cls_memory.config import WhiteningConfig
from cls_memory.whitening import Whitener, WhitenedEmbedder, anisotropy
from experiments import locomo
from experiments.recall_ablation import BGEEmbedder, dense_knn_ranker, score_ranking
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


def settle_superposed(members: list[int], beta: float,
                      mhn: ModernHopfieldNetwork) -> torch.Tensor:
    """Hold k memories in one vector, then SETTLE it -- the V.3 operation.

    Both halves are the library's: `ModernHopfieldNetwork.superpose` builds the
    normalised sum (the `hold`), and `step` runs the Hopfield update. This
    function is only the composition of the two, kept here because "what does
    settling do to a superposition" is the question V.3 asks and neither
    library method answers it alone.

    Previously this file carried its own copy of both halves (ticket 06), so
    Parts V.5/VI.2/VI.3 claimed to be measured "through the library classes"
    while actually running private duplicates.
    """
    state = mhn.superpose(members)
    for _ in range(8):
        nxt = mhn.step(state, beta)
        nxt = nxt / nxt.norm().clamp_min(1e-12)
        if float((nxt - state).norm()) < 1e-6:
            return nxt
        state = nxt
    return state


def capacity_curve(name: str, patterns: torch.Tensor, *,
                   trials: int, ks: list[int], generator: torch.Generator) -> None:
    """Per-item recall of the HOLD operation: k memories summed into one
    normalised vector, decoded by cosine (== attention-logit order, since
    scaling by a positive beta never changes a ranking) against the store.

    Deliberately **not settled**. RESULTS.md V.3 / `beta_sweep` below show that
    running the Hopfield update on a superposed state collapses it onto a
    single attractor at every beta tried -- that is a different operation
    (`retrieve`/`complete`), not this one (`superpose`/`hold`). An earlier
    version of this function settled the state before decoding it, which is
    exactly the V.3 mistake applied to V.2's own measurement: it silently
    stopped reproducing the published capacity table (1.000 at k=4) and instead
    reproduced the collapsed numbers V.3 already reports (0.253 at k=4).
    Capacity is a property of the code being superposed, not of beta.
    """
    n, d = patterns.shape
    # The hold and the decode are both the library's (ticket 06): `superpose`
    # builds the normalised sum, `decode` ranks by the attention LOGITS the
    # Hopfield head computes. For unit-norm patterns under a uniform prior that
    # ranking is identical to cosine against the store -- which is why this
    # reproduces the previously published table rather than moving it.
    mhn = ModernHopfieldNetwork(d, HopfieldConfig())
    mhn.write(patterns)
    print(f"\n  {name}: {n} memories, d={d}")
    print(f"    {'k':>3}  {'per-item recall':>15}  {'exact set':>10}  "
          f"{'mean cos to members':>20}")
    for k in ks:
        if k > n:
            continue
        recalled, exact, cosines = [], [], []
        for _ in range(trials):
            members = torch.randperm(n, generator=generator)[:k].tolist()
            state = mhn.superpose(members)

            top = mhn.decode(state, k).tolist()
            hit = len(set(top) & set(members))
            recalled.append(hit / k)
            exact.append(int(hit == k))
            cosines.append(float((patterns[members] @ state).mean()))

        print(f"    {k:>3}  {sum(recalled)/trials:>15.3f}  "
              f"{sum(exact)/trials:>10.2f}  {sum(cosines)/trials:>20.3f}")


def whiten(x: torch.Tensor, *, fit_on: torch.Tensor | None = None,
           floor: float = 1e-2) -> torch.Tensor:
    """Centre, ZCA-whiten, renormalise -- via `cls_memory.whitening.Whitener`.

    A thin adapter, not an implementation: it exists so the call sites below
    can stay one-liners while the actual transform is the library's fitted
    `Whitener` (ticket 06 -- this file used to carry its own SVD copy, so the
    published claim that V.5/VI.2/VI.3 were re-measured "through the library
    classes" was not reproducible from the code on the branch).

    `fit_on` decouples the fit from the transform: the mean/rotation/scale are
    estimated on `fit_on` and then applied to `x`. Left as `None` the fit is
    transductive -- fit on `x`, applied to `x` -- which flatters the result;
    RESULTS.md V.2 measures how much.
    """
    basis = x if fit_on is None else fit_on
    whitener = Whitener(x.shape[1], WhiteningConfig(floor=floor)).fit(basis)
    return whitener.transform(x)


def fit_size_check(facts: torch.Tensor, fit_pool: torch.Tensor,
                   generator: torch.Generator, *, trials: int) -> None:
    """How many held-out rows does the whitener actually need?

    `Whitener.fit` warns below `n < d` samples (rank-deficient covariance).
    This sweeps the *size* of the disjoint fit corpus from well below that
    threshold to the full pool and reports anisotropy and k=8/k=32 per-item
    recall on `facts` at each size, so the warning threshold can be checked
    against where the held-out measurement actually stops moving rather than
    assumed correct.
    """
    d = facts.shape[1]
    pool_n = fit_pool.shape[0]
    sizes = sorted(set(
        s for s in (64, 128, 256, d // 2, d - 1, d, d + 1, 512, 768, 1536, 3072, pool_n)
        if 2 <= s <= pool_n
    ))
    print(f"\n  fit-size check: Whitener.fit warns below n={d} samples "
          f"(rank-deficient covariance). Disjoint pool has {pool_n} rows.")
    print(f"    {'n_fit':>7}  {'< d warns?':>10}  {'anisotropy':>11}  "
          f"{'recall@k=8':>11}  {'recall@k=32':>12}")
    for n_fit in sizes:
        sub = fit_pool[:n_fit]
        white = whiten(facts, fit_on=sub)
        aniso = anisotropy(white, generator=generator)
        recalls = {}
        for k in (8, 32):
            recalled = []
            for _ in range(trials):
                members = torch.randperm(facts.shape[0], generator=generator)[:k].tolist()
                state = white[members].mean(dim=0)
                state = state / state.norm().clamp_min(1e-12)
                top = torch.topk(white @ state, k).indices.tolist()
                recalled.append(len(set(top) & set(members)) / k)
            recalls[k] = sum(recalled) / trials
        warns = "yes" if n_fit < d else "no"
        print(f"    {n_fit:>7}  {warns:>10}  {aniso:>+11.3f}  "
              f"{recalls[8]:>11.3f}  {recalls[32]:>12.3f}")

    # The sweep above holds domain fixed (fit corpus = 8 *other* conversations)
    # and varies size. It does not move much past a few hundred rows -- which
    # raises the question the size sweep cannot answer on its own: is the
    # residual anisotropy a SIZE problem (not enough disjoint rows) or a
    # DOMAIN problem (disjoint rows drawn from different conversations, with
    # different topics/speakers, don't share the scored corpus's local
    # geometry)? This holds size roughly fixed and varies domain instead: fit
    # on a disjoint half of the SAME two conversations the capacity table
    # scores, rather than the other eight.
    n_facts = facts.shape[0]
    perm = torch.randperm(n_facts, generator=generator)
    half = n_facts // 2
    fit_half, score_half = facts[perm[:half]], facts[perm[half:]]
    white_same_domain = whiten(score_half, fit_on=fit_half)
    aniso_same_domain = anisotropy(white_same_domain, generator=generator)
    print(f"\n  same-domain control: fit on a disjoint half of the SAME two "
          f"conversations (n={half}, well under the {pool_n}-row cross-"
          f"conversation pool above), scored on the other half (n={n_facts - half}):")
    print(f"    anisotropy {aniso_same_domain:+.3f}")
    print("    -> if this is close to the in-sample fit's anisotropy and far "
          "below the cross-conversation held-out fit's, the gap above is "
          "domain shift, not sample size, and the n<d warning (which only "
          "guards rank-deficiency) cannot detect it.")


def part2_capacity(embedder, trials: int) -> None:
    print("\n" + "=" * 78)
    print("PART 2 -- how many memories fit in ONE vector and stay decodable?")
    print("=" * 78)
    print("  per-item recall = fraction of the k superposed memories that rank")
    print("  in the top k. exact set = all k recovered, no intruders.")
    print("  'hold' is the plain normalised sum, no Hopfield settling -- see")
    print("  capacity_curve's docstring for why settling must not be used here.")

    generator = torch.Generator().manual_seed(0)
    ks = [1, 2, 4, 8, 16, 32, 64]

    conversations = locomo.load()
    scored_turns = [t.memory_text for c in conversations[:2] for t in c.turns]
    fit_turns = [t.memory_text for c in conversations[2:] for t in c.turns]
    facts = embedder.encode(scored_turns)
    fit_pool = embedder.encode(fit_turns)
    n, d = facts.shape

    print("\n  anisotropy (mean cosine between unrelated memories):")
    print(f"    BGE as shipped                          {anisotropy(facts, generator=generator):+.3f}")
    white_in_sample = whiten(facts)
    print(f"    BGE whitened, IN-SAMPLE fit (n={n}, transductive)  "
          f"{anisotropy(white_in_sample, generator=generator):+.3f}")
    white_held_out = whiten(facts, fit_on=fit_pool)
    print(f"    BGE whitened, HELD-OUT fit (n={fit_pool.shape[0]}, disjoint)     "
          f"{anisotropy(white_held_out, generator=generator):+.3f}")
    control = torch.randn(n, d, generator=generator)
    control = control / control.norm(dim=-1, keepdim=True)
    print(f"    random unit                             {anisotropy(control, generator=generator):+.3f}   "
          f"<- the isotropic ideal")

    capacity_curve("LoCoMo turns, BGE as shipped", facts,
                   trials=trials, ks=ks, generator=generator)
    capacity_curve("LoCoMo turns, BGE WHITENED (in-sample fit -- transductive, "
                   "flatters the result)", white_in_sample,
                   trials=trials, ks=ks, generator=generator)
    capacity_curve("LoCoMo turns, BGE WHITENED (held-out fit -- the honest number)",
                   white_held_out, trials=trials, ks=ks, generator=generator)
    capacity_curve("random unit vectors (ceiling)", control,
                   trials=trials, ks=ks, generator=generator)

    fit_size_check(facts, fit_pool, generator, trials=max(20, trials // 4))

    # V.3 (owned by a different section of RESULTS.md, unaffected by this
    # ticket's fit-scope fix) was published against the in-sample-whitened
    # vectors -- keep feeding it the same basis so it stays reproducible.
    beta_sweep(white_in_sample, ks=[2, 4, 8, 16, 32], trials=trials, generator=generator)


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

                state = settle_superposed(members, beta, mhn)
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


# ---------------------------------------------------------------------------
# Part 4: whitening as a RANKER (RESULTS.md V.5 / VI.3), not a substrate


def part4_ranking(embedder) -> None:
    """Reproduce RESULTS.md V.5 and VI.3: does whitening help retrieval?

    Review ticket 06: V.5 and VI.3 report LoCoMo turn-retrieval hit@k with and
    without whitening, but neither number had a reproducible source in this
    file -- Part 2 above only ever measured capacity (decoding a superposed
    state), never ranking. `dense_knn_ranker`/`score_ranking`
    (`experiments/recall_ablation.py`) are the ranking harness V.5/VI.3 were
    actually measured with; `WhitenedEmbedder` (`cls_memory.whitening`) is the
    library wrapper, not a private duplicate.

    Same 3 conversations, n=494 as V.5/VI.3. Two fits are reported, continuing
    review ticket 04's discipline:

      * **in-sample fit** -- the whitener fit on the SAME 1451 turns being
        ranked. This is what VI.3's "fitted once on 1451 turns" row measured;
        it is not transductive in the query direction (queries are held-out
        question text, never stored turns) but the *documents* it is scored
        against are exactly what it was fit on.
      * **held-out fit** -- fit on a disjoint pool of turns from the other 7
        LoCoMo conversations, applied unchanged to the 3 scored ones. The
        honest number for a deployment that fits once on historical traffic
        and then encodes conversations it has never seen.
    """
    print("\n" + "=" * 78)
    print("PART 4 -- whitening as a RANKER (RESULTS.md V.5 / VI.3), not a substrate")
    print("=" * 78)

    conversations = locomo.load()
    scored = conversations[:3]
    scored_corpus = [t.memory_text for c in scored for t in c.turns]
    fit_pool = [t.memory_text for c in conversations[3:] for t in c.turns]
    n_asked = sum(
        1 for c in scored for q in c.questions
        if {e for e in q.evidence if e in {t.dia_id for t in c.turns}}
    )
    print(f"  {len(scored)} conversations, {len(scored_corpus)} turns, "
          f"{n_asked} questions with in-corpus evidence\n")

    score_ranking(scored, lambda c: dense_knn_ranker(c, embedder), "BGE as shipped")

    in_sample = WhitenedEmbedder(embedder).fit(scored_corpus)
    score_ranking(scored, lambda c: dense_knn_ranker(c, in_sample),
                  "whitened, in-sample fit (on the 1451 scored turns)")

    held_out = WhitenedEmbedder(embedder).fit(fit_pool)
    score_ranking(scored, lambda c: dense_knn_ranker(c, held_out),
                  f"whitened, held-out fit (on {len(fit_pool)} turns, other 7 convs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(0)

    embedder = BGEEmbedder()
    part1_identity(embedder)
    part2_capacity(embedder, args.trials)
    part3_budget(embedder)
    part4_ranking(embedder)


if __name__ == "__main__":
    main()
