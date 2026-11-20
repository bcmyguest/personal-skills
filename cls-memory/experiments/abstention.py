"""Bet A: is the energy a better "I don't know" signal than cosine similarity?

    PYTHONPATH=. .venv/bin/python experiments/abstention.py

The question this project has never answered
--------------------------------------------
Every retrieval measurement so far asked "did you find the right memory?", and
on that question the Hopfield network ties plain cosine kNN exactly (0.172 vs
0.174 hit@1) -- because hit@1 against a single gold item is the metric where
nearest-neighbour is optimal by construction, so no attractor dynamics can beat
it. The one thing an energy model has that a similarity score does not is a
*normalised* density: `log_density` integrates to 1.0 (verified to 14 decimal
places in the energy tests), so it knows whether a query landed in a dense
region or an empty one. A cosine is an unnormalised similarity that cannot.

That difference should show up in abstention -- knowing when the answer is not
in memory at all -- and nowhere else.

The test set was already downloaded
-----------------------------------
LoCoMo category 5 (446 of 1986 questions) is adversarial: the answer field is
deliberately empty because the question presupposes something false, usually by
attributing one speaker's action to the other ("What did Caroline realize after
*her* charity race?" when it was Melanie's race). The correct behaviour is to
decline. Every previous harness in this repo threw these away.

They are a *hard* abstention set on purpose, and that cuts against the
hypothesis being tested here: the near-miss turn genuinely is in memory, so the
region really is dense and a density-based signal has no obvious edge. A
generic "is this out-of-distribution" test would be easier and would prove
less.

What is measured
----------------
Six confidence signals over the same embeddings and the same queries, so the
only thing that varies is the scoring function:

    cos_top1        max cosine to any stored turn           (the baseline)
    cos_margin      top1 - top2 cosine                      (stronger baseline)
    neg_attn_ent    peakedness of the retrieval attention   (free, MHN-ish)
    log_density     normalised log p_sigma(query)           (MHN only)
    neg_depth_nats  -(log p(nearest memory) - log p(query)) (MHN only)
    neg_settle      -(distance the query moved when settled)(MHN only)

Reported as ROC AUC with a bootstrap CI, plus a paired bootstrap on the
difference against `cos_top1`, plus a leave-one-conversation-out logistic fit
that asks the only question that really matters: does the energy add anything
*on top of* cosine, or is it a monotone function of it?

Two confound checks run first. If question length alone separates the classes,
or if the adversarial questions are trivially distinguishable from surface
form, the whole comparison is contaminated and no result below means anything.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from cls_memory import HopfieldConfig, ModernHopfieldNetwork, basin_depth, log_density
from experiments import locomo
from experiments.metrics import roc_auc
from experiments.recall_ablation import BGEEmbedder, HashedProjection, HybridEmbedder

SEED = 0
ADVERSARIAL = 5

SIGNALS = [
    "cos_top1",
    "cos_margin",
    "neg_attn_ent",
    "log_density",
    "neg_depth_nats",
    "neg_settle",
]


def collect(conversations, embedder_factory, beta: float) -> dict:
    """Score every question under every signal. Returns column-oriented data."""
    columns: dict[str, list[float]] = {s: [] for s in SIGNALS}
    columns["length"] = []
    labels: list[int] = []       # 1 = answerable, 0 = adversarial
    conv_index: list[int] = []   # for leave-one-conversation-out

    for c, conv in enumerate(conversations):
        docs = [t.memory_text for t in conv.turns]
        ids = {t.dia_id for t in conv.turns}
        embedder = embedder_factory(docs)
        matrix = embedder.encode(docs)
        encode_query = getattr(embedder, "encode_query", embedder.encode)

        mhn = ModernHopfieldNetwork(matrix.shape[1], HopfieldConfig(beta=beta))
        mhn.write(matrix)

        questions = [
            q for q in conv.questions if any(e in ids for e in q.evidence)
        ]
        if not questions:
            continue
        queries = encode_query([q.question for q in questions])

        for q, xi in zip(questions, queries):
            sims = mhn.patterns @ xi
            top2 = torch.topk(sims, min(2, sims.numel())).values
            attn = mhn.attention(xi)
            # Entropy of the retrieval attention: a peaked distribution means
            # one memory explains the cue, a flat one means none does.
            entropy = float(-(attn * attn.clamp_min(1e-30).log()).sum())
            report = basin_depth(mhn, xi)
            settled = mhn.step(xi)

            columns["cos_top1"].append(float(top2[0]))
            columns["cos_margin"].append(
                float(top2[0] - top2[1]) if top2.numel() > 1 else 0.0
            )
            columns["neg_attn_ent"].append(-entropy)
            columns["log_density"].append(float(log_density(mhn, xi)))
            columns["neg_depth_nats"].append(-report.depth_nats)
            columns["neg_settle"].append(-float((settled - xi).norm()))
            columns["length"].append(float(len(q.question)))
            labels.append(int(q.category != ADVERSARIAL))
            conv_index.append(c)

    return {"columns": columns, "labels": labels, "conv": conv_index}


# --------------------------------------------------------------------------
# statistics


def auc_with_ci(values: list[float], labels: list[int], *, draws: int = 2000,
                generator: torch.Generator) -> tuple[float, float, float]:
    """AUC plus a percentile bootstrap CI over questions."""
    v = torch.tensor(values, dtype=torch.double)
    y = torch.tensor(labels, dtype=torch.bool)
    point = roc_auc(v[y], v[~y])

    n = v.numel()
    samples = []
    for _ in range(draws):
        idx = torch.randint(0, n, (n,), generator=generator)
        yy = y[idx]
        if bool(yy.all()) or not bool(yy.any()):
            continue
        samples.append(roc_auc(v[idx][yy], v[idx][~yy]))
    if not samples:
        return point, float("nan"), float("nan")
    s = torch.tensor(samples, dtype=torch.double).sort().values
    return point, float(s[int(0.025 * len(s))]), float(s[int(0.975 * len(s))])


def paired_auc_delta(a: list[float], b: list[float], labels: list[int], *,
                     draws: int = 2000,
                     generator: torch.Generator) -> tuple[float, float, float]:
    """Bootstrap the *difference* AUC(b) - AUC(a) on the same resamples.

    Paired, because the two signals score the same questions -- comparing two
    independent CIs would badly overstate the uncertainty of the difference.
    """
    va = torch.tensor(a, dtype=torch.double)
    vb = torch.tensor(b, dtype=torch.double)
    y = torch.tensor(labels, dtype=torch.bool)
    point = roc_auc(vb[y], vb[~y]) - roc_auc(va[y], va[~y])

    n = va.numel()
    deltas = []
    for _ in range(draws):
        idx = torch.randint(0, n, (n,), generator=generator)
        yy = y[idx]
        if bool(yy.all()) or not bool(yy.any()):
            continue
        deltas.append(
            roc_auc(vb[idx][yy], vb[idx][~yy]) - roc_auc(va[idx][yy], va[idx][~yy])
        )
    d = torch.tensor(deltas, dtype=torch.double).sort().values
    return point, float(d[int(0.025 * len(d))]), float(d[int(0.975 * len(d))])


def fit_logistic(x: torch.Tensor, y: torch.Tensor, *, steps: int = 400,
                 l2: float = 1e-3) -> torch.Tensor:
    """Plain logistic regression (no sklearn here). Returns weights + bias."""
    n, d = x.shape
    design = torch.cat([x, torch.ones(n, 1, dtype=x.dtype)], dim=1)
    w = torch.zeros(d + 1, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            design @ w, y
        ) + l2 * w[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def loco_auc(data: dict, features: list[str]) -> float:
    """Out-of-fold AUC of a logistic combination, held out by conversation.

    Held out by *conversation* rather than by question because the embedder and
    the memory store are fitted per conversation: a random question split would
    let the fold see its own memory landscape and inflate the result.
    """
    cols = data["columns"]
    y_all = torch.tensor(data["labels"], dtype=torch.double)
    conv = torch.tensor(data["conv"])
    x_all = torch.stack(
        [torch.tensor(cols[f], dtype=torch.double) for f in features], dim=1
    )

    scores = torch.zeros_like(y_all)
    for held in conv.unique().tolist():
        test = conv == held
        train = ~test
        if not bool(train.any()) or not bool(test.any()):
            continue
        mu = x_all[train].mean(0)
        sd = x_all[train].std(0).clamp_min(1e-9)
        w = fit_logistic((x_all[train] - mu) / sd, y_all[train])
        z = (x_all[test] - mu) / sd
        scores[test] = torch.cat(
            [z, torch.ones(int(test.sum()), 1, dtype=z.dtype)], dim=1
        ) @ w

    mask = y_all.bool()
    return roc_auc(scores[mask], scores[~mask])


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation. Two signals at rho ~ 1.0 are the same signal.

    Worth reporting explicitly: at high beta the mixture density is dominated
    by its nearest component, so `log_density` degenerates into a monotone
    function of `cos_top1` and *cannot* differ from it, whatever the AUCs say.
    Any claim that the energy is a distinct confidence signal has to survive
    this check first.
    """
    def ranks(v: list[float]) -> torch.Tensor:
        t = torch.tensor(v, dtype=torch.double)
        order = t.argsort()
        r = torch.empty_like(t)
        r[order] = torch.arange(1, t.numel() + 1, dtype=torch.double)
        return r

    x, y = ranks(a), ranks(b)
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))


def operating_point(values: list[float], labels: list[int],
                    keep: float = 0.80) -> tuple[float, float]:
    """At the threshold retaining `keep` of answerable questions, what fraction
    of adversarial ones are correctly rejected?

    The operational form of the question: an abstaining system is only useful
    if it drops mostly-unanswerable queries while keeping mostly-answerable
    ones.
    """
    v = torch.tensor(values, dtype=torch.double)
    y = torch.tensor(labels, dtype=torch.bool)
    pos = v[y].sort().values
    cut = float(pos[int((1.0 - keep) * (pos.numel() - 1))])
    kept_pos = float((v[y] >= cut).double().mean())
    rejected_neg = float((v[~y] < cut).double().mean())
    return kept_pos, rejected_neg


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=10)
    parser.add_argument("--beta", type=float, default=128.0)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    conversations = locomo.load()[: args.conversations]
    bge = BGEEmbedder()

    def factory(docs):
        lexical = HashedProjection(docs, dim=4096, weighting="bm25", seed=SEED)
        return HybridEmbedder(lexical, bge, w=0.5)

    t0 = time.time()
    data = collect(conversations, factory, args.beta)
    labels = data["labels"]
    cols = data["columns"]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"{len(conversations)} conversations, {len(labels)} questions "
          f"({n_pos} answerable, {n_neg} adversarial), beta={args.beta:g}, "
          f"{time.time() - t0:.0f}s\n")

    print("CONFOUND CHECK  (0.500 = no signal; anything high means the classes")
    print("                 differ in surface form and the rest is suspect)")
    point, lo, hi = auc_with_ci(cols["length"], labels, draws=args.draws,
                                generator=generator)
    print(f"  question length alone            AUC {point:.3f}  [{lo:.3f}, {hi:.3f}]\n")

    print("ABSTENTION AUC  (1.0 = perfect separation of answerable from")
    print("                 adversarial; 0.5 = coin flip)")
    results = {}
    for name in SIGNALS:
        point, lo, hi = auc_with_ci(cols[name], labels, draws=args.draws,
                                    generator=generator)
        results[name] = point
        keep, reject = operating_point(cols[name], labels)
        tag = "  <- baseline" if name == "cos_top1" else ""
        print(f"  {name:<16} AUC {point:.3f}  [{lo:.3f}, {hi:.3f}]   "
              f"rejects {reject:.1%} of adversarial at {keep:.0%} coverage{tag}")

    print("\nIS THE ENERGY EVEN A DIFFERENT SIGNAL?  (Spearman vs cos_top1;")
    print("                 rho ~ 1.0 means it is a relabelling of cosine)")
    for name in SIGNALS:
        if name == "cos_top1":
            continue
        print(f"  {name:<16} rho {spearman(cols['cos_top1'], cols[name]):+.4f}")

    print("\nPAIRED vs cos_top1  (does the energy beat plain cosine?)")
    for name in SIGNALS:
        if name == "cos_top1":
            continue
        delta, lo, hi = paired_auc_delta(cols["cos_top1"], cols[name], labels,
                                         draws=args.draws, generator=generator)
        verdict = "TIE" if lo <= 0.0 <= hi else ("WINS" if delta > 0 else "LOSES")
        print(f"  {name:<16} delta {delta:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    print("\nINCREMENTAL VALUE  (out-of-fold AUC, held out by conversation)")
    base = loco_auc(data, ["cos_top1"])
    both = loco_auc(data, ["cos_top1", "cos_margin"])
    energy = loco_auc(data, ["cos_top1", "log_density", "neg_depth_nats"])
    every = loco_auc(data, SIGNALS)
    print(f"  cosine only                      AUC {base:.3f}")
    print(f"  cosine + margin                  AUC {both:.3f}")
    print(f"  cosine + energy signals          AUC {energy:.3f}")
    print(f"  everything                       AUC {every:.3f}")
    print(f"\n  energy adds {energy - base:+.3f} over cosine alone.")


if __name__ == "__main__":
    main()
