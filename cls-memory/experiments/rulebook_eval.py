"""Rules under a token budget: does the energy landscape beat plain retrieval?

    PYTHONPATH=. .venv/bin/python experiments/rulebook_eval.py

The cost model
--------------
Retrieved rules are only useful pasted into a prompt, so the currency is
tokens, not hit@k. Three numbers per arm, all measured with the real tokenizer:

    coverage@256    fraction of needed rules present in a 256-token budget
    tokens@cover    tokens that must be injected to cover every needed rule
    stale@256       fraction of situations where a SUPERSEDED rule got injected

`stale@256` is the one that matters and the one nobody measures. A missing rule
makes the model say "I don't know". An obsolete rule makes it confidently quote
a policy that was revoked a year ago -- so a stale injection is worse than an
empty context, and every similarity-ranked arm is prone to it because v1 and v2
of a rule differ by one token.

Arms
----
    BM25                     lexical, no notion of time
    BGE kNN                  dense, no notion of time
    BGE + recency rerank     the cheap fix, and the bar the architecture must clear
    MHN (recency log-prior)  this project's claim: decay folded into the energy
    ORACLE current-only      upper bound if supersession metadata were free

The oracle exists to size the prize. If the gap between BGE kNN and the oracle
is small, no amount of memory physics can pay for itself here.

Part 2 asks the write-time question separately: when a revised rule arrives,
can the system *tell* it supersedes an existing one? That is where a density
model could beat a similarity threshold, because "lands inside an existing
basin" is normalised against local crowding and "cosine > 0.9" is not.

Honest limits: 26 rules, 16 situations. Nothing here is statistically
resolvable -- per-situation detail is printed for exactly that reason. Treat
this as a mechanism check, not a benchmark.
"""

from __future__ import annotations

import argparse

import torch

from cls_memory import HopfieldConfig, ModernHopfieldNetwork
from experiments.metrics import roc_auc
from experiments.recall_ablation import BGEEmbedder, BM25Index
from experiments.rulebook import (
    RULES,
    SITUATIONS,
    Rule,
    by_id,
    current_rule_ids,
    stale_rule_ids,
)

BUDGET = 256
HALF_LIFE_DAYS = 180.0


def token_counts(embedder: BGEEmbedder, rules: list[Rule]) -> dict[str, int]:
    """Real token counts, not word counts -- the budget is a prompt budget."""
    encoded = embedder._tok.encode_batch([r.text for r in rules])
    return {r.rule_id: len(e.ids) for r, e in zip(rules, encoded)}


def evaluate(name: str, ranker, tokens: dict[str, int], *, verbose: bool) -> dict:
    """Walk each ranking under a token budget and score coverage vs staleness."""
    stale = stale_rule_ids()
    covered, spend, stale_hits, trap_hits = [], [], [], []
    detail = []

    for sit in SITUATIONS:
        ranked = ranker(sit.question)
        gold = set(sit.gold)
        traps = set(sit.traps)

        spent = 0
        injected: list[str] = []
        for rule_id in ranked:
            cost = tokens[rule_id]
            if spent + cost > BUDGET:
                continue
            spent += cost
            injected.append(rule_id)

        got = gold & set(injected)
        covered.append(len(got) / len(gold))
        stale_hits.append(int(bool(set(injected) & stale)))
        trap_hits.append(int(bool(set(injected) & traps)))

        # Tokens needed to cover every gold rule, ignoring the budget cap.
        running, need = 0, set(gold)
        for rule_id in ranked:
            running += tokens[rule_id]
            need.discard(rule_id)
            if not need:
                break
        spend.append(running if not need else sum(tokens.values()))
        detail.append((sit, injected, got, set(injected) & stale))

    n = len(SITUATIONS)
    result = {
        "coverage": sum(covered) / n,
        "tokens": sum(spend) / n,
        "stale": sum(stale_hits) / n,
        "trap": sum(trap_hits) / n,
    }
    print(f"  {name:<28} coverage@{BUDGET} {result['coverage']:.2f}   "
          f"tokens@cover {result['tokens']:5.0f}   "
          f"stale@{BUDGET} {result['stale']:.2f}   trap@{BUDGET} {result['trap']:.2f}")

    if verbose:
        for sit, injected, got, bad in detail:
            if bad or len(got) < len(sit.gold):
                flag = "STALE" if bad else "miss "
                print(f"      [{flag}] {sit.question[:58]:<58} "
                      f"got={sorted(got)} stale={sorted(bad)}")
    return result


# ---------------------------------------------------------------------------
# arms


def bm25_arm(rules: list[Rule]):
    index = BM25Index([r.text for r in rules])
    ids = [r.rule_id for r in rules]
    return lambda q: [ids[i] for i in index.rank(q, top=len(rules))]


def knn_arm(rules: list[Rule], embedder, *, recency: float = 0.0):
    """Dense kNN, optionally with a recency bonus added to the similarity.

    `recency` is in cosine units per half-life: the cheap, obvious fix that any
    engineer would reach for before building an energy model.
    """
    matrix = embedder.encode([r.text for r in rules])
    ids = [r.rule_id for r in rules]
    newest = max(r.day for r in rules)
    age = torch.tensor([(newest - r.day) / HALF_LIFE_DAYS for r in rules])

    def rank(question: str) -> list[str]:
        q = embedder.encode_query([question])[0]
        scores = matrix @ q - recency * age
        return [ids[i] for i in scores.argsort(descending=True).tolist()]

    return rank


def mhn_arm(rules: list[Rule], embedder, beta: float, *, decay: bool = True):
    """The architecture's own claim: forgetting folded into the energy.

    log_prior_i = -ln(2) * age / half_life is exactly the Ebbinghaus curve the
    design specifies, entering as a log-prior on the softmax logits rather than
    as a post-hoc rerank. Whether that is *different* from a post-hoc rerank is
    the thing being tested.
    """
    matrix = embedder.encode([r.text for r in rules])
    ids = [r.rule_id for r in rules]
    mhn = ModernHopfieldNetwork(matrix.shape[1], HopfieldConfig(beta=beta))
    newest = max(r.day for r in rules)
    if decay:
        prior = torch.tensor(
            [-torch.log(torch.tensor(2.0)) * (newest - r.day) / HALF_LIFE_DAYS
             for r in rules]
        )
    else:
        prior = torch.zeros(len(rules))
    mhn.write(matrix, log_prior=prior)

    def rank(question: str) -> list[str]:
        q = embedder.encode_query([question])[0]
        logits = mhn.logits(q)
        return [ids[i] for i in logits.argsort(descending=True).tolist()]

    return rank


def gated_arm(rules: list[Rule], embedder, threshold: float):
    """Stream the rules in date order and supersede on write, not on read.

    This is the ingestion gate the CLS design already specifies, pointed at the
    right problem: when a rule arrives that lands on top of one already stored,
    the old one is *replaced* rather than accumulated. Retrieval afterwards is
    ordinary kNN -- the intelligence has moved to write time, where it is paid
    for once instead of on every query.

    Also reports how many of its replacement decisions were correct, because an
    over-eager gate silently deletes live policy, which is a worse failure than
    anything measured in Part 1.
    """
    order = sorted(rules, key=lambda r: r.day)
    vectors = {r.rule_id: v for r, v in zip(rules, embedder.encode([r.text for r in rules]))}

    live: list[Rule] = []
    right = wrong = 0
    for rule in order:
        if live:
            stack = torch.stack([vectors[r.rule_id] for r in live])
            sims = stack @ vectors[rule.rule_id]
            best = int(sims.argmax())
            if float(sims[best]) >= threshold:
                replaced = live[best]
                if rule.supersedes == replaced.rule_id:
                    right += 1
                else:
                    wrong += 1
                    print(f"    [gate @{threshold:.2f} WRONGLY dropped "
                          f"{replaced.rule_id} when {rule.rule_id} arrived "
                          f"(cos {float(sims[best]):.3f})]")
                live.pop(best)
        live.append(rule)

    kept = {r.rule_id for r in live}
    missed = sum(1 for r in rules if r.supersedes and r.supersedes in kept)
    print(f"    [gate @{threshold:.2f}: {right} correct replacements, "
          f"{wrong} wrong, {missed} stale rules left in the store, "
          f"{len(live)}/{len(rules)} kept]")
    return knn_arm(live, embedder)


def oracle_arm(rules: list[Rule], embedder):
    live = [r for r in rules if r.rule_id in current_rule_ids()]
    inner = knn_arm(live, embedder)
    return inner


# ---------------------------------------------------------------------------
# Part 2: can supersession be detected at write time?


def supersession_probe(rules: list[Rule], embedder, beta: float) -> None:
    """Separate true supersession pairs from merely-similar pairs.

    Positives are the 7 annotated (new, old) pairs. Negatives are every other
    ordered pair where the older rule already existed -- the decisions a real
    ingest gate faces. If cosine already separates them perfectly there is
    nothing for an energy model to add.

    The store is reconstructed in the *stream* order the gate actually sees,
    including rules that took effect on the same day. An earlier version of
    this probe used `r.day < new.day`, which quietly excluded every same-day
    pair -- and the same-day pairs are precisely the scope siblings
    (exp.travel.eu/us, data.pii.eu/us) that the gate gets wrong. It reported
    AUC 1.000 for both signals while the gate was making mistakes two lines
    further down the file.
    """
    matrix = embedder.encode([r.text for r in rules])
    index = {r.rule_id: i for i, r in enumerate(rules)}
    order = sorted(rules, key=lambda r: r.day)

    pos_cos, neg_cos, pos_eng, neg_eng = [], [], [], []
    for position, new in enumerate(order):
        older = order[:position]
        if not older:
            continue
        # A store holding only the rules that existed when `new` arrived.
        mhn = ModernHopfieldNetwork(matrix.shape[1], HopfieldConfig(beta=beta))
        mhn.write(matrix[[index[r.rule_id] for r in older]])
        xi = matrix[index[new.rule_id]]
        sims = mhn.patterns @ xi
        # Energy drop from writing this pattern: how much the landscape already
        # explains it, normalised over the whole existing store.
        explained = float(torch.logsumexp(beta * sims, dim=0) - torch.log(
            torch.tensor(float(len(older)))
        )) / beta

        best = int(sims.argmax())
        matched = older[best].rule_id
        is_true = new.supersedes is not None and new.supersedes == matched
        (pos_cos if is_true else neg_cos).append(float(sims[best]))
        (pos_eng if is_true else neg_eng).append(explained)

    if not pos_cos:
        print("  [no positive supersession pairs found]")
        return
    print(f"  positives {len(pos_cos)}, negatives {len(neg_cos)}")
    print(f"  top-1 cosine      AUC {roc_auc(torch.tensor(pos_cos), torch.tensor(neg_cos)):.3f}"
          f"   (pos mean {sum(pos_cos)/len(pos_cos):.3f}, "
          f"neg mean {sum(neg_cos)/len(neg_cos):.3f})")
    print(f"  energy explained  AUC {roc_auc(torch.tensor(pos_eng), torch.tensor(neg_eng)):.3f}"
          f"   (pos mean {sum(pos_eng)/len(pos_eng):.3f}, "
          f"neg mean {sum(neg_eng)/len(neg_eng):.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=128.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)

    embedder = BGEEmbedder()
    tokens = token_counts(embedder, RULES)
    print(f"{len(RULES)} rules ({len(stale_rule_ids())} superseded), "
          f"{len(SITUATIONS)} situations, budget {BUDGET} tokens "
          f"(mean rule = {sum(tokens.values())/len(tokens):.0f} tokens, so a "
          f"budget holds ~{BUDGET // (sum(tokens.values())//len(tokens))} rules)\n")

    print("PART 1 -- retrieval under a token budget")
    evaluate("BM25", bm25_arm(RULES), tokens, verbose=args.verbose)
    evaluate("BGE kNN", knn_arm(RULES, embedder), tokens, verbose=args.verbose)
    evaluate("BGE + recency rerank", knn_arm(RULES, embedder, recency=0.02),
             tokens, verbose=args.verbose)
    evaluate("MHN, no decay", mhn_arm(RULES, embedder, args.beta, decay=False),
             tokens, verbose=args.verbose)
    evaluate("MHN + recency log-prior", mhn_arm(RULES, embedder, args.beta),
             tokens, verbose=args.verbose)
    for threshold in (0.80, 0.75, 0.70):
        evaluate(f"INGEST-GATED kNN (t={threshold:.2f})",
                 gated_arm(RULES, embedder, threshold), tokens,
                 verbose=args.verbose)
    evaluate("ORACLE current-only kNN", oracle_arm(RULES, embedder), tokens,
             verbose=args.verbose)

    print("\nPART 2 -- can supersession be detected when the rule is written?")
    supersession_probe(RULES, embedder, args.beta)


if __name__ == "__main__":
    main()
