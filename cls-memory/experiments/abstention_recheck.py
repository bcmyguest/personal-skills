"""Review ticket 12: is Part IV's abstention verdict a beta artefact?

    PYTHONPATH=. uv run python -u experiments/abstention_recheck.py --corpus locomo

RESULTS.md IV.1 measured abstention at beta=128, then separately swept beta up
to 512 and reported rho(`log_density`, `cos_top1`) rising from +0.242 (beta=8)
through +0.835 (beta=32) to +1.0000 (beta=128 and 512) -- i.e. IV.1's OWN
published table already shows `log_density` becoming a monotone function of
`cos_top1` at the exact operating point IV.1 used for its headline verdict.
At that point "the energy adds nothing over cosine" is restating arithmetic,
not testing a hypothesis: a signal that is provably a relabelling of cosine
cannot outscore cosine, whatever the abstention task looks like. (A
since-completed ticket-09 sweep is consistent with this: rho_signal 0.839 at
beta=32, 0.998-1.000 at beta=128/512 on LoCoMo -- cited here only to note
agreement, not as this module's source for the numbers above.)

TWO PROTOCOLS, not one grid
----------------------------
An earlier version of this module ran a single grid built on
`separation_beta_sweep.build_embedder`/`build_system` (LSA embeddings, a
cortex-derived key, salience/decay priors) and treated its baseline arm as
comparable to IV.1. It is not: IV.1's own harness
(`experiments/abstention.py::collect`) uses a HYBRID embedder (BM25-hashed
lexical + BGE dense, `HybridEmbedder(w=0.5)`), a BARE `ModernHopfieldNetwork`
written directly with no cortex and no key encoder (hence uniform log-priors,
no salience decay), and `encode_query` for cues. Every one of those is a
different write path, and every one changes the numbers. Silently swapping
the write path and calling the result "the beta=128 cell" would repeat
exactly the mistake this ticket exists to correct, just one level down.

So there are two protocols, printed with their own label, and only one of
them is allowed to speak for Part IV:

  **Protocol A -- "IV.1 replication" (PRIMARY).** Calls
  `abstention.collect(conversations, factory, beta)` UNMODIFIED, with IV.1's
  exact hybrid factory, swept over `--betas` instead of IV.1's single
  beta=128. The only separation lever available at this write path is
  whitening (no key encoder exists to run DG through), so Protocol A arms are
  {baseline, whitened} x betas: `whitened` wraps the same hybrid embedder in
  `cls_memory.whitening.WhitenedEmbedder`, fit on the same per-conversation
  docs `collect` already builds. `run_protocol_a` checks its own beta=128
  baseline row against RESULTS.md IV.1's published numbers (cos_top1 0.480,
  log_density 0.481, neg_depth_nats 0.481, neg_settle 0.505) and prints the
  comparison unconditionally -- a mismatch is reported as loudly as a match,
  because at smoke scale (this module never runs more than the caller's
  `--locomo` conversations, and BGE is slow enough that a real run is the
  caller's job, not this module's) a difference is expected from sample size
  alone and silence would look like either a false pass or a hidden failure.
  ONLY Protocol A's numbers may be used to confirm, narrow, or withdraw
  IV.1's verdict.

  **Protocol B -- "full system" (SECONDARY, not comparable to IV.1).** The
  original grid, kept because it is the only place the DG negative control
  can run (DG needs `cls_memory.pattern_separation.KeyEncoder`, which the
  bare-MHN write path in Protocol A does not have). Built on
  `separation_beta_sweep.build_embedder`/`build_system`: LSA embeddings, a
  full `OrganizationalMemory` with cortex-derived keys, and time-accurate
  salience priors via `store.refresh_priors`. These are legitimate
  engineering choices for the shipped system, but they are not IV.1's write
  path, so a Protocol B row and a Protocol A row at the same beta are not the
  same measurement and must not be read side by side as if they were.

`--betas` is a parameter, never a constant, in both protocols: the beta at
which separation stops being a relabelling of cosine is exactly what is under
test, so hardcoding one value would silently reintroduce the defect this
ticket exists to fix. Default is `2 8 32 128 512`, the same points IV.1's own
sweep and ticket 09's sweep both used.

Every cell in both protocols reports rho(signal, cos_top1) via
`metrics.spearman_rho` right next to its AUC, so a WINS verdict that is
secretly degenerate is visible without cross-referencing a second table. Both
protocols keep IV.1's original devices exactly as they were: out-of-fold
evaluation held out by conversation (`abstention.loco_auc`), percentile
bootstrap CIs (`abstention.auc_with_ci`, `abstention.paired_auc_delta`), and a
Bonferroni correction -- sized to each protocol's OWN family of paired tests
(arms x betas x 5 non-baseline signals for that protocol), since Protocol A
and B run different numbers of arms and pooling their corrections would
either over- or under-correct one of them.

LoCoMo category 5 is the only unanswerable-question set this project has --
QMSum's questions (`experiments/qmsum.py`) are all built from a real,
answerable `relevant_text_span` and carry category=1 unconditionally; there
is no QMSum analogue of "the presupposition is false; decline". `--corpus
qmsum` is therefore accepted (so the flag is honestly reachable) but the
script refuses to invent a negative class and says so instead of measuring
something that looks like abstention but is not.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from cls_memory import HippocampalKey, MemoryRecord, Persistence, basin_depth, log_density
from cls_memory.whitening import WhitenedEmbedder
from experiments import locomo
from experiments.abstention import (
    ADVERSARIAL,
    SEED,
    SIGNALS,
    auc_with_ci,
    collect,
    loco_auc,
    operating_point,
    paired_auc_delta,
)
from experiments.metrics import roc_auc, spearman_rho
from experiments.recall_ablation import BGEEmbedder, HashedProjection, HybridEmbedder
from experiments.separation_beta_sweep import ARMS as SWEEP_ARMS
from experiments.separation_beta_sweep import build_embedder, build_system
from experiments.threads import add_threads_arg, pin_threads

# Protocol B's negative-control triple ("both" is ticket 09's fourth arm and
# is not part of this recheck's design; Protocol A doesn't use this list at
# all, since it has no key encoder to run DG or "both" through).
ARMS: list[tuple[str, HippocampalKey, bool]] = [
    a for a in SWEEP_ARMS if a[0] != "both"
]

DEFAULT_BETAS: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0, 512.0)

# RESULTS.md IV.1's published beta=128 baseline row -- the only numbers
# Protocol A's reproduction check is allowed to compare against.
IV1_PUBLISHED_AUC_AT_BETA_128: dict[str, float] = {
    "cos_top1": 0.480,
    "log_density": 0.481,
    "neg_depth_nats": 0.481,
    "neg_settle": 0.505,
}


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    """Family-wise alpha spread over `n_tests` independent paired comparisons.

    Pure arithmetic, split out so each protocol and its test can agree on the
    number without re-deriving it inline. `n_tests` must be >= 1: a family of
    zero comparisons has nothing to correct and is almost certainly a caller
    bug (an empty --betas or --arms list), not a valid alpha of 0.05.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return alpha / n_tests


def paired_auc_delta_ci(
    a: list[float], b: list[float], labels: list[int], *, alpha: float,
    draws: int, generator: torch.Generator,
) -> tuple[float, float, float]:
    """`abstention.paired_auc_delta`, generalised to an arbitrary CI width.

    `paired_auc_delta` hardcodes the 95% percentile bootstrap IV.1 used and is
    left untouched (ticket 12 requires keeping it, and other callers rely on
    exactly that signature). This is a separate function, not an edit to that
    one, so the report can additionally ask "does this delta survive the
    Bonferroni-corrected alpha for THIS family's size" without changing what
    IV.1's original 95% CI means.
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
    lo = int((alpha / 2) * len(d))
    hi = int((1 - alpha / 2) * len(d))
    return point, float(d[lo]), float(d[min(hi, len(d) - 1)])


# --------------------------------------------------------------------------
# Protocol A -- IV.1 replication, unmodified `collect`, swept over beta


def _iv1_hybrid_factory(docs: list[str], bge: BGEEmbedder) -> HybridEmbedder:
    """Byte-for-byte the closure `abstention.py::main` builds inline.

    Not importable (it is a closure, not a module-level function), so it is
    reproduced here verbatim rather than approximated -- `dim=4096`,
    `weighting="bm25"`, `seed=SEED`, `w=0.5` all match IV.1 exactly.
    """
    lexical = HashedProjection(docs, dim=4096, weighting="bm25", seed=SEED)
    return HybridEmbedder(lexical, bge, w=0.5)


def _iv1_whitened_factory(docs: list[str], bge: BGEEmbedder) -> WhitenedEmbedder:
    """IV.1's hybrid embedder, whitened -- Protocol A's only separation lever.

    No key encoder exists on this bare-MHN write path (that's what makes it
    IV.1's path and not Protocol B's), so DG is not reachable here; whitening
    is. Fit on the SAME per-conversation `docs` `collect` already builds the
    lexical half from, matching the transductive fit every other whitening
    measurement in this project uses (see `cls_memory/whitening.py`).
    """
    hybrid = _iv1_hybrid_factory(docs, bge)
    whitened = WhitenedEmbedder(hybrid)
    whitened.fit(docs)
    return whitened


def check_reproduction(cell: dict, published: dict[str, float]) -> bool:
    """Compare a Protocol A beta=128 baseline cell's point AUCs against
    RESULTS.md IV.1's published numbers. Always prints; returns whether every
    compared signal matched within `tol`.

    Exact reproduction is not expected off a smoke-scale run (this project's
    `--locomo` default is 1-2 conversations for a smoke test vs IV.1's
    published 10 conversations / 1977 questions) -- but the comparison is
    printed regardless, with both numbers, so a real difference is never
    silently absorbed into "looks about right".
    """
    cols = cell["columns"]
    labels = cell["labels"]
    tol = 0.05
    all_match = True
    print("\n  reproduction check vs RESULTS.md IV.1 (beta=128, baseline, "
          f"tol={tol}; smoke-scale n={len(labels)} vs IV.1's n=1977, so an "
          "exact match is not expected -- this checks the CODE PATH, not "
          "sample-size-matched agreement):")
    for name, published_auc in published.items():
        point = roc_auc(
            torch.tensor(cols[name], dtype=torch.double)[torch.tensor(labels, dtype=torch.bool)],
            torch.tensor(cols[name], dtype=torch.double)[~torch.tensor(labels, dtype=torch.bool)],
        )
        diff = point - published_auc
        matched = abs(diff) <= tol
        all_match = all_match and matched
        flag = "OK" if matched else "MISMATCH"
        print(f"    {name:<16} measured {point:.3f}  published {published_auc:.3f}  "
              f"diff {diff:+.3f}  [{flag}]")
    if not all_match:
        print("    !! at least one signal is outside tol -- at smoke scale this is "
              "expected sampling noise, NOT evidence the code path is wrong; "
              "re-run with the caller's full --locomo default to check for real.")
    return all_match


def run_protocol_a(
    conversations, *, betas: tuple[float, ...], draws: int, generator: torch.Generator,
) -> None:
    """IV.1 replication: unmodified `abstention.collect`, swept over beta."""
    print("\n==================== PROTOCOL A: IV.1 replication (PRIMARY) "
          "====================")
    print("Uses abstention.collect() UNMODIFIED with IV.1's exact hybrid "
          "factory. Only this protocol's numbers may confirm/narrow/withdraw "
          "Part IV's verdict.")

    bge = BGEEmbedder()
    factories = {
        "baseline": lambda docs: _iv1_hybrid_factory(docs, bge),
        "whitened": lambda docs: _iv1_whitened_factory(docs, bge),
    }
    n_tests = len(factories) * len(betas) * (len(SIGNALS) - 1)
    alpha_corrected = bonferroni_alpha(n_tests)
    print(f"grid: {len(factories)} arms x {len(betas)} betas = "
          f"{len(factories) * len(betas)} cells, {n_tests} paired tests -> "
          f"Bonferroni alpha = 0.05/{n_tests} = {alpha_corrected:.5f}")

    for arm_label, factory in factories.items():
        t0 = time.time()
        for beta in betas:
            data = collect(conversations, factory, beta)
            report_cell(f"[A: IV.1 replication] {arm_label}", beta, data,
                       draws=draws, generator=generator,
                       alpha_corrected=alpha_corrected)
            if arm_label == "baseline" and beta == 128.0:
                check_reproduction(data, IV1_PUBLISHED_AUC_AT_BETA_128)
        print(f"  ({arm_label}: {time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------
# Protocol B -- full system (LSA + cortex-derived keys + salience priors)
# NOT comparable to IV.1 cell-for-cell; carries the DG negative control.


def collect_grid(
    conversations, *, dim: int, key_mode: HippocampalKey, whiten: bool,
    betas: tuple[float, ...],
) -> dict[float, dict]:
    """Build one separation arm ONCE; score every beta's abstention signals
    against that single store.

    This is Protocol B's write path -- LSA embeddings
    (`separation_beta_sweep.build_embedder`), a full `OrganizationalMemory`
    with a cortex-derived key (`build_system`), and time-accurate salience
    priors (`store.refresh_priors`). It is NOT IV.1's write path (see the
    module docstring) and exists so the DG negative control -- which needs
    `KeyEncoder`, absent from Protocol A's bare-MHN store -- has somewhere to
    run.

    Mirrors `separation_beta_sweep.sweep_arm`'s cost structure for the same
    reason it does there: beta only touches the read path (`mhn.logits` etc.
    all take an explicit `beta` override), so the LSA fit, whitener fit,
    per-conversation cortex fit and key encoding -- the expensive part -- do
    not need to be repeated per beta. Correctness of that shortcut is pinned
    for the recall grid by `test_separation_beta_sweep.py`.

    Returns one cell per beta, each shaped exactly like
    `abstention.collect()`'s return value (`{"columns", "labels", "conv"}`),
    so every downstream statistic in `experiments/abstention.py` -- AUC with
    CI, paired delta, out-of-fold logistic -- is reused unmodified rather than
    reimplemented against a new shape.
    """
    corpus = [t.memory_text for c in conversations for t in c.turns]
    embedder = build_embedder(corpus, dim=dim, whiten=whiten)

    cells: dict[float, dict] = {
        b: {"columns": {s: [] for s in SIGNALS}, "labels": [], "conv": []}
        for b in betas
    }
    lengths: list[float] = []

    for c, conv in enumerate(conversations):
        system = build_system(embedder, conv, key_mode=key_mode)
        store = system.store
        mhn = store.mhn

        ids: set[str] = set()
        for turn in conv.turns:
            embedding = embedder.encode([turn.memory_text])[0]
            if float(embedding.norm()) < 1e-8:
                continue  # degenerate text; the store rejects these
            latent = system.cortex.latent(embedding)
            key = system.key_encoder(embedding, latent)
            record = MemoryRecord(
                text=turn.memory_text,
                embedding=embedding,
                latent=latent,
                key=key,
                persistence=Persistence.TEMPORAL,
                created_at=turn.timestamp,
                last_reinforced_at=turn.timestamp,
            )
            store.add(record, now=turn.timestamp)
            ids.add(turn.dia_id)

        if len(store) == 0:
            continue
        questions = [q for q in conv.questions if any(e in ids for e in q.evidence)]
        if not questions:
            continue

        # `now` is the same instant for every question in this conversation,
        # so refreshing priors once (rather than per question, as the recall
        # sweep does defensively) is the identical computation for less work.
        now = max(t.timestamp for t in conv.turns)
        store.refresh_priors(now)

        for q in questions:
            xi = system.retrieval.encode_cue(q.question)
            label = int(q.category != ADVERSARIAL)
            lengths.append(float(len(q.question)))

            # Beta-invariant: plain cosine similarity of the cue to the
            # stored patterns does not depend on temperature. Computed once
            # per question, outside the beta loop below, rather than
            # redundantly per beta.
            sims = mhn.patterns @ xi
            top2 = torch.topk(sims, min(2, sims.numel())).values
            cos_top1 = float(top2[0])
            cos_margin = float(top2[0] - top2[1]) if top2.numel() > 1 else 0.0

            for beta in betas:
                attn = mhn.attention(xi, beta)
                # Entropy of the retrieval attention: a peaked distribution
                # means one memory explains the cue, a flat one means none
                # does. Same construction as `abstention.collect`.
                entropy = float(-(attn * attn.clamp_min(1e-30).log()).sum())
                report = basin_depth(mhn, xi, beta)
                settled = mhn.step(xi, beta)

                cols = cells[beta]["columns"]
                cols["cos_top1"].append(cos_top1)
                cols["cos_margin"].append(cos_margin)
                cols["neg_attn_ent"].append(-entropy)
                cols["log_density"].append(float(log_density(mhn, xi, beta)))
                cols["neg_depth_nats"].append(-report.depth_nats)
                cols["neg_settle"].append(-float((settled - xi).norm()))
                cells[beta]["labels"].append(label)
                cells[beta]["conv"].append(c)

    for cell in cells.values():
        cell["columns"]["length"] = list(lengths)

    return cells


def run_protocol_b(
    conversations, *, dim: int, betas: tuple[float, ...], draws: int,
    generator: torch.Generator, arm_filter: list[str] | None,
) -> None:
    """Full system: LSA + cortex-derived keys + salience priors. Carries the
    DG negative control; NOT comparable cell-for-cell to Protocol A / IV.1."""
    print("\n==================== PROTOCOL B: full system (SECONDARY, "
          "NOT comparable to IV.1) ====================")
    print("Different write path (LSA embeddings, cortex-derived key, "
          "salience priors) than IV.1's bare-MHN hybrid store. Exists to "
          "carry the DG negative control, which needs a key encoder.")

    arms = [a for a in ARMS if arm_filter is None or a[0] in arm_filter]
    if not arms:
        raise SystemExit(f"--arms matched nothing in {[a[0] for a in ARMS]}")

    n_tests = len(arms) * len(betas) * (len(SIGNALS) - 1)
    alpha_corrected = bonferroni_alpha(n_tests)
    print(f"grid: {len(arms)} separation arms x {len(betas)} betas = "
          f"{len(arms) * len(betas)} cells, {n_tests} paired tests -> "
          f"Bonferroni alpha = 0.05/{n_tests} = {alpha_corrected:.5f}")

    for label, key_mode, whiten in arms:
        t0 = time.time()
        cells = collect_grid(conversations, dim=dim, key_mode=key_mode,
                             whiten=whiten, betas=betas)
        # Confound check first, same device as abstention.py: if question
        # length alone separates the classes the rest of this arm is suspect.
        # beta-independent (length never depends on beta), so one cell speaks
        # for the whole arm.
        first = cells[betas[0]]
        clen, llo, lhi = auc_with_ci(first["columns"]["length"], first["labels"],
                                     draws=draws, generator=generator)
        print(f"\n[B: full system] {label}: question-length confound AUC "
              f"{clen:.3f} [{llo:.3f}, {lhi:.3f}]")
        for beta in betas:
            report_cell(f"[B: full system] {label}", beta, cells[beta],
                       draws=draws, generator=generator,
                       alpha_corrected=alpha_corrected)
        print(f"  ({label}: {time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------
# shared reporting -- both protocols hand `report_cell` the same cell shape


def _tensor(values: list[float]) -> Tensor:
    return torch.tensor(values, dtype=torch.double)


def report_cell(
    label: str, beta: float, cell: dict, *, draws: int, generator: torch.Generator,
    alpha_corrected: float,
) -> None:
    cols = cell["columns"]
    labels = cell["labels"]
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"\n  {label} @ beta={beta:g}  (n={n}: {n_pos} answerable, "
          f"{n_neg} adversarial)")
    if n_neg == 0 or n_pos == 0:
        print("    no adversarial (or no answerable) questions resolved -- skipped")
        return

    for name in SIGNALS:
        point, lo, hi = auc_with_ci(cols[name], labels, draws=draws, generator=generator)
        keep, reject = operating_point(cols[name], labels)
        if name == "cos_top1":
            rho_str = "  (baseline)"
        else:
            rho = spearman_rho(_tensor(cols[name]), _tensor(cols["cos_top1"]))
            rho_str = f"  rho vs cos_top1 {rho:+.4f}"
        print(f"    {name:<16} AUC {point:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"rejects {reject:.1%} @ {keep:.0%} coverage{rho_str}")

    print("    paired vs cos_top1 (95% CI, then Bonferroni-corrected "
          f"alpha={alpha_corrected:.5f} CI):")
    for name in SIGNALS:
        if name == "cos_top1":
            continue
        delta, lo, hi = paired_auc_delta(cols["cos_top1"], cols[name], labels,
                                         draws=draws, generator=generator)
        verdict = "TIE" if lo <= 0.0 <= hi else ("WINS" if delta > 0 else "LOSES")
        _, blo, bhi = paired_auc_delta_ci(
            cols["cos_top1"], cols[name], labels, alpha=alpha_corrected,
            draws=draws, generator=generator,
        )
        survives = "SURVIVES" if not (blo <= 0.0 <= bhi) else "does not survive"
        print(f"      {name:<16} delta {delta:+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict:<5}  "
              f"corrected [{blo:+.3f}, {bhi:+.3f}] {survives}")

    base = loco_auc(cell, ["cos_top1"])
    energy = loco_auc(cell, ["cos_top1", "log_density", "neg_depth_nats"])
    every = loco_auc(cell, SIGNALS)
    print(f"    out-of-fold: cosine only {base:.3f}  cosine+energy {energy:.3f}  "
          f"everything {every:.3f}  (energy adds {energy - base:+.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("locomo", "qmsum"), default="locomo")
    parser.add_argument("--locomo", type=int, default=10)
    parser.add_argument("--dim", type=int, default=1024,
                        help="Protocol B's LSA dimension (Protocol A's hybrid "
                             "embedder ignores this -- it uses IV.1's fixed "
                             "4096-d hashed lexical half + BGE's native dim)")
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--arms", type=str, nargs="+", default=None,
                        help="subset of Protocol B arm labels to run "
                             "(default: all three; does not affect Protocol A)")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--protocol", choices=("a", "b", "both"), default="both",
                        help="run only the IV.1 replication (a), only the full "
                             "system (b), or both (default)")
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)

    if args.corpus == "qmsum":
        print("QMSum has no analogue of LoCoMo category 5: every QMSum question "
              "is built from a real, resolvable relevant_text_span and is "
              "labelled category=1 unconditionally (see experiments/qmsum.py). "
              "There is no false-presupposition / deliberately-empty-answer "
              "set to abstain on. Rather than inventing a negative class "
              "(e.g. treating unresolved evidence as 'adversarial', which "
              "would measure corpus coverage, not abstention), this script "
              "refuses to score QMSum for abstention. Use --corpus locomo.")
        return

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    conversations = locomo.load()[: args.locomo]
    betas = tuple(args.betas)
    print(f"{args.corpus}: {len(conversations)} conversations, "
          f"{threads} threads")
    print("!! single-corpus run; no cross-corpus overfitting guard is applied "
          "to these numbers on their own")

    if args.protocol in ("a", "both"):
        run_protocol_a(conversations, betas=betas, draws=args.draws, generator=generator)
    if args.protocol in ("b", "both"):
        run_protocol_b(conversations, dim=args.dim, betas=betas, draws=args.draws,
                       generator=generator, arm_filter=args.arms)


if __name__ == "__main__":
    main()
