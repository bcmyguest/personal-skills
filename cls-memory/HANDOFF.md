# Handoff — CLS organizational memory

For whoever picks this up next. Written at commit `67f98cc` on branch
`claude/cls-organizational-memory-poc-s9afk0`.

**The one open task is recall.** Everything else is either done, measured, or
explicitly listed below as a known limitation.

- **§3A is the ordered plan** — priorities, blockers, critical path. Start there.
- §3 is the reasoning behind it; §1–2 are the context you need to not redo work.
- §5 is the failure pattern this project keeps repeating. Read it before
  trusting your own fixes.

---

## 0. Orientation

```bash
cd cls-memory
uv venv && uv pip install torch pytest        # HuggingFace is unreachable here
uv run pytest                                  # 131 pass, ~60s
PYTHONPATH=. uv run python examples/demo.py    # full lifecycle

# the two real datasets (gitignored, ~5.5 MB total)
curl -sLo experiments/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl -sLo experiments/data/qmsum_test.jsonl \
  https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/test.jsonl
```

Read in this order: `README.md` (design + §11–13 review history), `RESULTS.md`
(every measured number, Parts I–III), then this file.

Key scripts:

| script | what it answers | runtime |
|---|---|---|
| `experiments/recall_ablation.py` | where in the pipeline recall is lost | ~2 min |
| `experiments/recall_check.py` | did a change help, on **both** corpora | ~2 min |
| `experiments/benchmark.py` | synthetic corpus, all 7 mechanisms | ~8 min |
| `experiments/benchmark_locomo.py` | real-data retrieval / gate / decay | ~3 min |
| `experiments/significance.py` | **confounded, do not trust** (see §4) | — |

---

## 1. What is solid — do not re-litigate

Three independent adversarial reviews checked these. Re-deriving them is waste.

**Mathematics, verified numerically:**
- Ramsauer eq. 4 with an added per-pattern log-prior; reduces to the paper's
  form exactly under a uniform prior (difference 0.0).
- CCCP monotone energy descent, including under non-uniform priors *and* under
  masking (max increase 1.9e-6 over 300 trials).
- `exp(-βE)` is exactly the Gaussian mixture at σ²=1/β, at any pattern scale —
  this required folding `-β‖xᵢ‖²/2` into the logits instead of the paper's
  global `+½M²`.
- `log_density` integrates to 0.9999999999999575 on a 2-D grid.
- Tweedie/Miyasawa: one MHN update **is** one exact denoising step (agreement
  5e-16 in float64).
- ULA divergence bound `η > 4/β`, and the step caps' effect on the stationary
  distribution (0.05766 measured vs 0.05774 predicted).

**Consolidation** was validated by a held-out control, not by its authors'
own metric: surprise falls on stored episodes (0.902 → 0.530) while held-out
routine surprise *rises* ~2.8× in **both** arms, so the pruning criterion is
not simply getting easier. The store drains 21 → 0 over four passes at
`episodic_ratio=0.125`, against 21 → 21 at ratio 0.

**Design decisions that were measured, not guessed** (all in `config.py`
docstrings with numbers):
- `kl_weight=0.01` — at 1.0 the posterior collapses and the novelty signal
  drops from 0.57 to 0.08 separation.
- `beta=128` — at the old default of 8, *every query settled onto one global
  mixture* (six cues → six states at pairwise cosine 1.0000) while reporting
  `converged=True`. This was the single largest recall defect.
- `key=EMBEDDING` — dentate-gyrus separation beats the VAE latent but loses to
  the raw embedding on real text (0.121 vs 0.150).
- `input_dim=1024` — dimension is the binding constraint on the embedder.

---

## 2. Where recall actually stands

Measured on 3 LoCoMo conversations (1451 turns, 494 questions) and 6 QMSum
meetings (3473 turns, 38 questions). Full tables in `RESULTS.md` Part III.

| | recall@1 | recall@5 |
|---|---|---|
| LoCoMo, current defaults | **0.250** | 0.318 |
| QMSum, current defaults | **0.395** | 0.632 |
| LoCoMo, shipped defaults before the fixes | 0.004 | 0.012 |

**The pipeline is no longer the problem.** At β=128 with `key=EMBEDDING` the
full pipeline scores 0.172@1 against its embedder's own plain-kNN ceiling of
0.174@1. The memory system adds essentially no loss. *All* remaining headroom
is in the embedding.

**Retrieval ceilings** (plain kNN over turns, no memory system, LoCoMo):

| ranking | recall@1 | recall@5 |
|---|---|---|
| **TF-IDF cosine, sparse, no reduction** | **0.320** | **0.575** |
| BM25 | 0.298 | 0.526 |
| LSA-1024 (current default) | 0.255 | 0.528 |
| LSA-256 | 0.174 | 0.397 |
| hashing-256 (old default) | 0.077 | 0.176 |

A bag-of-ngrams from the 1970s beats the current dense embedding by 26%
relative. That gap is the task.

---

## 3. The task: close the gap to the sparse ceiling

### 3.1 The leading hypothesis — CONFIRMED on a smoke test, needs the full run

**Random projection should beat truncated SVD at equal dimension.**

Truncated SVD keeps the top-k directions and *discards the tail entirely*. For
short-query retrieval the discriminative signal lives in rare terms, which are
exactly the tail. A Johnson–Lindenstrauss random projection instead preserves
*all* directions approximately, with distortion `O(√(log n / d))`. At d=1024
that should land much closer to the sparse ceiling than SVD does.

Supporting evidence already in hand: the third review measured the randomised
SVD's per-component agreement with an exact SVD at |cos| ≈ 1.00 for the leading
components but as low as 0.07 in the tail (since fixed by re-orthonormalising —
see `embeddings.py`), and LSA recall rises steeply with dimension
(0.174 → 0.255 from 256 → 1024), which is what you expect if truncation is the
loss mechanism.

**Cheapest implementation** — signed feature hashing, no V×d matrix:
for each term, hash to `k` output indices with random signs, accumulate
`tfidf_weight * sign`. `k=2..4`. Memory-light, fast, no fitting beyond the
existing vocabulary/IDF pass, and it drops straight into the existing
`Embedder` protocol (`dim`, `encode`) with **no changes anywhere else**.

**The ceiling rows now exist** (`HashedProjection` in
`experiments/recall_ablation.py`, already satisfying the `Embedder` protocol).
On a **one-conversation** smoke test — 419 turns, so treat it as indicative:

| ranking | recall@1 | recall@5 |
|---|---|---|
| LSA-1024 (current default) | 0.250 | 0.515 |
| TF-IDF sparse (the "ceiling") | 0.311 | 0.582 |
| random-projection-1024 | 0.286 | 0.520 |
| random-projection-2048 | 0.306 | 0.546 |
| **random-projection-4096** | **0.327** | 0.556 |
| **BM25-weighted RP-4096** | **0.332** | 0.571 |

Random projection does not merely approach the sparse ceiling — at 4096 it
**passes** it, and BM25 term weighting adds a little more. This is what the
hypothesis predicted: SVD discards the tail, JL preserves it.

### 3.1b Confirmed on the full set — and the bag-of-words question, correctly

Full 3-conversation LoCoMo run, 494 questions.

**Lexical (bag of n-grams):**

| ranking (kNN ceiling) | @1 | @5 | @10 |
|---|---|---|---|
| LSA-1024 (current default) | 0.251 | 0.526 | 0.609 |
| TF-IDF sparse | 0.320 | 0.575 | 0.648 |
| random-projection-4096 | 0.322 | 0.555 | 0.630 |
| **BM25-weighted RP-4096** | **0.330** | 0.581 | 0.640 |

**Semantic (spaCy `en_core_web_lg`, 343k vectors) — pooling matters enormously:**

| pooling | @1 | @5 |
|---|---|---|
| mean, raw vectors | 0.196 | 0.399 |
| mean, L2-normalised tokens | 0.168 | 0.360 |
| IDF-weighted, L2 tokens | 0.267 | 0.474 |
| **SIF** (Arora et al. 2017) | **0.275** | 0.498 |
| SIF + first-PC removal | 0.273 | **0.514** |

**Hybrid (concatenate L2-normalised lexical and semantic, weight by alpha):**

| alpha (lexical share) | @1 | @5 | @10 |
|---|---|---|---|
| 0.8 | 0.332 | 0.587 | 0.646 |
| 0.6 | 0.358 | 0.599 | 0.666 |
| 0.5 | 0.358 | 0.593 | **0.682** |
| **0.4** | **0.360** | **0.603** | 0.662 |

**The correction that matters.** An earlier version of this section claimed
"moving off bag-of-words is a large regression", from a semantic score of 0.121.
That was wrong, and it was wrong for two avoidable reasons: it used
`en_core_web_md` (20k shared vector rows rather than lg's 343k unique) and a
naive unweighted mean. Pooled properly the same idea scores 0.275 — more than
double, and above the current LSA default. **Do not conclude anything about
representations from a naive mean of word vectors.** Token coverage was 99.8%
for both models, so coverage was never the issue; weighting was.

Best configuration in the project is the hybrid at alpha 0.4-0.5:
**0.360@1, 0.603@5, 0.682@10**, against the current default's 0.251@1. Note the
optimum is *semantic-dominant*, the opposite of what the flawed test suggested.

**A real transformer: BGE-small-en-v1.5 — reachable after all.** Qdrant's
fastembed mirror on `storage.googleapis.com` serves the ONNX weights, and that
host is *not* proxy-blocked. An earlier probe here recorded it as blocked; that
probe used a wrong object key (the objects are prefixed `fast-`) and GCS
answered 403, which was misread as a policy denial. HuggingFace, hf-mirror,
cdn-lfs, ModelScope, Kaggle, gitee, Stanford NLP, fbaipublicfiles and GitHub
LFS genuinely are blocked; PyPI, non-LFS `raw.githubusercontent.com`, GitHub
*release* assets and `storage.googleapis.com` are not.

`BGEEmbedder` (in the ablation) drives ONNX + `tokenizers` directly, so neither
`fastembed` nor `transformers` is needed at run time, and it fetches the weights
on first use.

| ranking (kNN ceiling) | @1 | @5 | @10 |
|---|---|---|---|
| BM25-weighted RP-4096 (best lexical) | 0.330 | 0.581 | 0.640 |
| BGE-small, with query prefix | 0.269 | 0.543 | 0.676 |
| BGE-small, no query prefix | 0.257 | 0.547 | 0.654 |
| hybrid RP-4096 + spaCy SIF (alpha=0.4) | 0.360 | 0.603 | 0.662 |
| **hybrid RP-4096 + BGE (alpha=0.5)** | **0.393** | **0.644** | **0.749** |
| hybrid RP-4096 + BGE (alpha=0.3) | 0.322 | 0.601 | 0.723 |

Three things worth internalising:

1. **BGE alone loses to lexical at recall@1** (0.269 vs 0.330) while *winning*
   at recall@10 (0.676 vs 0.640). That is the expected shape for entity-heavy
   short-query retrieval — LoCoMo questions name specific people, dates and
   events, and exact matching is hard to beat at rank 1. A dense encoder is not
   a drop-in replacement for lexical retrieval on this task.
2. **The hybrid is decisively best**: 0.393@1 and 0.749@10 against the current
   default's 0.251@1 / 0.609@10 — +57% relative at rank 1. The two halves fail
   differently, which is exactly why combining them pays.
3. **The query prefix matters** (0.269 vs 0.257). BGE v1.5 is asymmetric: the
   instruction goes on queries only, never on documents.

**Revised task 1.1:** promote the lexical + BGE hybrid.
1. Move `HashedProjection` into `cls_memory/embeddings.py` (BM25 weighting,
   dim 4096).
2. Move `BGEEmbedder` alongside it. Keep the weights download lazy and
   `onnxruntime`/`tokenizers` **optional** — the lexical half must still work
   without them, since the GCS mirror may not be reachable everywhere.
3. Move `HybridEmbedder`, default `alpha=0.5`.
4. Re-measure on **both** corpora, add tests, and check the footprint:
   4096+384 dense dims per memory is ~4.4x the current 1024. If that is too
   much, test RP-1024 + BGE — RP-1024 alone scores 0.306 against RP-4096's
   0.322, so the lexical half may not need full width once BGE carries
   recall@10.
5. `SpacyVectorEmbedder` is superseded and can be dropped, unless you want a
   fallback with a different reachability profile (spaCy models come from
   GitHub releases, BGE from GCS).

### 3.2 The fallback, and probably the better system anyway

**Hybrid sparse-shortlist + Hopfield rank.** Use BM25 or TF-IDF to shortlist
~100 candidates, then let the Hopfield layer settle over only those. This
almost certainly beats any pure-dense option because it starts from the 0.320
ceiling rather than a compressed approximation of it.

Architecturally heavier: the MHN currently holds every memory as a dense row,
so a shortlist means either building a transient sub-network per query, or
masking the logits (`hippocampus.logits` returns `(N,)` — a `-inf` mask over
non-candidates is the cheapest route and preserves the energy semantics for the
surviving set). Prefer the mask; it keeps `retrieve`/`energy`/`basin_depth`
working unchanged.

Watch out: a masked logit vector changes the normalising constant, so
`log_density` and `basin_depth` become conditional on the shortlist. Either
document that or compute the basin report on the unmasked set.

### 3.3 Cheaper things worth trying, roughly in order

- **Context windows.** LoCoMo evidence is single turns, and a turn like
  "Yes, definitely" is unretrievable alone. Index `turn ± 1` as the embedded
  text while keeping per-turn identity. Untested; plausibly a large win on
  dialogue, possibly a loss on QMSum where evidence is a contiguous span.
- **Query-side asymmetry.** Questions and turns are different registers.
  BM25's term saturation and length normalisation exist for this; the current
  cosine has neither.
- **`max_iter=1` for free cues.** One step is the Tweedie denoiser and was
  never worse in the reviewer's probes (5/6 at every β); iterating only matters
  for masked cues. Cheap latency win, possibly a small accuracy one.

### 3.4 The single biggest unknown

**Nothing has been tested with a real sentence encoder.** HuggingFace is
unreachable from this environment, so `SentenceTransformerEmbedder` exists and
is untried. If you have network access, run `experiments/recall_ablation.py`
with it *first* — it may make 3.1 and 3.2 irrelevant, or reveal that the
pipeline has a ceiling nobody has hit yet. Every retrieval number in this repo
is a floor.

### 3.5 How to know you succeeded

Run `experiments/recall_check.py` — it evaluates on **both** corpora because a
change that helps one and hurts the other is overfitting, and that has already
happened once here (`key=EMBEDDING` cost QMSum recall@5 0.526 → 0.421 while
helping recall@1). Beating 0.320@1 on LoCoMo means you have passed the sparse
ceiling; anything at or above ~0.30 is a real result.

---

## 3A. Ordered plan, with blockers

Priority is expected-value-per-effort, but **the ordering below is driven by
dependencies, not by size**. Two rules produce most of it:

1. *Fix the measurement before optimising against it.* Two of the numbers you
   would judge success by are currently too weak to judge with.
2. *Change one variable at a time.* The embedder question must settle before
   anything that changes what gets indexed, or you cannot attribute the result.

### P0 — make the verdict trustworthy (do first, ~half a day)

| # | Task | Blocks | Blocked by | Why first |
|---|---|---|---|---|
| 0.1 | Widen the QMSum slice well beyond 6 meetings / 38 questions | every recall verdict | — | 38 questions cannot separate a real gain from noise, and QMSum is half the overfitting guard. Cheap: `qmsum.load(max_meetings=...)`, 35 meetings and 244 queries are already in the file. |
| 0.2 | Add random-projection and BM25-weighted rows to the **ceilings** section of `recall_ablation.py` | 1.1, 1.2 | — | Decides which of the two recall routes is worth building, without writing production code. A ceiling row is ~20 lines. |
| 0.3 | If network allows, add a `SentenceTransformerEmbedder` ceiling row | 1.1, 1.2 (may moot both) | network access to HuggingFace | Biggest single unknown. Could make the dense-embedding work irrelevant, or reveal a pipeline ceiling nobody has hit. Do not block on it — it may never be available here. |

### P1 — the recall work (gated on P0)

| # | Task | Blocks | Blocked by | Notes |
|---|---|---|---|---|
| 1.1 | Random-projection embedder (signed feature hashing over TF-IDF) | — | 0.2 | Do **only if** 0.2 shows RP-1024 clearly beating LSA-1024's 0.255@1. Drops into the `Embedder` protocol with no other changes — lowest-risk route to the 0.320 ceiling. |
| 1.2 | Hybrid sparse shortlist + masked Hopfield logits | — | 0.2 | The higher-ceiling option and the likely end state, because it *starts* from 0.320 rather than approximating it. Heavier: masking changes the normalising constant, so `log_density`/`basin_depth` become shortlist-conditional. Do this if 1.1 underdelivers, or straight away if 0.2 shows RP is not enough. |

**1.1 and 1.2 are alternatives, not a sequence.** Pick from 0.2's numbers. Doing
both is only worth it if 1.1 lands close to the ceiling and you want the last
few points.

### P2 — compose on top (only after P1 settles)

| # | Task | Blocked by | Notes |
|---|---|---|---|
| 2.1 | `max_iter=1` for free cues | — (independent) | The one item with no dependencies at all — a freebie, safe to land any time. One step is the Tweedie denoiser and was never worse in probing. |
| 2.2 | Context windows (index `turn ± 1`, keep per-turn identity) | 1.1/1.2 | Changes *what is indexed*, so measuring it against a moving embedder confounds both. Plausibly large on LoCoMo dialogue, plausibly negative on QMSum spans — 0.1 matters here. |
| 2.3 | Query-side asymmetry (BM25 saturation + length norm) | 0.2 | Partly absorbed by 0.2 if you add the BM25-weighted row; whatever is left is a small refinement. |

### P3 — correctness and evaluation debt (does not block recall)

| # | Task | Blocks | Notes |
|---|---|---|---|
| 3.1 | Fix `significance.py`'s confound (deep-copy the store per arm, counterbalance arm order) | any statistical claim about `episodic_ratio` | The consolidation trade-off currently has **no** statistical backing. Not on the recall path, but it is the one place where a quoted number would be unsupported — which is why none is quoted. |
| 3.2 | Control the novelty gate for turn length (`corr = −0.48`) | trusting the gate on real text | Separate mechanism from retrieval. Matters for what *enters* the store, not what comes back. |
| 3.3 | Make `half_life_days` a per-source setting | realistic multi-source deployments | Currently global and wrong for anything on a multi-month cadence. |

### P4 — deployment blockers (out of scope for recall, in scope before any pilot)

| # | Task | Notes |
|---|---|---|
| 4.1 | Per-memory ACLs enforced **before** retrieval | Filtering after settling leaks information through the attractor. This is a correctness requirement, not a feature. |
| 4.2 | Persistence beyond `state_dict` | Records are plain dataclasses holding tensors; serialisation is straightforward but unwritten. |
| 4.3 | Honour `MemorySystemConfig.device` | Declared and ignored; ancillary tensor construction in `hippocampus`/`energy` is CPU-bound. |
| 4.4 | Housekeeping: drop the `record.key` duplicate, keep capacity across `remove()`, batch `basin_depth`, move `occlude` off the read path | All cosmetic or constant-factor. Cheapest last. |

### Critical path

```
0.1 (widen QMSum) ──┐
                    ├──▶ 0.2 (ceilings) ──▶ 1.1 or 1.2 ──▶ 2.2 ──▶ done
0.3 (encoder, if network) ──┘
```

Everything in P3 and P4 is off this path and can proceed in parallel or later.
2.1 is off it too and can land whenever.

**If you only have an hour:** do 0.2. It tells you which of the two real options
to build, and it is the difference between choosing on evidence and choosing on
taste.

---

## 4. Known limitations — accepted, not bugs to rediscover

- **`experiments/significance.py` is confounded.** Its arms share a store that
  `consolidate()` mutates via `reindex`, and each arm consumes a different
  amount of the global RNG stream. Both the drift (H1/H2) and pruning (H3) arms
  are affected. **No number from it is quoted anywhere**, deliberately. To fix:
  deep-copy and restore the whole store inside `drift_after`, and counterbalance
  arm order across seeds.
- **QMSum has only 38 questions** in the 6-meeting slice. Directions agree with
  LoCoMo at every step; absolute values are indicative only. Widen it if you
  need to lean on QMSum alone.
- **The novelty gate is weak on real text** — 13% separation between held-out
  and foreign-conversation turns (vs 13× on synthetic), and
  `correlation(surprise, turn length) = −0.48`, so it is substantially a length
  filter. Control for length before trusting it.
- **`half_life_days=30` is domain-specific.** On LoCoMo's 231-day spans it puts
  13.8% of the corpus below the prune floor and costs 1.6 points of recall.
- **CPU only** — `MemorySystemConfig.device` is declared and ignored. Ancillary
  tensor construction in `hippocampus`/`energy` does not honour device.
- `basin_depth` is single-query; `PatternCompleter.occlude` is evaluation
  tooling sitting on the read path; `record.key` duplicates its pattern row
  (dropping it for a view would halve key memory); `remove()` releases capacity
  so the next write reallocates; a `patterns` view held across a growing `write`
  silently detaches (documented on the property).
- **No persistence layer beyond `state_dict`, no access control, single-tenant.**
  An organisational deployment needs per-memory ACLs enforced *before*
  retrieval — filtering after settling leaks information through the attractor.

---

## 5. Traps — read this before trusting your own work

This project has a consistent failure pattern, and you will hit it too.

**Three separate times, a test written to verify a fix was itself wrong**, and
each time it hid a real defect:
- `test_reindex_is_atomic` passed a wrong-*width* key, caught by a dim check
  *before* the removal, so the rollback it existed to test never executed.
- `test_lsa_fit_uses_a_sparse_matrix` asserted nothing about sparsity and would
  have passed against the dense implementation.
- The first capacity-buffer fix read capacity from the live view instead of the
  allocated buffer and was still O(N²) — only a timing test caught it.

Treat "my fix + my test" as one unreviewed unit. Where you can, verify by a
route you did not use to build the thing — the held-out control that finally
validated consolidation is the model to copy.

**Every headline number produced by a defective default went unnoticed because
every experiment overrode it.** `beta=8` collapsed retrieval completely, and
nobody saw it because `demo.py`, `benchmark.py` and `benchmark_locomo.py` all
set `beta=32`. If you add a config knob, make at least one measurement run at
the actual default.

**Synthetic data flattered every mechanism.** Recall@1 was 1.000 on the
template corpus and 0.121 on real dialogue with identical code. AUC was 1.0000
for the novelty gate. Do not let a synthetic result stand as evidence; both real
corpora are wired up and cheap to run.

**Silent corruption outranks crashes.** The worst three defects found here all
reported success while returning wrong answers: `converged=True` on a collapsed
attractor, `improved=True` while training on 1e6-magnitude garbage, and
`load_state_dict` restoring zero padding as memories that took 100% of the
attention mass. Prefer probes that check *values*, not shapes and exit codes.
