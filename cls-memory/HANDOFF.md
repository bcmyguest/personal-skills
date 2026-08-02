# Handoff — CLS organizational memory

For whoever picks this up next. Written at commit `67f98cc` on branch
`claude/cls-organizational-memory-poc-s9afk0`.

**The one open task is recall.** Everything else is either done, measured, or
explicitly listed below as a known limitation. Section 3 is the actual work;
sections 1–2 are the context you need to not redo it.

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

### 3.1 The leading hypothesis (untested — this is where I stopped)

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

**Test it as a ceiling first**, before writing production code — add rows to
`experiments/recall_ablation.py` under "RETRIEVAL CEILINGS":
random projection at 1024 / 2048 / 4096, and a BM25-weighted variant. If RP-1024
does not clearly beat LSA-1024's 0.255, abandon this and go to 3.2.

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
