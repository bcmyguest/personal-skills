# Benchmark Results

Measured on a labelled synthetic corpus, not asserted. Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python experiments/benchmark.py          # full
PYTHONPATH=. .venv/bin/python experiments/benchmark.py --quick  # ~1 min
```

**Corpus**: 2000 routine items from 12 templates, 60 anomalies from 12 templates,
6 evergreen rules. 80/20 train/held-out split. Embeddings are the offline
`HashingEmbedder` (256-d); cortex 256→192→96→32, `kl_weight=0.01`, 60 epochs;
β=32, DG key 2048-d at k=256. Seed 0 throughout.

Training the schema on 1600 documents took 179s on CPU; final VAE loss 0.218.

---

## 1. Novelty gate — ROC AUC 1.0000

| population | n | mean surprise | median | p90 |
|---|---|---|---|---|
| train routine | 1600 | 0.0939 | 0.0934 | 0.1693 |
| held-out routine | 400 | 0.1020 | 0.1049 | 0.1763 |
| **anomaly** | 60 | **1.4038** | 1.3783 | 1.5574 |
| evergreen rule | 6 | 1.4441 | 1.5046 | 1.6038 |

Reconstruction error separates anomalies from routine text **perfectly**
(AUC 1.0000) — a >13× gap in mean surprise. Note this is an *upper bound*, not
a realistic field estimate: the corpus has only 12 routine templates, so the
schema is unusually learnable. Treat it as evidence the mechanism works, not as
a predicted production number.

**Confound controlled.** Anomalies carry unique incident references, which are
unseen tokens — so some of the measured novelty could be "contains a new
number" rather than "violates the schema". Re-scored against anomalies
generated *without* references: mean surprise 1.3744 vs 1.4038, AUC 1.0000 vs
1.0000. The effect is negligible; detection is structural.

### The default threshold is too permissive

| quantile | threshold | precision | recall | F1 |
|---|---|---|---|---|
| **0.999** | 0.2307 | **1.000** | **1.000** | **1.000** |
| 0.990 | 0.2039 | 0.870 | 1.000 | 0.930 |
| 0.975 | 0.1920 | 0.779 | 1.000 | 0.876 |
| 0.950 | 0.1830 | 0.682 | 1.000 | 0.811 |
| 0.900 | 0.1693 | 0.500 | 1.000 | 0.667 |
| 0.850 | 0.1567 | 0.432 | 1.000 | 0.603 |

At the configured 0.95 the gate admitted 17 false positives out of 400 routine
items (P=0.779, R=1.000). Recall saturates at 1.0 everywhere, so the entire
operating-point decision is about precision — i.e. how much routine noise you
are willing to let into the hippocampus. **With a well-trained schema, use
0.99–0.999.** A permissive quantile silently converts the hippocampus into an
undifferentiated log.

## 2. Key representation — the latent really is the wrong key

50% contiguous cues, scored only on cues that admit a single answer:

| key | dim | recall@1 | recall@3 | max off-diag cos | mean off-diag |
|---|---|---|---|---|---|
| **separated (DG)** | 2048 | **1.000** | 1.000 | 0.776 | 0.246 |
| embedding | 256 | **1.000** | 1.000 | 0.888 | 0.240 |
| latent | 32 | 0.364 | 0.455 | 0.961 | 0.367 |

Confirms the finding at 60 anomalies that was first seen at 5: the VAE latent
fuses distinct novel episodes (max pairwise cosine 0.961) and loses roughly
two-thirds of retrievals. DG separation and raw embeddings are both perfect
here; DG additionally gives an explicit separation knob.

## 3. Retrieval — graceful degradation

| cue fraction | recall@1 | recall@3 | recall@1 (unambiguous cues) | n unambiguous |
|---|---|---|---|---|
| 1.00 | 1.000 | 1.000 | 1.000 | 60 |
| 0.75 | 0.983 | 1.000 | 1.000 | 58 |
| 0.50 | 0.767 | 0.917 | 1.000 | 33 |
| 0.35 | 0.600 | 0.883 | 0.969 | 32 |
| 0.25 | 0.500 | 0.817 | 1.000 | 23 |

The overall recall@1 decline is **almost entirely cue ambiguity, not retrieval
failure**: at 25% of the text, only 23 of 60 cues still appear in exactly one
stored memory, and on those the system is essentially perfect. Reporting only
the aggregate number would have understated the mechanism by half.

Confabulation control — out-of-distribution queries are correctly flagged:

| query | basin depth | flagged |
|---|---|---|
| "quantum chromodynamics lattice gauge" | 0.719 | yes |
| "the marmalade recipe calls for seville oranges" | 0.697 | yes |

Caveat found in review: the *ranking* is reliable, the absolute cutoff is not.
On the smaller test config a legitimate partial-text cue sits at 9.1 nats /
0.41 cosine against a nonsense query's 9.8 / 0.35 — clearly ordered, barely
separated in absolute terms. `is_confabulation`'s thresholds are therefore
configurable and must be calibrated per deployment; prefer comparing
`depth_nats` across queries to trusting the boolean.

## 4. Pattern completion — robust to severe occlusion

The key is 2048-d with 12% of units active. Occluding raw *coordinates* is a
weak test, since most coordinates are zero anyway — so both protocols are run:

**(a) coordinates occluded**

| kept | recall@1 | active units kept | clamp exact | iters |
|---|---|---|---|---|
| 0.80 → 0.10 | **1.000** throughout | 0.803 → 0.101 | 1.000 | 3.0 → 5.5 |

**(b) active units occluded — the honest difficulty scale**

| kept | recall@1 | units kept | cue cosine | iters |
|---|---|---|---|---|
| 0.80 | 1.000 | 205 | 0.894 | 3.0 |
| 0.50 | 1.000 | 128 | 0.708 | 3.5 |
| 0.30 | 1.000 | 77 | 0.550 | 4.0 |
| 0.20 | 1.000 | 51 | 0.449 | 4.5 |
| 0.10 | 1.000 | 26 | 0.323 | 5.6 |
| 0.05 | 0.775 | 13 | 0.229 | 7.8 |

> **Corrected by VIII.2 (ticket 10).** These occlusion numbers reproduce on real
> data — but they were measured with no baseline. A single-shot cosine lookup on
> the *identical* degraded cues does as well or better in 45 of 48 real-data
> cells. The robustness stands; the implication that it demonstrates an advantage
> of iterative settling is withdrawn.

Perfect completion from 26 of 205 active units (cue cosine 0.323), breaking
down only at 13 units. Known coordinates stayed exactly clamped in every run,
and iteration count rises as the cue degrades — the network is doing more work
to settle, exactly as the energy picture predicts.

## 5. Capacity — load does not degrade recall

| stored | recall@1 | min separation | mean separation |
|---|---|---|---|
| 40 | 0.575 | 0.2492 | 0.3408 |
| 50 | 0.575 | 0.1603 | 0.3645 |
| 100 | 0.575 | 0.0000 | 0.2686 |
| 200 | 0.575 | −0.0000 | 0.1791 |
| 400 | 0.575 | 0.0000 | 0.0981 |

Recall is **flat at 400 memories**, a 10× increase in load, consistent with the
exponential-capacity claim for modern Hopfield networks. (The 0.575 level is
the ambiguous-cue baseline from §3, not a capacity effect — it does not move.)

Min separation hits 0 because the filler here is written straight to the store,
bypassing the near-duplicate check, and the routine corpus contains exact
repeats — identical keys have Δ=0 and are formally not separable fixed points.
That is a useful demonstration that **the deduplication step in the normal
ingestion path is load-bearing**, not an optional nicety.

## 6. Forgetting curve — exactly as specified

| simulated day | memories | evergreen | temporal | mean salience |
|---|---|---|---|---|
| 0 | 83 | 6 | 77 | 1.0241 |
| 30 | 83 | 6 | 77 | 0.5482 |
| 60 | 83 | 6 | 77 | 0.3102 |
| 90 | 83 | 6 | 77 | 0.1913 |
| 180 | 6 | 6 | 0 | 3.0000 |
| 365 | 6 | 6 | 0 | 3.0000 |

Salience halves per 30 days as designed. (Evergreen salience is `max_strength`=3.0, the ceiling — see README §7.) Episodic memories cross the 0.05 prune
floor between day 90 and 180 (2^−6 = 0.0156) and are swept. All 6 evergreen
rules survive a simulated year — asserted in the benchmark, not just observed.

## 7. Consolidation — interleaved replay cuts forgetting 55%

Schema surprise on 400 old routine documents after training on off-schema data:

| condition | before | after | drift |
|---|---|---|---|
| no replay (naive) | 0.0960 | 1.4795 | **+1.3835** |
| interleaved replay | 0.0960 | 0.7261 | **+0.6301** |

**54.5% less drift.** The core CLS claim reproduces: interleaving hippocampal
replay with new data measurably protects the existing schema. (The quick-mode
run shows 90.8% on a smaller corpus; the effect size is corpus-dependent, the
sign is not.)

This number has read 57.1%, 56.1%, 55.3% and 54.5% across reruns and code
changes. The last of those is with episodic replay switched on (below); the
same run with `episodic_ratio=0` measured 55.3%, so the fix costs 0.8 points,
which is inside the spread of the metric itself.

### 7b. Pruning — the hippocampus now drains, but only partly

`prune_predicted` used to remove **0 of 36** memories at any budget, because
replay fed the cortex its own decoder output. §9 below has the diagnosis. With
`episodic_ratio=0.125`, one bootstrapped cortex deep-copied per row:

| epochs | memories | pruned | evergreen | median drop | min drop | schema drift |
|---|---|---|---|---|---|---|
| 5 | 36 → 36 | 0 | 6 | 0.8927 | 0.6431 | +0.2978 |
| 20 | 36 → 30 | 6 | 6 | 0.5696 | 0.1396 | +0.4577 |
| **40** | 36 → 20 | **16** | 6 | 0.4968 | 0.1338 | +0.6450 |
| 60 | 36 → 23 | 13 | 6 | 0.5153 | 0.1536 | +0.6841 |
| 100 | 36 → 23 | 13 | 6 | 0.5119 | 0.1765 | +0.7250 |

Drop ratio is surprise-now / surprise-at-ingestion, measured over the whole
memory set *before* pruning — measuring after would survivor-bias the median
upward by exactly the memories the pass succeeded on. All 6 evergreen rules
survive every row (asserted).

Three things to read off this table:

1. **More budget is not the answer past ~40 epochs.** The median drop ratio
   falls 0.89 → 0.57 → 0.50 and then flattens; 100 epochs releases no more than
   40 does. Consolidation converges to a mixture-determined ceiling, and the
   knob that moves the ceiling is `episodic_ratio`, not `epochs`.
2. **The median lands almost exactly on the 0.5 criterion**, which is why the
   count wobbles between 13 and 16 across the plateau. Half the memory set is
   sitting on the threshold; treat the count as ±3.
3. **The pass drifts the schema** (+0.30 to +0.73). This is not caused by
   episodic replay — the old decoder-only replay drifted +0.48 in the same pass
   while learning nothing, because its low-norm outputs pull the VAE toward
   that region. A replay-only `sleep()` with no new data is a diagnostic, not
   an operating mode; real consolidation interleaves.

The earlier claim that pruning was merely "budget-sensitive" was wrong. It was
structurally impossible, and the budget was a red herring.

---

## Summary of what these runs changed

Running the system rather than reasoning about it corrected four things:

1. **The retrieval benchmark was initially ill-posed.** With no unique incident
   references, only 19 of 30 anomaly texts were distinct and 22 of 30 cues
   matched several memories verbatim; recall@1 of 0.25 was measuring corpus
   degeneracy. Fixed by adding identifiers and by scoring unambiguous cues
   separately.
2. **The default novelty quantile is too permissive** at 0.95 (P=0.779). The
   sweep says 0.99–0.999.
3. **Coordinate-level occlusion is too easy a test** for a sparse key. The
   active-unit protocol is the real one — and the system still passes it down
   to 10% of units.
4. **Consolidation pruning silently did nothing** at a realistic training
   budget, while every other mechanism worked. Chasing that number is what
   turned up §9 — the cause was not the budget.


---

## 8. Post-review re-run

All numbers above were reproduced after the review fixes landed (including the
Hopfield energy reformulation in §S1 below). Differences were within run noise:
ROC AUC 1.0000 unchanged, key-representation table identical, replay benefit
57.1% → 56.1%. The energy change is exactly behaviour-preserving for unit-norm
patterns, which is what the retrieval path uses, so this is the expected result
rather than a lucky one.

Re-run again after the replay fix in §9. Experiments 1–6 came out **byte for
byte identical** (the only differing line in the diff was the wall-clock timing
of the bootstrap), which is what should happen: the change touches replay
content only, and nothing in experiments 1–6 calls consolidation. Experiment 7
is the one that moved, and §7 and §9 give both sides of it.

---

## 9. The replay defect: why pruning could never fire

Experiment 7 reported 0 of 36 memories released and RESULTS blamed the training
budget. That was wrong. Instrumenting a consolidation pass — surprise per
stored memory, before and after — showed the budget was never the constraint:

| | before pass | after pass | ratio |
|---|---|---|---|
| mean surprise over 34 stored memories | 1.1526 | 1.1544 | **1.0016** |

Surprise went *up*. Across all 34 memories the per-item ratio ranged 0.9985 to
1.0046 — not one moved even 1% toward the 0.5 prune criterion, at 20, 60 or 150
epochs. The criterion was correct and permanently inert.

**The cause.** `replay()` returned `cortex.decode(latents)` and trained the VAE
on it. That is self-distillation: the reconstruction target is already what the
model emits, so there is no gradient. Measured on the same pass:

| quantity | value |
|---|---|
| cos(`decode(latent)`, the embedding it came from) | **0.047** |
| cortex surprise on its own decoder output | **0.0001** |
| cortex surprise on the real stored embeddings | **1.1526** |
| mean norm of decoder output vs stored embeddings | 0.44 vs 1.00 |
| max cos(replay sample, any stored embedding) | 0.17 |

The decoder is trained on routine text and has no resolution where novel
episodes live, so it cannot represent an anomaly latent at all. Replay was not
reinstating episodes; it was emitting near-garbage that the cortex reconstructs
perfectly. The 56% forgetting protection was real but was coming from
pseudo-rehearsal — pinning the current function — not from replaying memories.

**The fix and its price.** Replay is now a mixture: `episodic_ratio` of the
batch is Langevin-sampled in *embedding* space around the true stored
embeddings, the rest is the old generated path. Both halves are load-bearing.
Full benchmark, 30 episodic memories:

| `episodic_ratio` | drift reduction | pruned @20 ep | pruned @40 ep |
|---|---|---|---|
| 0.000 (old behaviour) | 55.3% | 0 | 0 |
| **0.125 (default)** | **54.9%** | **6** | **16** |
| 0.250 | 50.3% | 15 | 21 |
| 0.500 | 46.0% | 23 | 29 |

Pure episodic replay was measured too: it releases every memory and destroys
the schema doing it (schema drift +0.33 against +0.019 for the mixture on the
quick corpus). 0.125 is the knee — the drift cost is inside the metric's own
run-to-run spread, and it is the smallest share that closes the loop at all.

**What still does not drain.** 14 of 30 episodic memories survive a converged
pass, and raising the budget does not release them (§7b). Releasing them means
raising `episodic_ratio`, which costs measurable schema protection at a rate of
roughly one point per 2–3 extra memories. There is no setting measured here
that drains the store fully and keeps drift protection intact.


---

# Part II — LoCoMo: the same system on real data

Everything above uses a synthetic corpus of 12 routine templates. This part runs
the same system on [LoCoMo](https://arxiv.org/abs/2402.17753) — 10 real
multi-session dialogues, 5882 turns, real timestamps spanning ~231 days, and QA
pairs whose `evidence` field names the exact turns that answer them, giving
**ground-truth retrieval targets**.

```bash
curl -sLo experiments/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
PYTHONPATH=. .venv/bin/python experiments/benchmark_locomo.py --conversations 3
```

Run below: 3 conversations, 1451 turns, 494 questions with resolvable evidence.

## A. Retrieval — the synthetic number was mostly the generator

| embedder | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| hashing (was the default) | 0.024 | 0.045 | 0.061 |
| **LSA (now the default)** | **0.121** | **0.174** | **0.204** |
| chance, ~590 turns stored | — | — | ~0.017 |

Against **1.000 recall@1 on the synthetic corpus.** That gap is the single most
important number in this document. The template corpus made retrieval look
solved; on real dialogue the same code, same hyperparameters, gets 0.121.

LSA is ~4x hashing and ~12x chance at recall@10, so the mechanism is doing
something real — but 0.174 recall@5 is not a usable retrieval system. The
binding constraint is the embedding, exactly as §9 predicted; what §9 did not
convey is the *size* of the effect.

**Why the default changed.** `LatentSemanticEmbedder` (TF-IDF over word and
character n-grams, truncated SVD) is now the default over `HashingEmbedder`.
It needs fitting, which `bootstrap()` now does on the training corpus. It is
still a bag of n-grams with no word order — the floor, not the goal. A real
sentence encoder could not be used here because HuggingFace is unreachable from
this environment, so the headline retrieval number should be read as a lower
bound on what the architecture can do, not an estimate of it.

## B. Novelty gate — weakly discriminative, and partly a length filter

| population | n | mean surprise | median |
|---|---|---|---|
| training turns | 293 | 0.8614 | 0.8593 |
| held-out turns, same conversation | 126 | 0.8773 | 0.8779 |
| turns from a *different* conversation | 126 | 0.9903 | 0.9901 |

The gate does discriminate — turns from an unseen conversation are more
surprising than held-out turns from the trained one — but by **13%**, against
the **13x** gap on synthetic data.

Worse, and this is the finding that should temper any claim about the gate:

```
correlation(surprise, turn length in words) = -0.480
```

On real dialogue, roughly a quarter of the variance in "novelty" is explained
by how short the turn is. A template corpus could not reveal this because every
document was about the same length. The gate is partly a length filter wearing
a VAE, and any deployment must control for length before trusting it.

At quantile 0.95 the gate stored 13.5% of held-out turns.

## C. Forgetting curve — a 30-day half-life is wrong for this timescale

Conversations span 167–231 days against the specified 30-day half-life:

| quantity | value |
|---|---|
| mean salience at the newest timestamp | 0.2716 |
| memories already below the 0.05 prune floor | 58/419 (13.8%) |
| recall@5 with decay applied | 0.168 |
| recall@5 with all priors flat | 0.184 |

Decay costs **1.6 points of recall** here and would delete 13.8% of the
conversation outright. The 30-day half-life came from the brief and is right for
a fast-moving operational log; it is wrong for a 231-day relationship history.
Half-life must be set from the domain's actual event spacing — it is not a
universal default, and the synthetic benchmark (which simulated ageing rather
than measuring it) could never have surfaced that.

## What Part II changes about Part I

Part I's conclusion — "four of five mechanisms measure well" — holds only on
data generated by the same kind of process the model learns. On real language:

1. **Retrieval degrades by ~8x** and is bounded by the embedder, not the
   Hopfield layer.
2. **The novelty gate is weak and confounded by length**, where synthetic data
   showed a perfect separation.
3. **The forgetting curve's default is mis-specified** for realistic timescales.

None of this invalidates the architecture; the mechanisms still work in the
direction claimed. It does mean the synthetic numbers should never be quoted as
evidence of practical performance, and Part I now says so.


---

# Part III — Improving recall

Part II measured recall@1 0.121 on LoCoMo. This part diagnoses where the signal
was lost and fixes it, verified on **two unrelated corpora** so that a change
helping one and hurting the other shows up as overfitting rather than progress:
LoCoMo (two-person personal dialogue) and
[QMSum](https://arxiv.org/abs/2104.05938) (many-speaker organisational meetings,
queries annotated to relevant turn spans).

```bash
PYTHONPATH=. .venv/bin/python experiments/recall_ablation.py   # where is it lost
PYTHONPATH=. .venv/bin/python experiments/recall_check.py      # did the fix work
```

## Diagnosis: ceilings vs pipeline

Plain kNN over turns, no memory system — the ceiling each embedding allows:

| ranking, no memory system | recall@1 | recall@5 |
|---|---|---|
| **TF-IDF cosine (sparse, no SVD)** | **0.320** | **0.575** |
| BM25 | 0.298 | 0.526 |
| LSA-1024 | 0.255 | 0.528 |
| LSA-256 | 0.174 | 0.397 |
| hashing-256 | 0.077 | 0.176 |

Then the full pipeline on top of LSA-256:

| pipeline | recall@1 | recall@5 |
|---|---|---|
| key=separated, β=8 (**the shipped defaults**) | 0.004 | 0.012 |
| key=separated, β=32 | 0.121 | 0.174 |
| key=embedding, β=32 | 0.150 | 0.223 |
| key=embedding, β=128 | 0.172 | 0.235 |
| key=separated, sparsity_k=512 / 1024 | 0.117 / 0.113 | 0.168 / 0.166 |

Three findings, each acted on:

1. **β=8 collapses retrieval, and it was the shipped default.** At β=8 every
   query settles onto one global mixture — six distinct cues gave six settled
   states at pairwise cosine 1.0000, with `converged=True` reporting success.
   Nobody caught it because *every* experiment in this repo overrode β to 32.
   The default is now 128. At β=128 the pipeline reaches 0.172 against its
   embedder's own kNN ceiling of 0.174 — i.e. the memory system stops being
   lossy at all, and everything left is the embedding.
2. **Dentate-gyrus separation hurts on real text.** 0.121 separated vs 0.150
   embedding at matched β, and raising `sparsity_k` does not recover it.
   Sparsification discards lexical detail real retrieval needs. DG still beats
   the VAE latent (Part I) — but the raw embedding beats both. Default is now
   `HippocampalKey.EMBEDDING`.
3. **Dimension is the binding constraint on the embedder.** Sparse TF-IDF with
   no reduction at all beats every dense option; among dense, 1024 nearly
   doubles 256. Default `input_dim` is now 1024.

## Result: 63x on LoCoMo, 7.5x on QMSum

Cumulative, each row adding one change:

**LoCoMo** — 3 conversations, 1451 turns, 494 questions

| configuration | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| was: LSA-256, separated, β=8 | 0.004 | 0.012 | 0.036 |
| + β=128 | 0.154 | 0.235 | 0.291 |
| + key=embedding | 0.172 | 0.235 | 0.310 |
| **now: + LSA-1024** | **0.251** | **0.318** | **0.356** |

**QMSum** — 6 meetings, 3473 turns, 38 questions

| configuration | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| was: LSA-256, separated, β=8 | 0.053 | 0.211 | 0.316 |
| + β=128 | 0.158 | 0.526 | 0.658 |
| + key=embedding | 0.211 | 0.421 | 0.579 |
| **now: + LSA-1024** | **0.395** | **0.632** | **0.658** |

Honest notes on these tables:

- **QMSum has only 38 questions** across 6 meetings. Treat its absolute values
  as indicative; the *direction* agrees with LoCoMo at every step, which is the
  claim being made.
- **`key=embedding` costs QMSum recall@5** (0.526 → 0.421) while helping
  recall@1 (0.158 → 0.211). It is not a free win on both corpora at every k;
  the LSA-1024 step then recovers it to 0.632.
- The Part II figure of 0.121 was measured at β=32, because that benchmark
  overrode the default. Against the *actual shipped* default of β=8 the old
  number was 0.004, which is why the headline multiple is so large. Most of
  this is fixing a defect, not tuning.

## Remaining headroom

Sparse TF-IDF still beats the best dense configuration on its own ceiling
(0.320 vs 0.255 recall@1). A hybrid — sparse lexical retrieval to shortlist,
Hopfield settling to rank — is the obvious next step and is not implemented. A
real sentence encoder remains untried here because HuggingFace is unreachable
from this environment.

---

# Part IV — Does the energy earn its keep?

Parts I–III measured *retrieval quality* and found the Hopfield network ties
plain cosine kNN exactly (0.172 vs 0.174 hit@1). That is not a near miss: hit@1
against a single gold item is the metric where nearest-neighbour is optimal by
construction, so no attractor dynamics can beat it. Part IV tests the two
things an energy model has that a similarity score does not — a *normalised*
density, and a landscape that write-time decisions can reshape.

## IV.1 Abstention — `experiments/abstention.py`

LoCoMo category 5 is 446 adversarial questions whose answer field is
deliberately empty (the question presupposes something false, usually by
attributing one speaker's action to the other). Every earlier harness discarded
them. They are the natural test for "does the organisation actually know this?"

10 conversations, 1977 questions (1531 answerable, 446 adversarial), hybrid
RP-4096 + BGE embeddings, ROC AUC with a 2000-draw bootstrap:

| signal | AUC @ β=128 | rejects adversarial @ 80% coverage |
|---|---|---|
| `cos_top1` (baseline) | 0.480 [0.448, 0.511] | 20.6% |
| `cos_margin` | 0.471 [0.441, 0.504] | 18.6% |
| `neg_attn_ent` | 0.465 [0.433, 0.494] | 16.6% |
| `log_density` | 0.481 [0.449, 0.512] | 21.1% |
| `neg_depth_nats` | 0.481 [0.450, 0.512] | 20.9% |
| `neg_settle` | 0.505 [0.475, 0.536] | 20.6% |

**Every signal is at chance, and cosine is slightly _below_ it.** Adversarial
questions score *higher* confidence than answerable ones, which is the corpus
working as designed: an adversarial question is built by taking a real turn and
swapping the attribution, so it is a near-verbatim paraphrase of something that
genuinely is in memory. Density cannot help, because the region really is dense.

Out-of-fold (held out by conversation, since the embedder and store are fitted
per conversation) the energy adds **−0.000** over cosine alone: 0.512 → 0.512.

### The β dependence, and why it does not rescue the result

`log_density` was suspected of being a relabelling of `cos_top1`. Spearman ρ
against cosine, swept over β:

| β | ρ(`log_density`, `cos_top1`) | ρ(`neg_depth_nats`, `cos_top1`) | energy adds (out-of-fold) |
|---|---|---|---|
| 8 | +0.242 | +0.285 | +0.009 |
| 32 | +0.835 | +0.851 | +0.011 |
| 128 | ≈1.0 (Δ CI [−0.000, +0.003]) | ≈1.0 | −0.000 |
| 512 | **+1.0000** | **+1.0000** | −0.000 |

At high β the mixture density is dominated by its nearest component, so
`log_density` degenerates into a monotone function of top-1 cosine and
*cannot* differ from it. That is a structural fact, not a tuning accident, and
it explains the Part I–III tie directly.

At β=8 the energy **is** a genuinely distinct signal (ρ=0.24) and the paired
bootstrap calls it a win (+0.038 [+0.003, +0.077]). This does not rescue
anything: it moves AUC from 0.480 to ~0.518 — from below chance to chance. Over
5 signals × 4 values of β, two CIs excluding zero is roughly what multiple
comparisons predict. **No configuration of this system can tell an answerable
question from an unanswerable one.**

## IV.2 Rules under a token budget — `experiments/rulebook{,_eval}.py`

LoCoMo cannot measure the use case this design was actually pitched at:
remembering *rules* without bloating context. Nothing in it is ever superseded,
and it has no cost model for retrieving too much. `rulebook.py` is a 26-rule
organizational policy corpus built for the two failure modes that matter —
supersession chains (v1→v2→v3, lexically near-identical) and scope
near-duplicates (EU/US, employee/contractor, prod/staging) — with 16 situation
queries. Budget = 256 tokens ≈ 7 rules, counted with the real BGE tokenizer.

`stale@256` is the number nobody measures: did a **superseded** rule get
injected? A missing rule makes the model say "I don't know". An obsolete rule
makes it confidently quote revoked policy, which is worse than empty context.

Re-measured after review ticket 03 corrected two defects in this harness (see
IV.2a below). `admitted@256` is the number of rules the budget actually let in,
reported because the prose above claims "a budget holds ~7 rules" and nothing
was checking it.

| arm | coverage@256 | tokens@cover | stale@256 | trap@256 | admitted@256 |
|---|---|---|---|---|---|
| BM25 | 0.92 | 115 | 0.81 | 0.81 | 7.0 |
| BGE kNN | 0.94 | 76 | 0.88 | 1.00 | 7.0 |
| BGE + recency rerank | 0.94 | 83 | 0.81 | 1.00 | 7.0 |
| MHN, no decay | 0.94 | 76 | 0.88 | 1.00 | 7.0 |
| MHN + recency log-prior | 0.94 | 81 | 0.88 | 1.00 | 7.0 |
| INGEST-GATED kNN (t=0.80) | 0.94 | 106 | 0.56 | 0.75 | 7.0 |
| **INGEST-GATED kNN (t=0.75)** | 0.94 | 99 | **0.00** | 0.69 | 7.0 |
| INGEST-GATED kNN (t=0.70) | 0.91 | 135 | 0.00 | 0.69 | 7.0 |
| ORACLE (current-only kNN) | 1.00 | 58 | 0.00 | 0.75 | 7.0 |

Three findings:

1. **Finding the right rule is easy; not dragging the dead one along is not.**
   Coverage is 0.92–0.94 everywhere. Staleness is 81–88% everywhere on the
   read side. The entire prize is supersession.
2. **Read-side machinery does not touch it.** The Ebbinghaus log-prior folded
   into the energy — the mechanism this architecture is built around — moves
   `stale@256` from 0.88 to 0.88. A recency *rerank* does no better (0.81).
   Reordering cannot help when the budget admits ~7 rules and both versions
   rank in the top few: you inject them both either way.
3. **Write-time gating solves it completely.** Streaming the rules in date
   order and *replacing* a stored rule when a new one lands within cosine 0.75
   takes `stale@256` from 0.88 to **0.00** — matching the oracle. This
   conclusion **survives** the ticket-03 corrections unchanged; what changed is
   that it is now measured over the same 26-rule candidate set as every other
   arm, so the gate can be scored against rules it deleted rather than only
   against the ones it kept.

### IV.2a Corrections to this harness (review ticket 03)

Two defects were fixed. Both were real; only one moved a number, and neither
overturned a finding — recorded here because "we checked and it held" is a
result, and because an uncorrected version of this table was quoted as the
design's one clear win.

**Defect 1 — the budget backfilled.** The injection loop used `continue` when a
rule did not fit, so it skipped the expensive rule and kept descending the
ranking, quietly packing cheaper low-ranked rules into the leftover space. That
is not what a caller assembling a prompt does (it appends until it cannot), and
it meant the documented "budget admits about seven rules" was never checked.
Fixed to `break`. **Effect on this corpus: none measurable.** Coverage is
identical to the last digit for all nine arms, because `rulebook.py`'s rules are
uniform in length (mean 36 tokens, so the budget almost always fills exactly at
the boundary). `admitted@256` is now reported and is 7.0 for every arm, which
confirms the "~7 rules" claim that had been asserted for four sections without
evidence.

**Defect 2 — the gated arm was scored on its own shrunken store.** `gated_arm`
returned `knn_arm(live, ...)`, ranking only the rules the gate chose to keep. A
rule the gate deleted could not appear in the ranking at all, so `stale@256` for
a gate that deleted every superseded rule was guaranteed 0.00 by construction.
The gate now expresses its verdict as a **demotion** rather than a deletion: it
ranks all 26 rules and sorts the ones it would have dropped strictly last.

**What that correction did and did not change.** The staleness column is
**bit-identical** before and after — 0.81 / 0.88 / 0.56 / 0.00 / 0.00 across the
arms. So the ticket's premise that the headline 0.00 was *pure* arithmetic is
too strong: the old metric already discriminated, as the t=0.80 row shows, where
a gate that leaves one superseded rule in the store scores 0.56 rather than
0.00. What the old construction genuinely broke was `tokens@cover`. When the
gate wrongly deleted a *live* rule — and at every threshold it deletes
`exp.travel.eu`, a scope sibling, not a revision — that rule became
unreachable, so the cover walk fell through to its full-corpus penalty. Scoring
over the full candidate set removes that artifact: tokens@cover drops from 119
to 106 (t=0.80), 115 to 99 (t=0.75) and 169 to 135 (t=0.70). The gate's
coverage cost is therefore **materially smaller** than this section previously
reported, though still well above the oracle's 58.

**Honest limit of the corrected metric.** At t=0.75 the 0.00 is still close to
structural: 18 rules survive the gate, the budget admits 7, so a demoted rule
cannot reach the context on a corpus this size. The number that carries the
evidence is the t=0.80 row (0.56), where the metric is demonstrably free to
report failure and does. `tests/test_rulebook_eval.py` fails if either defect
returns.

### Where the gate fails, and what the energy adds (nothing)

Separating the 7 true supersessions from the 18 other write decisions:

| signal | AUC |
|---|---|
| top-1 cosine | 0.984 |
| energy explained (normalised over the store) | 0.984 |

Identical. The single error is the same for both: `exp.travel.eu` and
`exp.travel.us` sit at cosine 0.854 — *above* the weakest genuine supersession
pair — so any threshold that catches all real revisions also deletes a live
scope sibling. That costs the gate its coverage (tokens@cover 115 vs the
oracle's 58). **This is not a memory-model problem.** No density, energy, or
attractor formulation distinguishes "revision of the same rule" from "the
sibling rule for a different region"; that needs a scope/metadata comparison.

An earlier version of this probe reported AUC 1.000 for both signals while the
gate two functions away was visibly making mistakes. It used `r.day < new.day`,
which silently excluded every same-day pair — and the same-day pairs are
exactly the scope siblings. Fourth instance in this project of a check written
to verify a mechanism being wrong in the mechanism's favour (see HANDOFF §5).

## What Part IV means for the design

- The **Hopfield retrieval layer is not the valuable component.** At usable β it
  is provably a relabelling of cosine similarity; at β low enough to differ, it
  is no better at anything measured.
- The **ingestion gate is** — it is the only intervention that moved a metric
  (0.88 → 0.00 stale), and it is cheap, because it pays once per write instead
  of once per query.
- The remaining error is **scope**, not similarity, and wants structured
  metadata rather than more memory physics.

---

## IV.3 The retrieval harness can express the separation configurations (ticket 07)

`experiments/recall_check.py` had no whitening axis at all, so **no whitened
retrieval number had ever been produced by this project**, and pattern
separation was reachable but in no published sweep. Both are now ordinary
sweep axes, selectable per row, on either corpus (`--corpus`).

Each row reports the separation **actually achieved** — `anisotropy()` over the
embeddings and hippocampal keys that row really wrote to the store — not the
configuration that was requested. Anisotropy is the mean cosine between
unrelated pairs, so **lower is better separated**; 0.0 is isotropic.

LoCoMo, 3 conversations, 1451 turns, n=494:

| configuration | @1 | @5 | @10 | aniso_emb | aniso_key |
|---|---|---|---|---|---|
| was: LSA-256, separated, beta=8 | 0.002 | 0.010 | 0.026 | 0.080 | 0.152 |
| + beta=128 | 0.221 | 0.281 | 0.354 | 0.080 | 0.152 |
| + key=embedding | 0.217 | 0.289 | 0.340 | 0.080 | 0.080 |
| now: + LSA-1024 | 0.306 | 0.358 | 0.383 | 0.041 | 0.041 |
| + whitened | **0.342** | 0.356 | 0.364 | **−0.001** | **−0.001** |
| + key=separated | 0.291 | 0.356 | 0.399 | 0.041 | 0.134 |
| + key=separated, whitened | 0.322 | 0.348 | 0.352 | −0.001 | 0.117 |

The first four rows reproduce the pre-refactor harness **exactly**, to three
decimals on all three metrics, so the new axes did not disturb the cumulative
history they were added alongside.

QMSum, **8 meetings, 4201 turns, n=53** — a deliberately reduced slice (the
25-meeting slice costs hours per LSA-1024 row). 53 questions cannot resolve a
recall difference at this project's ~0.04 limit; the anisotropy columns are
unaffected by that, being properties of the code and corpus rather than of the
question set:

| configuration | @1 | @5 | @10 | aniso_emb | aniso_key |
|---|---|---|---|---|---|
| was: LSA-256, separated, beta=8 | 0.019 | 0.226 | 0.396 | 0.074 | 0.151 |
| + beta=128 | 0.245 | 0.528 | 0.642 | 0.074 | 0.151 |
| + key=embedding | 0.264 | 0.472 | 0.585 | 0.074 | 0.074 |
| now: + LSA-1024 | 0.396 | 0.623 | 0.736 | 0.048 | 0.048 |
| + whitened | 0.415 | 0.642 | 0.679 | **0.000** | **0.000** |
| + key=separated | 0.358 | 0.604 | 0.660 | 0.048 | **0.139** |
| + key=separated, whitened | 0.434 | 0.623 | 0.660 | 0.000 | 0.118 |

**The de-separation replicates on the second corpus.** DG raises key anisotropy
0.048 → 0.139 on QMSum, against 0.041 → 0.134 on LoCoMo — the same effect, the
same size, on two unrelated corpora. Whitening drives it to 0.000 on both. The
recall directions also agree across corpora (whitening +0.036 LoCoMo / +0.019
QMSum; DG −0.015 / −0.038), so there is no corpus-overfitting signal here in
either direction — but the QMSum magnitudes rest on 53 questions and must not
be quoted as effects.

**The measured column immediately contradicted the premise it was added to
test.** Review ticket 08 is written on the assumption that "two mechanisms in
this codebase raise separation directly — whitening and dentate-gyrus pattern
separation". Measured, only one of them does:

- **Whitening raises separation as advertised**: key anisotropy 0.041 → −0.001,
  i.e. fully isotropic, exactly the effect V.2 reports for BGE.
- **Dentate-gyrus separation *lowers* it.** `key=SEPARATED` moves key
  anisotropy the wrong way, 0.041 → **0.134** at LSA-1024, and 0.080 → 0.152 at
  LSA-256. The DG keys for unrelated text are **more** alike than the raw
  embeddings they are computed from. The mechanism named "pattern separation"
  is, on this embedder, a pattern *de*-separation.

Recall agrees with the diagnostic: `key=separated` scores 0.291@1 against the
0.306 baseline, while whitening scores 0.342. This is precisely what ticket
07's "report the measured quantity, not the intent" requirement exists to
catch, and it is a correction to the reasoning ticket 08 was to be built on.

**Not yet resolvable.** Whitening's +0.036 hit@1 over baseline sits just under
this project's ~0.04 resolution limit at n=494 and has not been McNemar-tested
here; it is not yet a finding. The QMSum half is a reduced slice and is
reported separately. Do not promote a default off this table.

---

## IV.4 The separation mechanisms, turned on and measured (ticket 08)

The system has never been measured with its separation mechanisms enabled.
`experiments/separation_check.py` runs four arms at one operating point
(LSA-1024, beta=128) on both corpora and compares each against the baseline
with the **exact paired McNemar test**.

**The ticket's premise did not survive contact with the measurement**, and the
experiment was rebuilt around what is actually true. Ticket 08 assumes two
mechanisms raise separation. Only whitening does; dentate-gyrus keys *lower*
it (IV.3). DG is therefore carried as a **negative control**: if separation is
what the capacity story says it is, the arm that lowers separation should not
beat the arm that raises it. That is falsifiable in a way the original framing
was not.

LoCoMo, n=494:

| arm | @1 | @10 | aniso_key | @1 vs base | @10 vs base |
|---|---|---|---|---|---|
| baseline | 0.306 | 0.383 | +0.041 | — | — |
| **whitened** (raises separation) | **0.342** | 0.364 | **−0.001** | **+0.036, p=0.0001** | −0.018, p=0.25 |
| DG key (lowers separation) | 0.291 | 0.399 | +0.134 | −0.014, p=0.26 | +0.016, p=0.32 |
| both | 0.322 | 0.352 | +0.117 | +0.016, p=0.20 | −0.030, p=0.049 |

QMSum, 8 meetings, **n=53** — underpowered by construction; directions only:

| arm | @1 | @10 | aniso_key | @1 vs base |
|---|---|---|---|---|
| baseline | 0.396 | 0.736 | +0.048 | — |
| whitened | 0.415 | 0.679 | **+0.000** | +0.019, p=1.00 |
| DG key | 0.358 | 0.660 | +0.139 | −0.038, p=0.63 |
| both | 0.434 | 0.660 | +0.118 | +0.038, p=0.69 |

**Six tests were run, so the threshold is Bonferroni-corrected to 0.05/6 =
0.0083.** Exactly one result clears it: **whitening improves LoCoMo hit@1 by
+0.036 (p=0.0001, 20 questions flipped right against 2 flipped wrong)**. The
"both" arm's hit@10 regression (p=0.049) does **not** survive correction and is
not a finding. Nothing on QMSum is resolvable at n=53.

**The ~0.04 heuristic and the exact test disagree here, and the exact test
wins.** HANDOFF §3.1d's rule of thumb — "differences below ~0.04 hit@1 are not
resolvable at n=494" — would have discarded this +0.036 as noise. The paired
test resolves it decisively because the discordance is lopsided (20:2). The
heuristic is a rate-difference approximation; McNemar conditions on the pairs
and is the correct instrument. **The rule of thumb is too conservative for
strongly paired comparisons and should not be applied mechanically.**

**The one directional result, stated plainly.** The only arm that raised
separation is the only arm that improved retrieval, and the arm that lowered
separation improved nothing on either corpus. That is consistent with the
capacity story's claim that separation is the quantity that matters — but it is
a single significant cell on one corpus, not a confirmation of the theory.

### Defaults: unchanged, with the measurement as the reason

Whitening stays **off** by default and DG stays **off** (`KeyConfig.mode`
already defaults to `EMBEDDING`).

- **DG off is now justified by measurement rather than by the earlier
  recall-only comparison.** It does not merely fail to help; it moves the
  quantity it is named for in the wrong direction, on two unrelated corpora.
- **Whitening is not promoted despite winning its one powered test.** This
  project's standing rule is that a change must hold on both corpora, and
  QMSum's 53-question slice could not support it. **That powered QMSum run has
  since been made** (25 meetings, 170 q — VI.6) and it does **not** show the
  same direction: QMSum hit@1 is **flat, 0.335 → 0.335**, against LoCoMo's
  +0.036. A delta of exactly zero needs no significance test to fail to
  replicate. hit@5 does move the same way (0.541 → 0.571) and hit@10 slips
  slightly on both corpora, so the picture is "helps the top of one corpus'
  ranking, does nothing at rank 1 on the other".

  **This is the both-corpora rule doing its job.** A change that improved
  LoCoMo hit@1 with p=0.0001 and survived correction still does not become a
  default, because it buys nothing at rank 1 on an unrelated corpus. Whitening
  remains a per-deployment choice, justified by measurement on the deployment's
  own corpus — which is also what V.2a concluded about fitting the whitener.

### Do the earlier retrieval conclusions survive?

Partly, and the part that does not is bounded. Everything above is still at
beta=128, where the energy is provably a monotone function of top-1 cosine, so
none of it tests the Hopfield layer against cosine — that is ticket 09's grid,
which has not been run. What IS now established is that the earlier conclusions
were drawn at a separation the mechanisms could have changed and nobody had
changed: whitening moves key anisotropy from +0.041 to −0.001 and measurably
improves rank-1 retrieval, so "separation was low and we compensated with beta"
was a real choice with a real alternative, not a forced one.

---

# Part V — The MHN as a substrate, not a ranker

Parts I–IV all scored the Hopfield network as a **ranker of text chunks**. That
was the wrong axis, and the answer was fixed before any of it ran:
`softmax(beta X xi) X` over stored rows *is* an attention head, which is a soft
kNN. "Ties cosine kNN" was a tautology reported as a finding.

The claims that actually separate an associative memory from a vector database
are about representation. `experiments/superposition.py` tests them.

## V.1 One MHN update is exactly one attention head

| beta | float32 | float64 |
|---|---|---|
| 1 → 512 (5 values) | ≤ 8.94e-08 | **0.00e+00** |

Not "analogous to" — bit-identical in double precision. The stored energy folds
a `-beta*||x||^2/2` term into the logits; patterns are unit-norm by config, so
that term is the same constant for every memory and cancels inside the softmax.

**Consequence:** memories do not have to be retrieved *before* the model and
pasted into the prompt. They are a K/V pair the model can attend to directly,
at a cost of **zero context tokens**.

One number that matters for how that behaves: a real transformer head uses
scale `1/sqrt(d)`, which for d=384 is **beta = 0.051** — four orders of
magnitude below the beta=128 this store needs for episodic recall. Memories
injected into a real head therefore land deep in the metastable regime, not the
single-attractor regime.

## V.2 Superposition capacity — and the anisotropy that was destroying it

A sum of k near-orthogonal unit vectors retains each component at cosine
~1/sqrt(k). Sentence embeddings are not near-orthogonal:

| code | mean cosine between *unrelated* memories |
|---|---|
| BGE as shipped | **+0.649** |
| BGE whitened, **in-sample fit** (n=788, transductive) | −0.001 |
| BGE whitened, **held-out fit** (n=5094, disjoint conversations) | +0.170 |
| random unit vectors | +0.000 |

BGE vectors sit in a narrow cone. There is barely any component structure to
decode, so a mixture of 4 is indistinguishable from any other mixture of 4.
This is the same defect the `HopfieldConfig.beta` docstring already recorded
from a different angle — "DG keys for unrelated text sit at cosine ~0.71,
giving Delta=0.29" — and Delta is precisely the separation term Ramsauer's
capacity results multiply by beta.

**Review ticket 04 — the original table below was measured in-sample:** the
whitener was fitted on exactly the 788 vectors it was then scored on, which is
transductive and flatters the result. Re-measured with the whitener fitted on
a disjoint pool of 5094 turns from the *other* 8 LoCoMo conversations, then
applied unchanged to the 788 scored turns:

Per-item recall of k memories superposed into **one 384-d vector**, then
decoded against 788 stored memories:

| k | BGE as shipped | whitened, in-sample fit | whitened, held-out fit | random unit (ceiling) |
|---|---|---|---|---|
| 2 | 0.920 | **1.000** | 0.988 | 1.000 |
| 4 | 0.190 | **1.000** | 0.556 | 1.000 |
| 8 | 0.064 | **0.999** | 0.221 | 1.000 |
| 16 | 0.052 | **0.976** | 0.106 | 0.966 |
| 32 | 0.060 | **0.882** | 0.095 | 0.847 |
| 64 | 0.098 | **0.766** | 0.122 | 0.703 |

(The as-shipped and in-sample columns reproduce the original table within
trial/embedding noise, confirming the harness — the in-sample fit is not new,
only correctly labelled now.)

**The in-sample fit was not a small artifact.** "Whitening recovers the full
theoretical capacity" is only true of the transductive measurement. Fitted on
data the scored vectors were never part of, whitening still fixes anisotropy
partially (+0.649 → +0.170, against a −0.001 in-sample ideal) but per-item
recall at k=8 falls from 0.999 to 0.221 — barely above the unwhitened 0.064 —
and the k=4 exact-set rate drops from 1.00 to 0.08. **Whitening genuinely
helps** (compare the whitened-held-out and as-shipped columns throughout), but
the "recovers full capacity" and "8 memories losslessly" claims do not survive
an honest fit.

**Why: domain shift, not sample size.** `experiments/superposition.py`'s
`fit_size_check` sweeps the size of the disjoint fit pool from 64 to 5094 rows
(13x the 384-dimension warning threshold) and the residual anisotropy does not
move: +0.168 at n=64, +0.188 at n=384, +0.167 at n=5094. A second control
fits on a disjoint *half of the same two conversations* being scored (n=394,
well under the cross-conversation pool's size) and gets anisotropy **+0.003**
— matching the in-sample fit, not the cross-conversation held-out one. The gap
above is which conversations the fit corpus comes from, not how many rows it
has. `Whitener.fit`'s `n < d` warning is calibrated correctly for what it
guards — rank-deficient covariance — and the sweep shows genuine instability
below n=384 (recall@k=8 ranges 0.152–0.215 there vs a tighter 0.19–0.25 band
above it). But it is not sufficient: a fit corpus can clear n≥d by 13x and
still leave most of the anisotropy gap unclosed if it is drawn from a
different distribution than what gets transformed. A deployment should fit the
whitener on a representative sample of what it will encode, not merely a large
one; `cls_memory/whitening.py`'s `Whitener.fit` docstring now says so.

Reproduce with `PYTHONPATH=. uv run python experiments/superposition.py`
(Part 2's anisotropy block, capacity tables, and the `fit-size check` /
`same-domain control` sections).

### V.2b Now reproducible from the library (ticket 06)

This experiment previously defined **private copies** of the whitening,
superposition and decode operations, so V.5/VI.2/VI.3's claim to have been
re-measured "through the library classes rather than the prototype" could not
be reproduced from the code on the branch. It now calls
`cls_memory.whitening.Whitener`, `cls_memory.whitening.anisotropy`,
`ModernHopfieldNetwork.superpose` and `ModernHopfieldNetwork.decode`; no
duplicate implementations remain.

**Every cell in V.2 above is bit-identical before and after the port.** That is
the expected result rather than a lucky one: for unit-norm patterns under a
uniform prior the attention logits order the store exactly as cosine does, so
swapping the hand-rolled cosine top-k for `decode` cannot move a ranking.

Four tests in `tests/test_superposition.py` pin the implementations together
(in-sample fit, held-out fit, superpose+decode ranking, and the end-to-end
recall metric). One found a subtlety worth recording: `decode` ranks by
`beta*(X xi - ||x||^2/2) + log w`, and at beta=128 that amplifies float
rounding enough to swap two candidates whose cosines differ by ~1e-8
(measured: 0.969203949 vs 0.969204009). The tests therefore compare the
selected *similarities*, not the indices — asserting index equality would fail
on an exact tie while catching nothing real.

## V.3 Settling destroys superposition — at every beta

Per-item recall on whitened vectors, `none` = the raw normalised sum with no
Hopfield update applied:

| k | none | b=0.051 | b=0.5 | b=2 | b=8 | b=32 | b=128 |
|---|---|---|---|---|---|---|---|
| 2 | **1.000** | 0.005 | 0.005 | 0.005 | 0.510 | 0.730 | 0.620 |
| 4 | **1.000** | 0.005 | 0.005 | 0.005 | 0.253 | 0.253 | 0.253 |
| 8 | **1.000** | 0.011 | 0.011 | 0.013 | 0.149 | 0.149 | 0.149 |
| 16 | **0.974** | 0.016 | 0.016 | 0.020 | 0.109 | 0.110 | 0.109 |
| 32 | **0.892** | 0.040 | 0.039 | 0.042 | 0.107 | 0.116 | 0.114 |

No beta preserves a mixture. Low beta pulls the state to the global centroid
(0.005); high beta snaps it to one attractor (0.15–0.25). **The update is a
cleanup operator and is antithetical to holding a superposition.**

> **⚠ The table above no longer reproduces, and is not yet re-published.** The
> current code prints a materially different high-beta half — at k=2:
> `none` 0.980, b=8 **0.003**, b=32 **0.355**, b=128 **0.573**, against the
> 1.000 / 0.510 / 0.730 / 0.620 tabulated above. The low-beta columns and the
> qualitative conclusion (no beta preserves a mixture; the update is a cleanup
> operator) are unaffected — every settled number remains far below the
> unsettled `none` column.
>
> **The ticket-06 port is not the cause.** The changed figures appear in a run
> made *before* the port, and the post-port run reproduces them exactly, so
> the library swap moved nothing here either. The likely cause is an RNG-stream
> shift: ticket 04 added `fit_size_check` and the same-domain control, both of
> which draw from the same `generator` before `beta_sweep` runs, so the sweep
> now samples different member sets. That is the confound `experiments/
> significance.py` is already documented as suffering from (HANDOFF §4), here
> reaching a published table. Settling outcomes are bimodal (member vs global
> centroid), so a reseed can swing a mean hard — but a 0.510 → 0.003 move is
> larger than that explanation comfortably covers, and it has **not been
> attributed by measurement**. Do not quote either version of the high-beta
> columns until `beta_sweep` is given its own independent generator and re-run.

The architectural correction that follows:

    hold      superposed sum of whitened memory vectors     (one K/V slot)
    decode    attention LOGITS against the store            (X @ state)
    complete  ITERATE the update, high beta                 (one episode out)

`X @ state` is the MHN's own logit computation. So the Hopfield machinery *is*
the decoder — you use its **logits**, never its settled output, unless you
specifically want to collapse to a single episode. Every experiment before this
one used the settled output.

## V.4 What it costs

Mean rule = 36 real tokens.

| rules applied | as text | as K/V | ratio |
|---|---|---|---|
| 3 | 108 tokens | 1 slot | 108x |
| 10 | 360 tokens | 1 slot | 360x |
| 100 | 3600 tokens | 1 slot | 3600x |

> **Corrected by VIII.5 (ticket 13).** These ratios price a slot count, not a
> working memory, and they were framed as a *capacity* result. Two corrections.
> (a) **The ratio the measurement supports is not usable capacity.** At 26 rules in
> one slot (610x measured) the memory scores NLL 3.625 against a no-memory
> baseline's 3.612 — indistinguishable from supplying nothing. A learned read-in
> lifts that only on situations it trained on, and generalises *worse* than
> untrained (VIII.5). The honest cost table is below.
> (b) **This was never a Hopfield capacity claim.** Modern-Hopfield capacity
> (Ramsauer et al. 2021, arXiv:2008.02217) governs storing N patterns as N rows and
> retrieving *one* — `step`/`retrieve`. Summing N into one vector and reading
> several back is `superpose`/`decode`, governed by the vector-symbolic /
> holographic-reduced-representation literature (Plate, *Holographic Reduced
> Representations*, IEEE Transactions on Neural Networks, 1995), whose crosstalk
> bound is far below what these ratios imply and requires near-orthogonal codes —
> which is what V.2's whitening result rediscovered empirically.
>
> | rules in 1 slot | token ratio | recitation NLL | vs no memory (3.612) |
> |---|---|---|---|
> | 26 (`kv_super_all`) | 610x | 3.625 | **no better than nothing** |
> | 26 (`kv_learned_all`, held out) | 610x | 5.126 | **worse than nothing** |
>
> The token ratio is real. What it buys, at one slot, is not.

Text injection is linear in rule count. A superposed state is one slot,
bounded by V.2's capacity rather than by the prompt.

## V.5 Whitening is not a free win for ranking

Measured separately, LoCoMo turn retrieval, n=494:

| | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| BGE as shipped | 0.269 | 0.543 | 0.676 |
| BGE whitened | 0.304 | 0.492 | 0.551 |

It helps hit@1 by +0.035 (at this project's ~0.04 resolution limit, so a tie)
and clearly *hurts* depth. That is the expected trade: anisotropy makes
everything mildly similar, which helps fuzzy top-10 recall and destroys the
component structure superposition needs. **Whiten for the substrate, not for
the ranker** — they want opposite geometries, and that is a real design
decision the system currently does not expose.

## Status of the three claims

| claim | verdict |
|---|---|
| memory as K/V inside attention | **proved exactly** (0.00e+00); end-to-end demo blocked — HuggingFace is 403 by proxy policy, so no decoder LM is reachable here |
| N rules → one vector | **holds, after whitening**: 8 lossless, 16 at 97%, 32 at 89% — but only without settling |
| schema absorption (CLS) | **tested, negative** — gist recall never beats a dynamics-free centroid of the same neighbours, on either corpus, at any temperature; see VIII.3 |

---

# Part VI — Promoting Part V into the library

Part V was measured in `experiments/superposition.py`. This section records what
was moved into `cls_memory/` and what changed when it was re-measured through
the library classes rather than the prototype.

## VI.1 What was promoted

| prototype | library |
|---|---|
| `whiten(x, floor=1e-2)` — refits an SVD on whatever matrix it is handed | `cls_memory.whitening.Whitener` — `fit(corpus)` once, then the **same** mean/rotation/scale applied to documents and queries; serialises through `state_dict` |
| `anisotropy(x, generator)` | `cls_memory.whitening.anisotropy(x)` — fixed default seed so it can be asserted on |
| `superpose(...)` — seeded from the member mean and then **settled** | `ModernHopfieldNetwork.superpose(members)` — the normalised sum, explicitly *not* settled |
| `torch.topk(patterns @ state, k)` inline | `ModernHopfieldNetwork.decode(state, k)` — the store's own `logits()` path |
| — | `WhiteningConfig` (opt-in, `enabled=False`) + `WhitenedEmbedder`, fitted in `OrganizationalMemory.bootstrap` |

`step` and `retrieve` now carry the V.3 table in their docstrings and say
plainly that they collapse mixtures and must never be used to hold one. That
mistake was made in every experiment before Part V and is the single thing most
likely to be repeated.

## VI.2 Part V reproduces through the library

Same 788 LoCoMo turns, 200 trials, `superpose` + `decode`:

| k | as shipped | whitened | whitened, settled b=128 | whitened, settled b=2 |
|---|---|---|---|---|
| 2 | 0.950 | **1.000** | 0.625 | 0.005 |
| 4 | 0.190 | **1.000** | 0.258 | 0.009 |
| 8 | 0.056 | **0.999** | 0.151 | 0.012 |
| 16 | 0.049 | **0.974** | 0.109 | 0.023 |
| 32 | 0.059 | **0.887** | 0.115 | 0.043 |
| 64 | 0.092 | **0.769** | 0.141 | 0.079 |

Anisotropy: **+0.649** as shipped, **−0.001** whitened — identical to V.2. The
capacity numbers match V.2/V.3 to within trial noise (the k=2 unwhitened cell
moves 0.915 → 0.950 across seeds; nothing else moves by more than 0.006).

## VI.3 A correction to V.5: half the ranking loss was the refit

V.5 reported that whitening costs retrieval depth. Re-measured with the fitted
`Whitener` instead of the prototype's per-call refit, same 3 conversations,
n=494:

| | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| BGE as shipped | 0.269 | 0.543 | 0.676 |
| whitened, **fitted once** on 1451 turns | 0.302 | 0.536 | 0.628 |
| whitened, refit per conversation (= V.5) | 0.304 | 0.492 | 0.551 |

The as-shipped row reproduces V.5 exactly, so the harness is the same. The
difference is the fit: `dense_knn_ranker` encodes one conversation at a time, so
the prototype was refitting on 369–663 turns in 384 dimensions — a rank-deficient
covariance. **Roughly half the published depth loss was that artifact.** The
direction of V.5 survives: hit@1 +0.033 (inside the ~0.04 resolution limit, i.e.
a tie) and hit@10 −0.048, still a real loss. Whiten for the substrate, not for
the ranker — but the honest price is 0.628, not 0.551.

Fitted on all 10 conversations (n=1977) the same comparison is 0.252/0.502/0.615
as shipped against 0.274/0.509/0.610 whitened, i.e. the loss shrinks further as
the fit corpus grows. That is the expected behaviour of a covariance estimate
and it is why `Whitener.fit` warns when it is given fewer rows than dimensions.

### VI.3a Re-checked: `WhitenedEmbedder` had no `encode_query`

Review ticket 02 found that `WhitenedEmbedder` (above) never defined
`encode_query`, so `dense_knn_ranker`'s `getattr(embedder, "encode_query",
embedder.encode)` fallback silently used the *document* path for the whitened
arm's queries, while the "BGE as shipped" row used `BGEEmbedder.encode_query`
and kept BGE's query instruction prefix — the same asymmetry class that
already forced one retraction (§3.1d in `HANDOFF.md`). `WhitenedEmbedder` now
delegates to the wrapped embedder's `encode_query` (falling back to its
`encode` if it has none) and applies the same fitted whitener transform;
`tests/test_superposition.py::test_whitened_encode_query_is_not_the_document_path`
fails if that delegation is ever removed.

Re-measured with `experiments/whitening_query_check.py`, same 3 conversations,
n=494, "whitened, fitted once" row:

| | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| pre-fix (`encode_query` absent, query used the document path) | 0.300 | 0.534 | 0.632 |
| post-fix (`encode_query` delegates + whitens) | 0.302 | 0.536 | 0.628 |

The published VI.3 table above (0.302/0.536/0.628) already matches the
post-fix row exactly, so **the table is confirmed unchanged** — it was, by
coincidence of the measurement, already run in a way that reproduces the
post-fix numbers. The fix moves the pre-fix figures by +0.002/+0.002/−0.004,
far under the ~0.04 resolution limit. Exact paired McNemar between "BGE as
shipped" and the post-fix whitened row: delta +0.032 hit@1, b01=54, b10=38,
**p=0.117** — consistent with, and slightly tighter than, the "inside the
resolution limit" tie already recorded two paragraphs up. No conclusion in
this section changes.

## VI.4 Tests

`tests/test_superposition.py`, 24 tests (4 added for VI.3a's `encode_query`
delegation), on real BGE embeddings of LoCoMo turns
(synthetic Gaussians are isotropic by construction and would make the anisotropy
test vacuous — HANDOFF §5). Each asserts *both ends* of a gap, and the suite was
checked by mutation rather than by inspection:

| mutation | tests killed |
|---|---|
| `Whitener.transform` returns the input unwhitened | 8 |
| `superpose` settles the sum before returning it | 7 |
| `decode` ranks the settled state instead of the held one | 6 |
| the `state_dict` overrides removed | 2 |

Full suite: **160 passing, 0 skipped** (151 before this review round; 131
before Part VI). The 9 added are 4 for VI.3a's `encode_query` delegation and
5 in `tests/test_rulebook_eval.py` for IV.2a's budget and gating corrections.
Whitening is off by default, so every number in Parts I–V was re-measured
unchanged.

## VI.5 Ceilings now share one IDF convention (ticket 05), and the RP-4096 retraction is re-checked

Review ticket 05 found that `recall_ablation.py`'s ceilings section compared
`TfidfIndex`/`BM25Index` (fit per-conversation) against `HashedProjection`
(fit on the whole corpus) — two different IDF conventions in the same table.
That mismatch is what HANDOFF.md §3.1d item 1's retraction of "RP-4096 passes
the sparse TF-IDF ceiling" rests on. `BM25Index` now takes the same
`idf_corpus` hook `TfidfIndex` already had, `main()` threads one shared
`idf_corpus` (the whole loaded corpus) through TF-IDF, BM25 *and*
`HashedProjection`, prints which corpus was used, and every ceiling row is
checked with `assert_shared_idf` — a run that ever fits one arm on a different
pool than the others now raises immediately instead of silently confounding
the comparison. (Decoupling `BM25Index`'s IDF fit from its ranked-document set
also surfaced a real bug: `rank()` indexed `self.postings[term]` unconditionally
and raised `KeyError` for a term present in the shared IDF corpus but absent
from the conversation being ranked; fixed to `self.postings.get(term, ())`.)

Re-run with `experiments/idf_retraction_check.py`, same 3 conversations,
n=494, shared IDF corpus (1451 docs):

| ranking | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| TF-IDF cosine (exact-sparse, **the ceiling**) | **0.322** | 0.565 | 0.642 |
| BM25 (exact-sparse) | 0.304 | 0.524 | 0.599 |
| RP-4096-bm25, seed 0 | 0.330 | 0.581 | 0.640 |
| RP-4096-bm25, seed 1 | 0.318 | 0.555 | 0.628 |
| RP-4096-bm25, seed 2 | 0.310 | 0.579 | 0.648 |
| RP-4096-bm25, seed 3 | 0.314 | 0.555 | 0.634 |
| RP-4096-bm25, seed 4 | 0.332 | 0.573 | 0.636 |
| **RP-4096-bm25, mean ± sd over 5 seeds** | **0.321 ± 0.010** | 0.569 ± 0.013 | 0.637 ± 0.007 |

Exact paired McNemar, RP-4096-bm25 vs exact-sparse TF-IDF, hit@1, per seed:

| seed | delta | b01 | b10 | p |
|---|---|---|---|---|
| 0 | +0.0081 | 15 | 11 | 0.5572 |
| 1 | −0.0040 | 12 | 14 | 0.8450 |
| 2 | −0.0121 | 7 | 13 | 0.2632 |
| 3 | −0.0081 | 9 | 13 | 0.5235 |
| 4 | +0.0101 | 17 | 12 | 0.4583 |

**Verdict: the retraction is CONFIRMED, not revised.** Unifying the IDF
convention moves both figures slightly (TF-IDF 0.320 → 0.322, RP-4096-bm25
mean 0.319 → 0.321 — the fix was worth about 0.002, an order of magnitude
below the ~0.04 resolution limit) but does not change the conclusion: RP-4096
does not pass the exact-sparse ceiling. Every seed sits within ±0.012 of the
ceiling and every McNemar test is far from significant (p ≥ 0.26, well above
0.05), so the two are indistinguishable under this project's resolution rule —
a tie, not a win, exactly as the original retraction stated. This is also the
structurally expected result: RP-4096-bm25 is a Johnson–Lindenstrauss
approximation of the same BM25-weighted cosine `TfidfIndex`/`BM25Index`
compute exactly, so it cannot systematically beat what it approximates.

## VI.6 The powered QMSum run IV.4 flags as not yet made

IV.3/IV.4 cover ticket 07's deliverable (whitening and `key_mode` as
independent sweep axes on `experiments/recall_check.py`, measured
`aniso_emb`/`aniso_key` per row) and the HANDOFF §2 correction to the stale
baseline; neither is repeated here. `tests/test_recall_check.py` adds a
synthetic-data regression guard for the two axes, independent of the real-data
numbers below. This section supplies one specific thing IV.4 names as missing:

> "Whitening is not promoted despite winning its one powered test, because
> this project's standing rule is that a change must hold on both corpora and
> QMSum's 53 questions cannot support it. Flipping this default needs a
> powered QMSum run (25 meetings, 170 q) showing the same direction. That run
> costs hours per row and has not been made." (IV.4, "Defaults: unchanged")

That run, at the harness's real default size:

| configuration | hit@1 | hit@5 | hit@10 | aniso_emb | aniso_key |
|---|---|---|---|---|---|
| now: + LSA-1024 (baseline) | 0.335 | 0.541 | 0.682 | 0.056 | 0.056 |
| + whitened | 0.335 | 0.571 | 0.671 | 0.000 | 0.000 |
| + key=separated | 0.294 | 0.535 | 0.659 | 0.056 | 0.143 |
| + key=separated, whitened | 0.329 | 0.524 | 0.647 | 0.000 | 0.118 |

It does not show the same direction at hit@1: whitening moves LoCoMo hit@1
+0.036 (0.306 → 0.342, IV.4's one significant result) but QMSum hit@1 here is
**flat** (0.335 → 0.335) at 170 questions, not the +0.019 IV.3's 53-question
slice suggested. hit@5 does move the same way as LoCoMo's direction (0.541 →
0.571) and hit@10 drops slightly, as on LoCoMo (0.682 → 0.671). DG
`key=separated` replicates its LoCoMo de-separation exactly in kind (`aniso_key`
0.056 → 0.143 here, 0.041 → 0.134 on LoCoMo) and again helps nothing.
No significance test is run on this table -- that instrument, and the call on
what it means for the standing "both corpora" rule, is ticket 08's, not this
one's. This section only supplies the measurement IV.4 said was missing.

Reproduce with:

```bash
PYTHONPATH=. uv run python experiments/recall_check.py
PYTHONPATH=. uv run pytest tests/test_recall_check.py -p no:warnings
```

---

Part VII — the end-to-end K/V demo this table records as blocked, run against a
real decoder LM now that HuggingFace is reachable — is in
[`RESULTS-PART-VII.md`](RESULTS-PART-VII.md). It confirms the zero-context-token
claim exactly and retracts V.4's compression ratio.

---

# Part VIII — Do the attractor dynamics earn their keep? (review tickets 09–13)

Parts III–VI improved retrieval and tightened the harness, but every gain they
found belonged to the *embedding* — dimension, whitening, an honest IDF. None
of it was attributable to the Hopfield layer itself. Tickets 09–13 ask the
question that leaves directly: **is there anything the attractor dynamics do
that a cosine nearest-neighbour lookup over the same vectors does not?**

Five experiments, each with the baseline the original claim never had. Every
number below is measured at a pinned CPU thread count (see VIII.0, defect 2)
and stated with the resolution limit that applies to it.

## VIII.0 Three defects found while building these harnesses

These were found *while* measuring tickets 09–13 and each one invalidates
numbers published earlier, so they are recorded before the results that depend
on them.

### Defect 1 — softmax underflow silently ranked by insertion order

`PatternCompleter._settle` ranked candidates with
`torch.topk(trace.weights, k)`. In float32 a softmax entry below ~1e-38 flushes
to exactly 0.0, so at high inverse temperature most of the store ties at
exactly zero and `topk` fell back to **insertion order** for every rank below
the first. The retrieval "ranking" past position 1 was, in those cells, the
order the memories happened to be written in.

Fixed by ranking on `mhn.logits(trace.state, beta)` — identical ordering in
exact arithmetic, no underflow. Regression test
`test_ranking_survives_softmax_underflow_at_high_beta` in
`tests/test_hardening.py`, **verified to fail on the pre-fix code**; a test
that only passes after the fix would prove nothing here.

Measured cost of the bug (LoCoMo n=494, QMSum n=170):

| corpus | beta | questions affected | hit@10 before | after |
|---|---|---|---|---|
| LoCoMo | 128 | 85.4% | 0.383 | 0.439 |
| LoCoMo | 512 | 100% | 0.324 | 0.451 |
| QMSum | 512 | — | 0.371 | 0.676 |

**hit@1 is unchanged at every beta on both corpora** — the maximum weight never
tied, so only the ranking below the top was ever affected. That is why the bug
survived this long: every headline number in this project is a hit@1.

### Defect 2 — DG/SEPARATED numbers depended on the CPU thread count

The dentate-gyrus key thresholds on `h.topk(sparsity_k).values[..., -1:]`. A
unit sitting within float noise of that cut flips in or out when the matmul
reduction order changes — and the reduction order changes with the intra-op
thread count. The key itself was a function of how many cores torch happened to
grab.

Exact reproduction, LoCoMo 3 conversations, dim=1024, key=SEPARATED, beta=128:

| threads | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| 6 | 0.2976 | 0.3623 | 0.3988 |
| 8 | 0.2895 | 0.3502 | 0.4008 |

Deterministic at a fixed thread count, not portable across them. The spread
(0.008 at hit@1, n=494) sits inside this project's ~0.04 resolution limit, so it
overturns no conclusion — but without a pinned count **no third party can
reproduce a published DG number at all.**

The prediction that makes this a diagnosis rather than a story: both EMBEDDING
arms have no top-k selection, so every *settled* cell in them should be
**immune**. Re-running ticket 09's full grid at 6 threads against the earlier
32-thread run, on both corpora:

| arm | key | LoCoMo cells moved | QMSum cells moved |
|---|---|---|---|
| baseline | EMBEDDING | **0 — bit-identical** | 1 (see below) |
| whitened | EMBEDDING | **0 — bit-identical** | **0 — bit-identical** |
| DG (neg control) | SEPARATED | 10 | 16 |
| both | SEPARATED | 12 | 17 |

The prediction holds for all 40 settled EMBEDDING cells across both corpora. The
single QMSum exception is not a settled cell at all: it is the **cosine ceiling**
row, hit@10 0.8176 -> 0.8235, which is exactly one question of 170 changing
places at the rank-10 boundary.

That exception is worth stating rather than rounding away, because it narrows the
claim. Float reduction order can flip *any* ranking whose members are within
noise of each other, including plain cosine's -- it is not a DG-specific
phenomenon. What is DG-specific is the magnitude and the mechanism: the top-k
threshold changes the stored **key itself**, which is why the SEPARATED arms move
in 10-17 cells rather than one, and why they move at hit@1 where the cosine
ceiling does not.
Fixed by pinning, not by changing the library: `experiments/threads.py` adds a
`--threads` flag (default 6) to every DG-bearing harness, calls
`torch.set_num_threads()`, and records `torch.get_num_threads()` — the count
actually in force, not the one requested — in every log and JSON. `cls_memory`
itself is unchanged; a caller who wants reproducible DG keys is the one who has
to pin. `tests/test_threads.py` guards both the mechanism and the wiring.

### Defect 3 — a JSON crash lost a completed arm

`separation_beta_sweep._jsonable` emitted cells keyed by both `int` and `str`,
and `json.dumps(sort_keys=True)` raised `TypeError` after the first arm, killing
a 532-second LoCoMo run that had already computed a full arm's results. Fixed by
stringifying cell keys in `_cell_json`; the harness now writes after **every**
arm so a long run is recoverable.

## VIII.1 Separation × inverse temperature — the decisive grid (ticket 09)

The project's central hypothesis, stated so it can fail: *with separation raised
by whitening and pattern separation, does the Hopfield layer diverge from cosine
nearest-neighbour retrieval at a moderate inverse temperature — the regime where
the two are provably distinct — or does the tie survive even when the capacity
results' precondition is satisfied?*

Four separation arms × five betas, both corpora, with cosine kNN over the *same*
stored vectors as the ceiling in every cell. `experiments/separation_beta_sweep.py`.

**Best Hopfield score per arm, against its own cosine ceiling.** Each Hopfield
column is the maximum over betas *independently*, so a row may combine cells from
two different betas — it is the most generous reading of the layer available, and
it still loses:

| corpus | arm | cosine @1/@5/@10 | best Hopfield @1/@5/@10 | max rho_rank |
|---|---|---|---|---|
| LoCoMo (n=494) | baseline | 0.306 / 0.561 / 0.650 | 0.308 / 0.395 / 0.451 | 0.32 |
| LoCoMo | whitened | 0.342 / 0.579 / 0.648 | 0.342 / 0.433 / 0.484 | 0.20 |
| LoCoMo | DG (neg control) | 0.296 / 0.514 / 0.603 | 0.298 / 0.362 / 0.411 | 0.29 |
| LoCoMo | both | 0.310 / 0.528 / 0.595 | 0.318 / 0.362 / 0.383 | 0.24 |
| QMSum (n=170) | baseline | 0.324 / 0.694 / 0.824 | 0.335 / 0.541 / 0.682 | 0.51 |
| QMSum | whitened | 0.335 / 0.676 / 0.794 | 0.341 / 0.559 / 0.718 | 0.30 |
| QMSum | DG (neg control) | 0.276 / 0.629 / 0.782 | 0.271 / 0.524 / 0.653 | 0.43 |
| QMSum | both | 0.365 / 0.688 / 0.788 | 0.365 / 0.559 / 0.712 | 0.29 |

**The two corpora agree, in all four arms:**

1. **The tie at hit@1 survives — at usable temperatures.** At beta = 128 and 512
   every arm on both corpora ties its cosine ceiling at the top rank (exact paired
   p from 0.3438 to 1.0000), and the largest hit@1 difference there is +0.012, well
   inside the ~0.04 resolution limit. This is **not** true at every beta and should
   not be stated as though it were: at beta = 2 and 8 the layer loses hit@1
   outright with p = 0.0000 in all eight arm/corpus combinations, and at beta = 32
   the three DG-bearing arms still lose (LoCoMo DG d@1 -0.067 p = 0.0000, LoCoMo
   both p = 0.0001, QMSum DG p = 0.0075). The tie is a high-beta phenomenon.
2. **Below the top rank the layer loses, everywhere.** Against its own cosine
   ceiling the best Hopfield cell gives up **0.076 to 0.212** at hit@10 and 0.105
   to 0.166 at hit@5, in all eight arm/corpus combinations. The smallest gaps
   (0.076) are QMSum whitened and QMSum both; the largest (0.212) is LoCoMo both.
   There is no cell on either corpus where settling beats cosine below rank 1.
3. **The tie is not degeneracy.** `rho_rank` never exceeds 0.51. If the layer
   were cosine kNN in disguise the rank correlation would sit at 1.0; instead it
   genuinely reorders — top-10 overlap with cosine peaks at 0.44 — and the
   reordering is what costs the recall.
4. **Raising separation does not rescue it.** Whitening drives anisotropy to
   ~0.000 on both corpora, satisfying the capacity results' precondition, and the
   gap to cosine persists unchanged. Whitening lifts the cosine ceiling and the
   Hopfield arm by the same amount, so that gain belongs to embedding geometry,
   not to attractor dynamics.
5. **The negative control behaves as a control should.** DG raises key anisotropy
   (LoCoMo 0.041 -> 0.134, QMSum 0.056 -> 0.143) and loses recall on both corpora.
   The falsifiable prediction IV.4 offered — the arm that lowers separation should
   not beat the arm that raises it — held.

**Why the tie at rank 1 and the loss below it are the same fact.** After settling,
the state *is* the attractor, so ranks 2..k are ordered by similarity to the
attractor rather than to the query. The query's own information is spent getting
there. That predicts precisely what the grid shows: agreement about the single
best match (`top1=cos` reaches 0.955-1.000 at high beta) and degradation
everywhere below it.

**The answer, in one sentence:** *For retrieval, the Hopfield layer offers nothing
that cosine nearest neighbours does not — it ties at rank 1 by reducing to
attention over the same vectors, and every reordering it contributes below rank 1
makes recall worse, on both corpora, at every separation configuration tested
including the one that satisfies the capacity precondition.*

## VIII.2 Pattern completion on real data — the claim had no baseline (ticket 10)

Section 4 reports perfect completion from 26 of 205 active units on the
*synthetic* corpus. Part II already established that the synthetic retrieval
numbers were largely an artifact of the generator and retracted them on those
grounds; the completion number is the same class of number and had never faced
the same test. `experiments/completion_check.py` runs it on both real corpora —
and, critically, adds **the single-shot cosine arm on identical degraded cues**
that section 4 never had.

**LoCoMo, 400 cues per cell; QMSum, 1000 cues per cell. `delta` is completion
minus cosine, so negative means iterative settling lost:**

| arm / protocol | corpus | 80% | 50% | 30% | 20% | 10% | 5% kept |
|---|---|---|---|---|---|---|---|
| baseline (dense) coords | LoCoMo | +0.000 | +0.000 | +0.000 | +0.000 | -0.015 | **-0.215** |
| whitened (dense) coords | LoCoMo | +0.000 | +0.000 | +0.000 | +0.000 | -0.007 | -0.113 |
| DG (sparse) coords | LoCoMo | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | -0.182 |
| **DG (sparse) units** | LoCoMo | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | **+0.005** |
| baseline (dense) coords | QMSum | -0.001 | **+0.001** | -0.003 | -0.017 | -0.046 | **-0.209** |
| whitened (dense) coords | QMSum | -0.002 | -0.002 | -0.002 | -0.002 | -0.010 | -0.054 |
| DG (sparse) coords | QMSum | -0.002 | -0.002 | -0.003 | -0.003 | -0.011 | -0.207 |
| **DG (sparse) units** | QMSum | -0.002 | -0.002 | -0.002 | -0.002 | -0.001 | **+0.011** |

**The mechanism is not broken — that is what makes this decisive.** Convergence is
1.00 in every cell but one (0.999, QMSum DG coords at 5% kept), clamping is exact,
and settling takes 2.0-4.2 iterations, rising as the cue degrades exactly as the
energy picture predicts. The
network genuinely settles onto the right memory. Cosine simply already found it,
without iterating, from the same ruined cue.

**What survives and what does not.** The measurement in section 4 reproduces on
real data — in the sparse-key, active-unit protocol it is *better* than synthetic
(1.000 at 12.6 surviving units on LoCoMo, against synthetic's 0.775 at 13). So the
occlusion-robustness number itself is not retracted. What is retracted is the
interpretation it invited: that this demonstrates something attractor dynamics buy
you. Against the baseline it never had, completion beats cosine in **3 of 48 cells
across both corpora**: the most-degraded sparse-key/units cell on each corpus
(+0.005 LoCoMo, +0.011 QMSum), and one isolated QMSum dense-coords cell at 50%
kept (+0.001). Everywhere else it ties or loses, by as much as -0.215.

The two units cells are worth naming precisely rather than dismissing: they are the
*same* cell on both corpora, and it is exactly the regime section 4's original
claim was made in. The direction is consistent. The third (+0.001, a single
question either way at n=1000) is noise and is listed only so the count is honest. But the magnitudes are inside this
project's resolution limit, this harness computes no paired significance test for
completion-vs-cosine, and no other cell supports them. **A replicated direction
inside the noise floor is a reason to measure again, not a result.**

**Section 4 is corrected, not deleted:** the occlusion numbers stand; the claim
that they show iterative completion outperforming a one-shot lookup is withdrawn,
because it was never compared against one.
## VIII.3 Gist recall: the metastable regime does not exist here (ticket 11)

Schema absorption — the "complementary learning systems" idea the project is named
for — measured for the first time. `experiments/gist_check.py`.

The task is derived from the corpora's own human annotation, not generated: keep
only questions whose `evidence` set has >= 2 distinct turns, and score how much of
that *set* each arm recovers in its top-k. Identifying one memory cannot suffice
by construction. Three arms on identical queries: `gist()` (low-beta settle),
`recall()` (ordinary settle), and — the control the ticket demands — a plain
unweighted **centroid** of the cosine-top-5 keys, with no settling, no iteration,
no dynamics at all.

| corpus | recall | **centroid (no dynamics)** | gist @ best beta | gist when genuinely mixed |
|---|---|---|---|---|
| LoCoMo (n=423) | 0.130 | **0.192** | 0.110 (beta=32) | 0.062 (factor=0.15) |
| QMSum (n=165) | 0.068 | **0.070** | 0.064 (beta=32) | 0.043 (factor=0.15) |

**Gist loses to a plain average on both corpora**, and loses to ordinary
single-episode recall on LoCoMo. But the diagnostics are what settle the question,
because they show there is no operating point where the claimed mechanism is even
*present* alongside usable accuracy:

| corpus | beta | coverage | top1 weight | effective_n | is_mixture |
|---|---|---|---|---|---|
| LoCoMo | 2 | 0.001 | 0.007 | 326.2 | 1.000 |
| LoCoMo | 32 | 0.110 | 0.993 | 3.4 | 0.007 |
| LoCoMo | factor=0.15 | 0.062 | 0.464 | 166.9 | 0.544 |
| QMSum | 2 | 0.017 | 0.004 | 521.5 | 1.000 |
| QMSum | 32 | 0.064 | 0.935 | 1.8 | 0.079 |
| QMSum | factor=0.15 | 0.043 | 0.575 | 19.0 | 0.455 |

Every setting is one of three things and never the fourth. At low beta the state
collapses onto a near-uniform blend of the entire store (effective_n 326 and 522 —
that is the global centroid, which carries no information about the query). At
beta=32 coverage is at its best but `top1` is 0.99 and `effective_n` is under 4:
that is **single-episode retrieval wearing a mixture's name**, not a schema. The
one genuinely mixed setting scores worst of all. **A regime that is both a real
mixture and useful does not appear anywhere on either corpus.**

This reproduces exactly what `hippocampus.step`'s own docstring predicts, which is
why it is recorded as the expected honest result rather than a bug.

**Status table correction.** "schema absorption (CLS): not yet tested" becomes
**tested, and negative**: gist recall never beats a dynamics-free average of the
same neighbours, on either corpus, at any temperature — and where it is genuinely
metastable it is at its worst.

## VIII.4 Abstention: Part IV narrowed, not confirmed (ticket 12)

Part IV concluded "every signal is at chance, and the energy adds nothing over
cosine." That was measured at an inverse temperature where this project had
already proved the density degenerates into a monotone function of top-1 cosine.
Under those conditions the conclusion is not a finding about energy; it is the
arithmetic restated. `experiments/abstention_recheck.py` gives it the chance to
fail.

**Two protocols, and only one of them may speak to IV.1.** Protocol A calls
`abstention.collect()` unmodified with IV.1's exact hybrid BM25+BGE bare-MHN
factory — same write path, same store, so its numbers are comparable. Protocol B
runs the full `OrganizationalMemory` (LSA embeddings, cortex-derived key, salience
priors) and exists only to carry the DG negative control, which needs a key
encoder. **Protocol B is not comparable to IV.1 and cannot confirm or withdraw
it.** Out-of-fold evaluation, bootstrap CIs and Bonferroni correction across all
paired tests are retained throughout, as the ticket requires.

**The replication check passes exactly.** Protocol A at beta=128 reproduces IV.1's
published AUCs to ±0.000 (cos_top1 0.480, log_density 0.481, neg_depth_nats 0.481,
neg_settle 0.505), so what follows is a measurement of the same code path, not a
different experiment.

**Degeneracy is now visible per row, which is what the ticket asked for:**

Every cell, both protocols, `log_density` against `cos_top1`:

| protocol / arm | beta | rho(log_density, cos_top1) | delta AUC | survives Bonferroni? | energy adds (out-of-fold) |
|---|---|---|---|---|---|
| A baseline | 128 | **+0.9989** | +0.001 | no | -0.000 |
| A baseline | 32 | +0.8351 | +0.024 | no | +0.011 |
| A baseline | 8 | **+0.2423** | +0.038 | no | +0.009 |
| A whitened † | 128 | +0.9998 | -0.000 | no | +0.000 |
| A whitened † | 32 | +0.9948 | +0.001 | no | -0.003 |
| A whitened † | 8 | +0.8311 | +0.016 | no | -0.001 |
| B baseline | 128 | +0.9902 | +0.000 | no | -0.005 |
| B baseline | 32 | +0.8881 | +0.012 | no | -0.001 |
| **B baseline** | **8** | **+0.2852** | **+0.063** | **YES** | **+0.022** |
| B whitened | 128 | +0.9929 | +0.001 | no | -0.004 |
| B whitened | 32 | +0.9223 | +0.007 | no | -0.011 |
| **B whitened** | **8** | **+0.4049** | **+0.069** | **YES** | **-0.003** |
| B DG (neg control) | 128 | +0.9761 | +0.002 | no | -0.008 |
| B DG (neg control) | 32 | +0.7294 | +0.024 | no | -0.011 |
| B DG (neg control) | 8 | +0.2394 | +0.040 | no | +0.003 |

† **These three rows are not trustworthy and are shown only for completeness.**
`_iv1_whitened_factory` fits the whitener per conversation, on 419 samples in 4480
dimensions. `cls_memory/whitening.py` warns on exactly this and the warning fired
ten times in the run (`abstention_recheck_t6.log:59`): the covariance is
rank-deficient, so `transform()` projects onto a 419-dimensional subspace. Its own
docstring calls this "the transductive fit every other whitening measurement in
this project uses", but `separation_beta_sweep.build_embedder` — imported by the
same module — says the opposite, that fitting per conversation "is rank-deficient
and is the mistake that inflated RESULTS.md V.5". **The sweep is right and the
abstention harness is wrong.** No conclusion below rests on an A-whitened row.

**The verdict: Part IV's conclusion is narrowed to the high-temperature regime,
not confirmed in general.**

1. **At beta=128 the original finding stands, and now it is explained.** The
   energy correlates with top-1 cosine at rho = 0.999. It adds nothing because at
   that temperature it *is* cosine, relabelled. Part IV could not have detected an
   effect there even if one existed.
2. **At beta=8 the signals genuinely decorrelate** (rho falls to 0.24-0.29) and
   positive effects appear. In Protocol A they reach +0.038 to +0.040 AUC and win
   at the uncorrected 95% CI, but **none survive Bonferroni correction across 30
   tests**, so Protocol A does not establish an effect — it only shows Part IV's
   claim was never tested where it could fail.
3. **In Protocol B, four signals survive correction**, all at beta=8 and all in
   the two non-control arms: baseline `log_density` +0.063 [+0.001, +0.116] and
   `neg_settle` +0.057 [+0.015, +0.113]; whitened `log_density` **+0.069**
   [+0.010, +0.118] and `neg_settle` +0.056 [+0.008, +0.103]. **These are the only
   Bonferroni-surviving positive results anywhere in this review**, and the largest
   of them is the B-whitened one. They are also, by construction, not comparable to
   IV.1.

   But the out-of-fold column disagrees with the AUC column, and that matters: the
   B-baseline effect is corroborated out-of-fold (energy adds **+0.022**) while the
   larger B-whitened effect is **not** (-0.003). A signal whose paired AUC gain
   survives correction but which adds nothing to a held-out classifier is a weaker
   claim than its confidence interval makes it look. The corroborated result is the
   smaller one.
4. **The negative control behaves correctly.** DG at beta=8 shows the same effects
   smaller (+0.040, +0.042) and neither survives correction.

**The caveat that decides how much this is worth.** beta=8 is the temperature at
which *retrieval* collapses — LoCoMo hit@1 is 0.004 there (VIII.1). So the energy
carries abstention signal precisely where the system cannot retrieve anything. That
is not fatal, because abstention and retrieval are separate reads and nothing
forces them to share a temperature: a system could settle at beta=8 to decide
*whether* to answer and at beta=128 to decide *what* to answer. But that is a
design proposal this project has not measured, not a result. Protocol B also
carries a question-length confound at AUC 0.421, which is not controlled for here.

**What replaces Part IV's sentence:** *the energy adds nothing over cosine at the
operating point where it is a monotone relabelling of cosine, which is where Part
IV measured it; at low inverse temperature it decorrelates and shows a positive
effect that survives multiple-comparison correction in the full system but not in
IV.1's own protocol, at a temperature where retrieval itself does not work.*

## VIII.5 A learned read-in memorises rather than compresses (ticket 13)

Part VII's compression result was weakened by three things the ticket names: the
decoder's query projections were never trained to read a memory slot, the
compression was crude mean-pooling, and whitening cannot be applied inside a
frozen model. `experiments/kv_learned_readin.py` fixes the first two — a 249,600-
parameter `LearnedReadIn` trained by backpropagation through the frozen 135M
decoder — and re-runs Part VII's own metric with its matched-noise controls.

**The read-in trains, and on its training data it transforms the result:**

| arm | NLL | forced choice | margin | slots | compression |
|---|---|---|---|---|---|
| none (no memory) | 3.612 | 0.500 | +0.100 | 0 | — |
| `kv_super_all` (mean-pooled, untrained) | 3.625 | 0.562 | +0.085 | 1 | 610x |
| **`kv_learned_all`** | **1.284** | **0.875** | **+1.640** | 1 | 610x |
| `noise/learned` (matched control) | 12.001 | 0.375 | -0.199 | 1 | 610x |

**The full compression curve, re-run against the learned read-in with Part VII's
own metric and matched-noise controls** (`--pools 1,2,4,8,12,16,20,24`, the
published Part VII pool set):

| arm | NLL | forced choice | slots | compression | matched noise NLL |
|---|---|---|---|---|---|
| `kv_pool1` | 3.606 | 0.500 | 26 | 23.5x | 8.786 |
| `kv_pool2` | 3.632 | 0.500 | 52 | 11.7x | 9.975 |
| `kv_pool4` | 3.752 | 0.500 | 104 | 5.9x | 10.898 |
| `kv_pool8` | 3.988 | 0.312 | 208 | 2.9x | 11.712 |
| `kv_pool12` | 3.119 | 0.250 | 312 | 2.0x | 12.141 |
| `kv_pool16` | 2.700 | 0.250 | 414 | 1.5x | 12.821 |
| `kv_pool20` | 2.529 | 0.250 | 488 | 1.2x | 13.054 |
| `kv_pool24` | 2.243 | 0.312 | 530 | 1.2x | 12.678 |

The curve reproduces Part VII's shape: **forced-choice accuracy falls from 0.500 to
0.250 as pooling tightens**, and NLL improves past pool 12 only because the arm is
approaching the uncompressed cache (530 of 610 slots at pool 24 is 1.2x, not
compression). The matched-noise controls sit at 8.8-13.1 throughout, so the harness
is discriminating memory from noise exactly as it should.

**That row is contaminated and must not be read as a compression win.** It scores
all 16 situations, 12 of which the projection trained on. The honest number is the
held-out split:

| arm | train NLL | held-out NLL |
|---|---|---|
| `kv_learned_all` | **0.003** | **5.126** |
| `kv_super_all` (untrained) | 3.630 | 3.609 |

**The learned read-in generalises worse than not training at all** — 5.126 against
the untrained 3.609 — while driving training NLL to 0.003. 249,600 parameters
memorising 12 situations.

**This is robust to the hyperparameter, which is the only reason it is publishable.**
The first full run used lr=0.1 and *diverged* (loss 3.63 -> 5.86); reporting that as
"the mechanism fails to generalise" would have been an artifact of a bad optimiser
setting. Learning rate was then selected on **training-loss convergence only** —
never on the held-out metric, which would be tuning on the test set:

| lr | final train loss | train NLL | held-out NLL |
|---|---|---|---|
| 0.1 | 5.855 (diverged) | 5.760 | 7.645 |
| 0.03 | 0.093 | 0.089 | 4.833 |
| **0.01** | **0.003** | 0.003 | 5.126 |
| 0.003 | 0.017 | 0.016 | 5.058 |

Every converged setting generalises worse than the untrained baseline. The result
is the mechanism's, not the optimiser's.

**The zero-context-token claim is re-confirmed exactly and stays separate.** 610
tokens of rule text supplied as prefilled cache instead of prompt: **0 context
tokens, same 610 slots, dNLL -1.490e-07.** This never depended on compression or on
`LearnedReadIn`, and it remains the one exact positive result in this area.

**V.4's cost table and the capacity claim.** "100 rules -> 1 slot at 3600x" was
never a Hopfield capacity claim. Modern-Hopfield capacity (Ramsauer et al. 2021,
arXiv:2008.02217) concerns storing N patterns as N separate rows and retrieving
*one* by settling — `step`/`retrieve` in this repo. Summing N patterns into one
vector and reading several back out is `superpose`/`decode`, a different pair of
operations, and the literature that governs it is vector-symbolic architectures /
holographic reduced representations — Plate's circular-convolution binding and
vector-sum superposition (Plate, "Holographic Reduced Representations", IEEE
Transactions on Neural Networks, 1995), whose crosstalk bound sets reliable
unbinding far below what V.4's framing implied and requires near-orthogonal codes.
That is precisely what the whitening result in V.2 rediscovered empirically. The
ratio this measurement supports at one slot is **not** 610x usable capacity: it is
610x compression at the cost of the memory becoming indistinguishable from no
memory at all (3.625 against none's 3.612).

## VIII.6 Part III's table, re-measured — and the two causes separated

Part III's headline table predates review tickets 01-08 and defect 1. Every row of
it has moved. This section restates it at a pinned thread count and **separates the
two causes**, because "the numbers changed" is not a finding until you can say what
changed them.

The separation is possible because `separation_beta_sweep.py` reports both read
paths per cell: the fixed ranking and the **shipped** one (ranking by
`trace.weights`, which underflows). Running it at Part III's own configurations
gives the pre-fix and post-fix numbers on identical post-ticket-01-08 code.

**LoCoMo** — 3 conversations, 1451 turns, 494 questions:

| configuration | published | after tickets 01-08 | + defect-1 fix |
|---|---|---|---|
| was: LSA-256, separated, β=8 | 0.004 / 0.012 / 0.036 | 0.002 / 0.010 / 0.026 | unchanged |
| + β=128 | 0.154 / 0.235 / 0.291 | 0.221 / 0.279 / 0.354 | unchanged |
| + key=embedding | 0.172 / 0.235 / 0.310 | 0.217 / 0.289 / 0.340 | unchanged |
| **now: + LSA-1024** | 0.251 / 0.318 / 0.356 | 0.306 / 0.358 / 0.383 | **0.306 / 0.395 / 0.439** |

**The defect-1 fix contributes exactly 0.000 to the first three rows** — the
dim-256 grid's shipped and fixed rows are identical at both β=8 and β=128 (0 and
1.2% of questions tie respectively, and the ties change nothing). All of their
movement is tickets 01-08. Only the LSA-1024 row is shared, and there hit@1 is
entirely tickets 01-08 while the *whole* hit@5/hit@10 gain is the fix.

**QMSum** — 6 meetings, 3473 turns, 38 questions:

| configuration | published | now (6 threads) |
|---|---|---|
| was: LSA-256, separated, β=8 | 0.053 / 0.211 / 0.316 | 0.079 / 0.158 / 0.237 |
| + β=128 | 0.158 / 0.526 / 0.658 | 0.263 / 0.500 / 0.553 |
| + key=embedding | 0.211 / 0.421 / 0.579 | 0.211 / 0.553 / 0.684 |
| **now: + LSA-1024** | 0.395 / 0.632 / 0.658 | **0.447 / 0.658 / 0.684** |

At 38 questions one question is 0.026, so no QMSum row here resolves anything on
its own; the table is restated for consistency, not as evidence.

**Two honest qualifications:**

- The first two LoCoMo rows use the DG key, so a few thousandths of their movement
  is thread count rather than ticket work. That contribution is bounded at ~0.008
  by defect 2's measured 6-vs-8 spread, against movements of +0.067 — small, but
  stated rather than implied away.
- **The "63x" headline should not be restated as a multiple.** It now computes to
  153x on LoCoMo (0.002 -> 0.306), but the denominator is one question in 494. A
  ratio whose denominator is a single question is not a measurement. The absolute
  numbers are the result; the multiple never was.

Reproduce with:

```bash
PYTHONPATH=. uv run python -u experiments/recall_check.py --corpus locomo --locomo 3 --threads 6
PYTHONPATH=. uv run python -u experiments/recall_check.py --corpus qmsum --qmsum 6 --threads 6
PYTHONPATH=. uv run python -u experiments/separation_beta_sweep.py --corpus locomo --locomo 3 \
    --dim 256 --betas 8 128 --arms baseline "DG (neg control)" --threads 6
```
