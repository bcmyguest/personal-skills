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

| arm | coverage@256 | tokens@cover | stale@256 |
|---|---|---|---|
| BM25 | 0.92 | 115 | 0.81 |
| BGE kNN | 0.94 | 76 | 0.88 |
| BGE + recency rerank | 0.94 | 83 | 0.81 |
| MHN, no decay | 0.94 | 76 | 0.88 |
| MHN + recency log-prior | 0.94 | 81 | 0.88 |
| **INGEST-GATED kNN (t=0.75)** | 0.94 | 115 | **0.00** |
| ORACLE (current-only kNN) | 1.00 | 58 | 0.00 |

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
   takes `stale@256` from 0.88 to **0.00** — matching the oracle.

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
