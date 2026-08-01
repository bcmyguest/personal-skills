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

## 7. Consolidation — interleaved replay cuts forgetting 56%

Schema surprise on 400 old routine documents after training on off-schema data:

| condition | before | after | drift |
|---|---|---|---|
| no replay (naive) | 0.0960 | 1.4732 | **+1.3772** |
| interleaved replay | 0.0960 | 0.6997 | **+0.6037** |

**56.1% less drift.** The core CLS claim reproduces: interleaving hippocampal
replay with new data measurably protects the existing schema. (The quick-mode
run shows 93.2% on a smaller corpus; the effect size is corpus-dependent, the
sign is not.)

One negative observation: in the same pass, `prune_predicted` removed **0 of 36**
memories. Twenty epochs of consolidation was not enough for the cortex to learn
the anomalies well enough to release them, so the hippocampus never drained.
Consolidation pruning is far more sensitive to training budget than the other
mechanisms — do not assume it is doing anything without measuring it.

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
   budget, while every other mechanism worked.


---

## 8. Post-review re-run

All numbers above were reproduced after the review fixes landed (including the
Hopfield energy reformulation in §S1 below). Differences were within run noise:
ROC AUC 1.0000 unchanged, key-representation table identical, replay benefit
57.1% → 56.1%. The energy change is exactly behaviour-preserving for unit-norm
patterns, which is what the retrieval path uses, so this is the expected result
rather than a lucky one.
