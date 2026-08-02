# CLS Organizational Memory — Technical Plan & Proof of Concept

An organizational memory system built on the **Extended Model** of Complementary
Learning Systems (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis &
McClelland 2016). PyTorch, CPU, no network access required.

```
                 ┌──────────────────────────────────────────┐
   text ──▶ embedder ──▶ NEOCORTEX (β-VAE) ──▶ surprise ──▶ gate
                 │            schema, slow             │       │
                 │                ▲                    │  > θ  │  ≤ θ
                 │                │                    ▼       ▼
                 │         consolidation        HIPPOCAMPUS  pruned
                 │         replay + interleave   (Modern     (schema
                 │                │              Hopfield)    covers it)
                 │                └───────────────────┘
                 └──────────────────────────────────────────┘
```

Two systems, deliberately mismatched in speed. The cortex learns the *statistics*
of organizational life slowly and with interleaving. The hippocampus stores
*individual episodes* in one shot. Novelty decides which one gets the write, and
consolidation moves knowledge from fast to slow, then releases the fast copy.

## Status

123 tests pass. `examples/demo.py` runs the full lifecycle end to end.
`experiments/benchmark.py` evaluates on a labelled synthetic corpus and
`experiments/benchmark_locomo.py` on **real conversational data with
ground-truth retrieval targets** — results in [RESULTS.md](RESULTS.md).

> Read RESULTS.md Parts II and III before quoting any number from Part I.
> On real data retrieval recall@1 is 0.251 (LoCoMo) and 0.395 (QMSum), against
> 1.000 on the synthetic corpus. Part III diagnoses and fixes a default that
> collapsed retrieval entirely. Everything below
labelled "measured" was measured in this repo, not assumed.

The package has been through an adversarial review; the defects it found and
the fixes are in §11.

```bash
uv venv && uv pip install torch pytest
uv run pytest                       # 123 passed
PYTHONPATH=. uv run python examples/demo.py
```

---

## 1. Architecture

| Module | Role | Key type |
|---|---|---|
| `embeddings.py` | text → vector (pluggable; fitted LSA default) | — |
| `neocortex.py` | β-VAE schema + novelty gate | slow, gradient |
| `pattern_separation.py` | dentate-gyrus key encoder | fixed, random |
| `hippocampus.py` | Modern Hopfield Network | fast, one-shot |
| `records.py` | evergreen/temporal records + forgetting curve | — |
| `store.py` | record ↔ pattern-row bookkeeping, decay sweep | — |
| `ingestion.py` | novelty-gated write path | — |
| `retrieval.py` | pattern completion read path | — |
| `energy.py` | MHN ↔ diffusion bridge, energy diagnostics | — |
| `consolidation.py` | replay, interleaved training, pruning | — |
| `system.py` | facade wiring all of the above | — |

---

## 2. The neocortex: β-VAE as a surprise estimator

Standard VAE over embeddings. Surprise is per-sample reconstruction error
(`mode="recon"`) or the full negative ELBO (`mode="elbo"`, a proper bound on
−log p(x)).

**`kl_weight` defaults to 0.01, not 1.0.** Embeddings are unit-norm, so the
reconstruction term is O(1) while KL is O(latent_dim). At `kl_weight=1` the KL
dominates, the posterior collapses, and the decoder emits the corpus mean —
reconstruction error then measures distance-from-mean rather than schema
violation. Measured gap between on- and off-schema text:

| `kl_weight` | on-schema | off-schema | separation |
|---|---|---|---|
| 1.0 | 0.752 | 0.919 | **0.083** |
| 0.1 | 0.359 | 0.815 | 0.456 |
| **0.01** | 0.271 | 0.839 | **0.568** |

The gate self-calibrates against a rolling quantile of recent surprise rather
than a fixed threshold, because the absolute scale drifts as the cortex trains.

> **Property worth knowing:** a quantile gate admits (1 − q) of its own
> calibration distribution by construction. At q = 0.8, ~20% of the training
> corpus will read as novel. That is arithmetic, not a defect — but it means
> "an exact training sentence is always pruned" is *not* a property this design
> has, and the tests assert the routing *rate* instead.

## 3. The hippocampus: Modern Hopfield Network

`hippocampus.py` implements Ramsauer et al. (2021, arXiv:2008.02217) with a
per-pattern log-prior added so salience can modulate retrieval:

**Energy**

```
E(ξ) = −β⁻¹ logsumexp_i( β·xᵢ·ξ − β‖xᵢ‖²/2 + log wᵢ ) + ½‖ξ‖²
```

With uniform wᵢ = 1/N and unit-norm patterns this reduces exactly to the paper's
eq. 4 — asserted in `test_uniform_prior_matches_ramsauer_energy`. The
per-pattern −β‖xᵢ‖²/2 replaces the paper's global +½M²; the two are identical
for unit-norm patterns (a constant factors out of the logsumexp) but only the
per-pattern form keeps `exp(−βE)` equal to the *untilted* Gaussian mixture when
patterns are not normalised — which is the configuration consolidation runs in.

**Update** (CCCP, monotonically non-increasing energy)

```
ξ ← Xᵀ softmax(β X ξ + log w)
```

Salience enters as a **log-prior on the logits**, not as a rescaling of the
pattern vector. Rescaling would distort the attractor geometry and corrupt the
energy; a log-prior is exactly a mixing weight in the Gaussian mixture the
network encodes (§6), which is the mathematically clean place to put it.

Also exposed: `separation(i)` = Δᵢ, Ramsauer's separation. **Its exponential
error bound assumes a uniform prior and therefore does not transfer to this
network** — a decayed memory can be geometrically well separated and still lose
retrieval. Use `effective_separation(i)` = Δᵢ + (log wᵢ − max_{j≠i} log wⱼ)/β,
which is the quantity that actually governs whether pattern i is a fixed point
here.

## 4. Pattern separation — a measured deviation from the brief

**The brief specifies storing the VAE latent in the MHN. That measured worst of
the available options, and the default deviates from it.**

The failure is structural. The cortex is trained on *routine* text, so its
encoder has no resolution in the region where novel items live — precisely the
items the hippocampus exists to hold. Two unrelated incidents landed at cosine
**0.905** in latent space and fused into one attractor.

CLS predicts this: the dentate gyrus performs pattern separation (sparse
expansion + decorrelation) specifically so similar episodes don't interfere.
`pattern_separation.py` implements it as a fixed random expansion followed by
k-winner-take-all. Measured on 5 incidents / 5 partial-cue queries:

| hippocampal key | correct recall | max off-diagonal cosine |
|---|---|---|
| VAE latent (as briefed) | 2/5 | 0.905 |
| raw embedding | 5/5 | 0.395 |
| **DG-separated, k=256** | **5/5** | 0.405 |
| DG-separated, k=16 | 4/5 | 0.059 |

The default is `HippocampalKey.SEPARATED`. Set `KeyConfig(mode=LATENT)` for the
original design — it remains selectable and tested. Note the sparsity trade-off:
lower k decorrelates far more but leaves too few active units shared with a
degraded query, so recall drops.

The expansion is **fixed and random, never learned** — a drifting projection
would silently invalidate every previously stored key.

## 5. Synaptic ingestion

`ingest()` routes in three steps:

1. **Near-duplicate check first.** Cosine ≥ 0.97 to an existing memory →
   `REINFORCED` (strengthen, don't duplicate), regardless of the gate.
2. **Evergreen bypass.** Business rules are stored even when unsurprising.
   A rule lost to a lossy reconstruction is not an acceptable failure mode for
   a system of record. Toggle with `IngestionConfig.always_store_evergreen`.
3. **Gate.** surprise > θ → `STORED`; surprise ≤ θ → `PREDICTED`, not stored.

## 6. Energy tracking via the MHN ↔ diffusion equivalence

Give the Hopfield energy a Gibbs measure p(ξ) ∝ exp(−βE(ξ)). Substituting E and
using β·xᵢ·ξ − ½β‖ξ‖² = −½β(‖ξ − xᵢ‖² − ‖xᵢ‖²):

```
exp(−βE(ξ)) ∝ Σᵢ wᵢ · exp(−β‖ξ − xᵢ‖²/2)
```

The per-pattern norm term in the logits absorbs the residual exactly, so **the
Gibbs measure of the MHN is exactly a Gaussian mixture centred on the stored
memories, with σ² = 1/β — at any pattern scale** — which is precisely the noised marginal p_σ of a
diffusion model over the memory set. Therefore:

- **β is not a free knob; it is the inverse noise level.** Low β = early
  diffusion time = broad schema basins. High β = late = sharp episodic
  attractors.
- **Closed-form score:** ∇log p_σ(ξ) = −β∇E(ξ) = β(Xᵀsoftmax(·) − ξ).
- **One MHN update = one exact denoising step.** By Tweedie/Miyasawa,
  E[x|ξ] = ξ + σ²∇log p_σ(ξ) = Xᵀsoftmax(βXξ + log w) — identically the
  Hopfield update. (Ambrogioni 2023; Hoover et al. 2023.)

All three are asserted numerically in `tests/test_hippocampus.py`
(`test_update_equals_tweedie_denoiser`, `test_gibbs_measure_is_the_gaussian_mixture`).

This buys three concrete things, not just an analogy:

1. **`log_density`** — a properly normalised log p, comparable across memory
   sets of different sizes, where raw energy is only defined up to a constant.
2. **`basin_depth`** — retrieval confidence / confabulation detection.
   Measured at the **cue** for free-text queries (after settling the state is
   inside a basin by construction, reporting depth 0 for everything including
   nonsense) and at the **settled state** for masked cues (an occluded cue is
   far from every memory by construction, which would flag perfect completions).
   Reported in nats via `depth_nats`, so a cutoff means the same thing at any β.
   The ranking is reliable; the absolute cutoff needs per-deployment
   calibration — see RESULTS.md §3.
3. **`langevin_replay`** — principled hippocampal replay for consolidation,
   sampling the smoothed memory distribution with the exact score.

### Negative result: annealed retrieval does not work

Coarse-to-fine β scheduling is the obvious thing to try given the equivalence.
It was implemented and measured, and it **loses**: on clustered patterns it
retrieved correctly 17/40 against 31/40 for the plain one-step update. Low-β
passes pull the state toward cluster means, discarding exactly the cue detail
that distinguishes memories within a theme. Annealing belongs on the sampling
side (`langevin_replay`), not the retrieval side. The code was removed rather
than shipped.

## 7. Evergreen vs temporal

```
evergreen:  w = max_strength                          (never decays)
temporal:   w = strength · 2^(−age_days / half_life)  (30-day half-life)
```

Evergreen sits at the *ceiling*, not at 1.0. With evergreen pinned to 1.0 a
temporal memory recalled often enough reached strength 3.0 and outranked every
business rule — the opposite of the intended guarantee.

Decay is applied **lazily at read time** (`refresh_priors`) — one tensor write,
no background job, always time-accurate. Recall triggers reconsolidation:
`access_count += 1`, strength rises, decay clock resets.

Evergreen records are protected from *both* the forgetting curve and
consolidation pruning. The demo shows 8 → 3 memories after 200 simulated days,
with all three business rules intact.

## 8. Consolidation

1. **Replay** — Langevin sampling of the memory landscape with the exact score,
   seeded from salience-weighted patterns so decayed episodes consolidate more
   weakly.
2. **Interleaved training** — replay mixed with new data. This is the mechanism
   that prevents catastrophic forgetting, and the whole reason biology bothers
   with a two-speed design. Verified in
   `test_interleaved_replay_resists_catastrophic_forgetting`.
3. **Prune the now-predictable** — memories the updated cortex reconstructs
   below threshold have been absorbed into semantic knowledge; the hippocampal
   trace is released. Evergreen exempt.

### Replay is two mechanisms, and conflating them broke the loop

Replay originally returned `decode(latents)` only, and that is a *no-op for
learning*: the reconstruction target is already what the model emits. Measured,
`decode(latent)` of a stored anomaly had cosine **0.047** with the embedding it
was supposed to be reinstating (the decoder is trained on routine text and has
no resolution where novel episodes live), and the cortex's surprise on its own
decoder output was **0.0001** against 1.15 on the real memories. Consolidation
therefore left surprise on the stored memories at **1.0016×** its value at
ingestion after 150 epochs, and pruning — correctly — never fired.

The batch is now a mixture, because the two halves do different jobs:

| component | space | what it does |
|---|---|---|
| **episodic** (`episodic_ratio`) | embedding | noised samples around the *true* stored embeddings. The only part that teaches the cortex anything. |
| **generated** (the rest) | latent → decoder | pseudo-rehearsal (Robins 1995): the cortex trained on its own output. Teaches nothing; pins the existing schema. |

Neither alone works. Measured on the full benchmark (30 episodic memories,
drift reduction against naive training on off-schema data):

| `episodic_ratio` | drift reduction | pruned @20 ep | pruned @40 ep |
|---|---|---|---|
| 0.000 (the old behaviour) | 55.3% | 0 | 0 |
| **0.125 (default)** | **54.9%** | **6** | **16** |
| 0.250 | 50.3% | 15 | 21 |
| 0.500 | 46.0% | 23 | 29 |

0.125 is the knee. Its 0.4-point drift cost is inside run-to-run noise — the
same measurement has read 57.1%, 56.1% and 55.3% across reruns — and it is the
smallest share that closes the loop at all. Raise it deliberately if draining
the fast store matters more than schema stability, at roughly one point of
drift protection per 2–3 extra memories released.

The generated component runs in **latent** space, not key space: the decoder
reads latents, and the DG keys are a space it has never seen.
`_latent_landscape()` rebuilds the energy landscape over stored latents
carrying the same priors. Patterns are left unnormalised there (normalising
would move them off the decoder's manifold); the Gibbs measure is still a
Gaussian mixture, with mixing weights tilted by exp(β‖zᵢ‖²/2).

The episodic component runs in **embedding** space, where the cortex actually
trains, via `_embedding_landscape()`. Its noise level is `replay_sigma=0.05`,
**not** the hippocampal β. β is chosen for retrieval sharpness over 2048-d
sparse keys; reused as a noise level over 256-d unit-norm embeddings it gives
σ = 1/√32 = 0.177, whose samples have norm 3.41 and cosine 0.295 to the nearest
stored memory — the same high-dimensional Gaussian-mixture failure that
`langevin_replay`'s damped seed noise already works around. Measured cosine to
the nearest stored memory: 0.939 at σ=0.02, 0.737 at 0.05, 0.481 at 0.1.

The ULA step size is capped at σ² there. Unadjusted Langevin over a Gaussian of
precision β = 1/σ² = 400 contracts by (1 − βη/2) per step and **diverges** for
η > 4/β = 0.01; the configured `replay_step_size` of 0.05 is five times that
and blows the samples up by ~1300× (asserted in
`test_replay_is_numerically_stable_at_the_configured_noise_level`).

---

## 9. Known limitations

- **Retrieval is bounded by the embedder.** After the Part III fixes, recall@1
  is 0.251 on LoCoMo and 0.395 on QMSum against 1.000 on synthetic data. Sparse
  TF-IDF still beats the best dense configuration on its own ceiling (0.320 vs
  0.255), so a sparse-shortlist hybrid is the obvious unimplemented next step.
  Swap in a sentence encoder for real use.
- **The novelty gate is partly a length filter on real text**: correlation
  between surprise and turn length is −0.48 on LoCoMo, where the synthetic
  corpus (uniform-length documents) hid the effect entirely.
- **The 30-day half-life is domain-specific.** On LoCoMo's 231-day spans it
  discards 13.8% of the corpus and costs 1.6 points of recall.
- **Deletion is O(N·d).** `remove()` compacts the pattern tensor. Fine at PoC
  scale; batch deletions or use a tombstone + periodic compaction beyond ~10⁵
  memories.
- **No persistence layer.** Everything is in-memory. Records are plain
  dataclasses holding tensors, so serialization is straightforward but unwritten.
- **Single-tenant, no access control.** An organizational deployment needs
  per-memory ACLs enforced *before* retrieval — filtering after settling leaks
  information through the attractor.
- **Half-life is global.** Real organizations want per-source half-lives (a
  Slack message and a post-incident review should not decay alike).
- **Consolidation pruning trusts the VAE.** A lossy reconstruction below
  threshold is not proof the content is recoverable. Evergreen protection
  mitigates the worst case; a text-level round-trip check would be stronger.
  The best available evidence that `relative_drop=0.5` is not too loose is
  indirect: released memories reconstructed at cosine 0.843 to their own
  embedding, and at worst 0.496 absolute surprise against 0.798 mean for
  routine documents under the same cortex — i.e. better predicted than the
  schema's own training data.
- **Consolidation pruning is budget-sensitive**, far more so than any other
  mechanism here. The criterion is a step function on a quantity that moves
  gradually, so a pass that has consolidated most of the way still releases
  nothing: on the full benchmark, 0 of 30 at 5 epochs, 6 at 20, 16 at 40, 30 at
  60. Benchmark experiment 7 prints the whole curve plus the median drop ratio
  for this reason. Do not assume the hippocampus drains — read the sweep.
- **A pure `sleep()` with no new data drifts the schema** (+0.5 to +0.7 on the
  full benchmark). This is *not* caused by episodic replay — the old
  decoder-only replay drifted +0.48 in the same pass while learning nothing,
  because its low-norm decoder outputs pull the VAE toward that region.
  Consolidation is meant to be interleaved with real new data; `sleep(None)` is
  a diagnostic, not an operating mode.
- **Replay spends part of its budget on records that can never be pruned.**
  Seeding is salience-weighted, and evergreen sits at `max_strength`=3.0 against
  1.0 for a fresh episode — so on the benchmark's 6 evergreen / 30 episodic
  store, evergreen draws ~37% of replay while being exempt from pruning. It is
  defensible (business rules genuinely should become semantic knowledge, they
  are merely also kept verbatim) but it is not free, and it is why a small
  `replay_batch` stalls consolidation: at 64 samples over 6 memories the
  episodes get under one sample each. `episodic_ratio` interacts with
  `replay_batch` and the evergreen fraction, not just with epochs.
- **Roughly half the store still does not drain.** 14 of 30 episodic memories
  survive a converged pass and more epochs do not release them — the ceiling is
  set by `episodic_ratio`, and raising it costs schema protection. No setting
  measured here both drains the store fully and keeps drift protection intact.
- **`langevin_replay` is not annealed** despite the diffusion framing, and its
  seed noise is deliberately damped (`init_noise=0.1`): a full σ=1/√β isotropic
  seed has radius σ√d, which in 1024-d exceeds the spacing between memories and
  washes salience-weighted seeding into noise around the centroid.

## 10. Suggested next steps

1. Replace the embedder and re-measure §4 — a semantic encoder may narrow the
   latent-vs-separated gap, and `separation()` tells you directly.
2. Tune `sparsity_k` against the capacity/robustness trade-off on real data.
3. Add persistence + ACLs before any pilot.
4. Instrument `basin_depth` in production as a confabulation guard on retrieval.

## References

- Ramsauer et al. (2021). *Hopfield Networks is All You Need.* arXiv:2008.02217
- McClelland, McNaughton & O'Reilly (1995). *Why there are complementary
  learning systems in the hippocampus and neocortex.* Psych. Review 102(3).
- Kumaran, Hassabis & McClelland (2016). *What Learning Systems do Intelligent
  Agents Need? Complementary Learning Systems Theory Updated.* TiCS 20(7).
- Ambrogioni (2023). *In search of dispersed memories: Generative diffusion
  models are associative memory networks.* arXiv:2309.17290 — shows the energy
  function of a diffusion model trained on discrete patterns is asymptotically
  identical to that of a modern Hopfield network.
- Hoover, Strobelt, Krotov, Hoffman, Kira & Chau (2023). *Memory in Plain Sight:
  Surveying the Uncanny Resemblances of Associative Memories and Diffusion
  Models.* arXiv:2309.16750
- Miyasawa (1961) / Efron (2011). *Tweedie's formula* — posterior mean denoising.

---

## 11. Review findings and fixes

The package was reviewed adversarially, with every claim checked numerically
rather than by reading. The review **confirmed** the core mathematics: the
Ramsauer energy with a log-prior, CCCP descent (including under masking and
non-uniform priors), the Gibbs/Gaussian-mixture identity, the normalisation of
`log_density` (numerically integrated to 0.99999999999996), the Tweedie identity
(agreement to 5e-16 in float64), and the ULA step. The equivalence in §6 is real.

The defects were at the seams. All of the following were reproduced, fixed, and
are now covered by `tests/test_review_regressions.py`:

| # | Defect | Fix |
|---|---|---|
| 1 | `score`/`denoise` and `log_density` described **different distributions** when `normalize_patterns=False` — the configuration consolidation runs in. Gradient norms diverged by 24; the Gibbs measure was norm-tilted, distorting replay mixing weights by 21×. | Folded −β‖xᵢ‖²/2 into the logits, replacing the global +½M². Behaviour-preserving for unit-norm patterns, exact at any scale. |
| 2 | `separation()` carried Ramsauer's exponential-error claim, which assumes a uniform prior. A memory with log-prior −25 reported Δ=0.85 ("healthy") but was not a fixed point. | Added `effective_separation()`; corrected the docs. |
| 3 | `basin_depth().depth` documented as ">0 always"; measured −0.004 at a mixture point, so `is_confabulation` **failed open** exactly at ambiguous cues. Its 0.5 cutoff was also a raw energy, not portable across β. | Documented the true sign; added scale-free `depth_nats`; made thresholds configurable; fixed the masked-cue path. |
| 4 | `prune_predicted` compared against the **live gate quantile**, which drifts upward as new items arrive — measured deleting 1 of 4 memories with the cortex untouched since those memories were written. | Switched to a relative-drop criterion against each record's surprise at ingestion. Pruning can now only fire if this cortex actually improved on this item. |
| 5 | Consolidation trains the cortex, silently invalidating latent-derived keys. In `LATENT` mode, the same text encoded to cosine **0.327** against its own stored key after one `sleep()`. | `MemoryStore.reindex()` re-encodes every key after training. |
| 6 | `record.key` **aliased** `record.latent`; `store.keys()` disagreed with `mhn.patterns` for non-unit-norm keys; no validation on duplicate ids or key dimension. | Clone on init, write the normalised pattern back, validate on `add`. |
| 7 | `gist()` reinforced by default, so a schema-level read bumped the strength and reset the decay clock of whichever memory happened to top a metastable mixture. Basin was also computed *after* reinforcement, so trace and basin used different priors. | `reinforce=False` for `gist`; basin measured before reinforcement. |
| 8 | `mode="elbo"` returned `recon + kl_weight·kl` — a β-VAE objective, not a bound on −log p(x). | Uses the full KL. |
| 9 | Evergreen salience was 1.0 while a reinforced episode could reach 3.0, inverting the "evergreen always wins" guarantee. | Evergreen sits at `max_strength`. |
| 10 | **Consolidation pruning never fired** — 0 of 36 memories released. The criterion (defect 4) was correct; `replay()` was empty. Training the VAE on `decode(latents)` is self-distillation: measured cosine **0.047** between a decoded latent and the embedding it came from, surprise **0.0001** on the cortex's own output, and surprise on the stored memories still at **1.0016×** its ingestion value after 150 epochs. The hippocampus could not drain, which is the entire justification for a two-store architecture. | Replay is now a mixture: `episodic_ratio` (0.125) sampled in *embedding* space around the true stored embeddings, the rest generated as before. Pruning goes 0 → 16 of 30 at a 40-epoch pass; drift reduction 55.3% → 54.9%, inside run noise. |

Test-quality problems the review found, also fixed: two consolidation tests
passed `threshold=float("inf")`, which prunes unconditionally and tested nothing
about learning; `test_partial_cue_completes_the_missing_half` asserted cos > 0.8
where the true value is 1.0 and a degenerate mean-returning implementation
scores 0.35; `test_basin_depth_flags_an_out_of_distribution_query` compared two
values that were both exactly 0.5; and two assertions were true by construction.
Coverage was added for energy descent under priors and masking, the `log_density`
normalising constant, and the replay *distribution* rather than just its shape.

## 12. Hardening review

A second review probed robustness rather than mathematics. It confirmed a good
deal was already sound — deduplication, empty-store handling on every entry
point, LSA determinism, the randomised SVD's numerics, bounded gate window, no
mutable defaults, no aliasing in returned collections. Fixed since, each
reproduced before being changed and now covered by `tests/test_hardening.py`:

| Defect | Why it mattered | Fix |
|---|---|---|
| One NaN/Inf pattern poisoned `energy`, `attention`, `retrieve`, `log_density` and replay for the **whole store, every query, permanently** — silently | unrecoverable without hand-locating the row | reject non-finite rows at `write` |
| Degenerate text (empty, whitespace, punctuation, emoji, CJK, pure OOV) produced zero embeddings; dedup cannot catch them (cosine of two zero vectors is 0) and their logit is `log(w)` regardless of query, beating any memory below cosine 0.5 | one junk record hijacked retrieval for unrelated queries at weight 0.985 | new `IngestionAction.REJECTED`; `write` rejects zero-norm |
| `MemoryStore.reindex` removed before writing | a bad `reencode_key` left records with zero patterns — store permanently unusable, reachable automatically from `consolidate` | build and validate first, restore on failure |
| Seeding was `torch.manual_seed`, i.e. process-global | constructing a second system silently changed an existing one's training and replay | per-instance `torch.Generator`, `fork_rng` for module init |
| `write` reallocated the whole buffer per call | O(N²): doubling N cost 3.6×, hours of copying at 100k | geometric-growth capacity buffer — now **1.9× per doubling**, 8000 writes 37.9s → 0.78s |
| Generated replay did not cap its ULA step (the episodic half already did) | at β=128 samples blew up 1e6× and were trained on with `improved=True` reported | same `4/β` bound both sides |
| `consolidate` guarded *after* `torch.cat([])` | `sleep()` on a fresh system died with an opaque torch error | guard before |
| LSA fit allocated a dense N×V matrix | 2.1 GB peak at 10k docs, ~8 GB at 100k | sparse COO + `torch.sparse.mm` |
| `relative_drop >= 1` accepted | silently reintroduced an already-fixed defect (pruned an untouched cortex) | validate all config at construction |
| `bootstrap()` did not re-index, unlike `consolidate` | calling it twice staled LATENT keys | re-index when the store is non-empty |

Also fixed: store/pattern desynchronisation now raises via `assert_consistent`;
`remove` deduplicates ids; iteration is snapshotted against concurrent removal;
`replay` honours `now` instead of wall-clock; `gist(beta=...)` no longer raises;
config is deep-copied rather than mutated in place; `sweep` and `stats` agree on
the empty-store sentinel; and several loader bugs (a date fallback that raised,
`span_days` on empty input, an O(Q·E·T) rebuild, blank-`dia_id` collisions).

Known and accepted, not fixed: `MemorySystemConfig.device` is ignored (CPU
only), `basin_depth` is single-query, `PatternCompleter.occlude` is evaluation
tooling on the read path, and `record.key` duplicates its pattern row (dropping
it in favour of a view would halve key memory).
