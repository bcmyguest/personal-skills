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

58 tests pass. `examples/demo.py` runs the full lifecycle end to end. Everything
below labelled "measured" was measured in this repo, not assumed.

```bash
uv venv && uv pip install torch pytest
uv run pytest                       # 58 passed
PYTHONPATH=. uv run python examples/demo.py
```

---

## 1. Architecture

| Module | Role | Key type |
|---|---|---|
| `embeddings.py` | text → vector (pluggable; offline hashing default) | — |
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
E(ξ) = −β⁻¹ logsumexp_i( β·xᵢ·ξ + log wᵢ ) + ½‖ξ‖² + ½M²
```

With uniform wᵢ = 1/N this reduces exactly to the paper's eq. 4 — asserted in
`test_uniform_prior_matches_ramsauer_energy`.

**Update** (CCCP, monotonically non-increasing energy)

```
ξ ← Xᵀ softmax(β X ξ + log w)
```

Salience enters as a **log-prior on the logits**, not as a rescaling of the
pattern vector. Rescaling would distort the attractor geometry and corrupt the
energy; a log-prior is exactly a mixing weight in the Gaussian mixture the
network encodes (§6), which is the mathematically clean place to put it.

Also exposed: `separation(i)` = Δᵢ, Ramsauer's separation bound — retrieval
error is exponentially small in β·Δᵢ, so a small Δ flags a memory at risk of
fusing with a near neighbour.

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
exp(−βE(ξ)) ∝ Σᵢ wᵢ · exp(−β‖ξ − xᵢ‖²/2) · exp(β(‖xᵢ‖² − M²)/2)
```

With unit-normalised patterns the trailing factor is constant, so **the Gibbs
measure of the MHN is exactly a Gaussian mixture centred on the stored
memories, with σ² = 1/β** — which is precisely the noised marginal p_σ of a
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
   Measured at the **cue, not the settled state**: after settling the state is
   inside a basin by construction, which would report depth 0 for every query
   including nonsense. Demo: 0.370 for a real cue vs 0.678 for gibberish.
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
evergreen:  w = 1                                    (never decays)
temporal:   w = strength · 2^(−age_days / half_life)  (30-day half-life)
```

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

Replay runs in **latent** space, not key space: the decoder reads latents, and
the DG keys are a space it has never seen. `_latent_landscape()` rebuilds the
energy landscape over stored latents carrying the same priors. Patterns are
left unnormalised there (normalising would move them off the decoder's
manifold); the Gibbs measure is still a Gaussian mixture, with mixing weights
tilted by exp(β‖zᵢ‖²/2).

---

## 9. Known limitations

- **The `HashingEmbedder` is lexical, not semantic.** It exists so the demo runs
  offline. Retrieval quality is bounded by embedding geometry far more than by
  anything in the Hopfield layer — swap in `SentenceTransformerEmbedder` or an
  API encoder for real use.
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
