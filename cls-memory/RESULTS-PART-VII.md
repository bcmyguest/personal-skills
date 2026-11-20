# Part VII — The end-to-end demo, unblocked

`RESULTS.md` closes with a status table whose first row reads:

> | memory as K/V inside attention | **proved exactly** (0.00e+00); end-to-end
> demo blocked — HuggingFace is 403 by proxy policy, so no decoder LM is
> reachable here |

That environment constraint no longer holds. `huggingface.co` answers 200 on
both the API and `resolve/main`, so the demo is now runnable, and this part
runs it.

**Setup.** `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M params, 30 layers,
9 attention heads over 3 KV heads, head_dim 64, float32 on CPU) against the
26-rule corpus in `experiments/rulebook.py`. Reproduce with:

```bash
uv venv && uv pip install torch pytest transformers huggingface_hub numpy
PYTHONPATH=. .venv/bin/python experiments/kv_injection.py --pools 1,2,4,8,12,16,20,24
```

Runtime ~4 min on CPU. Raw numbers in `experiments/kv_injection_results.json`.

**Baseline check first.** With `fastembed` + `onnxruntime` installed the suite
is **151 passed**, reproducing VI.4 exactly. Without `onnxruntime` it is 136
passed, 15 skipped — and the 15 skips are precisely the real-BGE tests that
Part V and VI rest on. The suite reports success while testing none of the
claims it was written to hold in place.

---

## VII.1 "Zero context tokens" is exactly true

The two arms below are the **same 610-token sequence**, split differently:
`text` puts the rules in the prompt; `kv_full` prefills them, keeps the cache,
and never supplies the tokens again.

| arm | prompt tokens | recitation NLL |
|---|---|---|
| `text` | 610 | 0.9416318 |
| `kv_full` | **0** | 0.9416320 |

**ΔNLL = −1.49e-07** — float32 round-off, the same order as V.1's 8.94e-08.
The claim holds end-to-end on a real decoder, at the precision V.1 was argued
at, and it is worth stating plainly because it is the part of Part V that
survives contact with a transformer intact.

*Metric.* `recitation NLL` is the mean token NLL of the gold rule's own text
continuing the situation prompt. It asks only "did the content arrive", not
"can the model reason", which matters because a 135M model cannot do the
latter (VII.5).

## VII.2 The compression claim fails at every ratio

`kv_full` costs 610 KV slots. It is prefix caching: the tokens leave the
prompt, the slots do not. V.4 prices the mechanism at 100 rules → **1 slot**,
a 3600× saving. Testing that directly, by mean-pooling each rule's cache down
to *n* slots and by superposing all 26 rules into one:

| arm | slots | compression | NLL | % of the `kv_full` gain recovered |
|---|---|---|---|---|
| `kv_full` | 610 | 1.0× | 0.942 | **100%** |
| `kv_pool24` | 530 | 1.2× | 2.243 | 51.3% |
| `kv_pool20` | 488 | 1.3× | 2.529 | 40.6% |
| `kv_pool16` | 414 | 1.5× | 2.700 | 34.1% |
| `kv_pool12` | 312 | 2.0× | 3.119 | 18.4% |
| `kv_pool8` | 208 | 2.9× | 3.988 | −14.1% |
| `kv_pool4` | 104 | 5.9× | 3.752 | −5.2% |
| `kv_pool2` | 52 | 11.7× | 3.632 | −0.7% |
| `kv_pool1` | 26 | 23.5× | 3.606 | 0.2% |
| **`kv_super_all`** | **1** | **610×** | 3.625 | **−0.5%** |
| `kv_super_k8` (k=8, gold included) | 1 | 610× | 3.645 | −1.2% |
| `none` | 0 | — | 3.612 | 0% by definition |

Read the last column, not the NLL column. **Nothing below 2× compression
recovers anything.** Superposition into one slot — the V.2 mechanism, the
thing V.4 prices at 3600× — lands at −0.5%, i.e. indistinguishable from having
no memory at all. Even one slot per rule (23.5×, still 26 slots) recovers
0.2%. You must keep ~87% of the slots to retain half the benefit.

The curve is also **non-monotonic**: `kv_pool8` (2.9×) is *worse* than no
memory, while `kv_pool1` (23.5×) is merely inert. Mild pooling leaves enough
structure to distract and not enough to inform; heavy pooling produces a blur
attention learns nothing from and mostly ignores.

**These arms are not failing because the injection is broken.** Each has a
noise control with identical slot count and per-layer scale:

| slots | pooled memory | matched Gaussian noise |
|---|---|---|
| 26 | 3.606 | 8.786 |
| 52 | 3.632 | 9.975 |
| 104 | 3.752 | 10.898 |
| 208 | 3.988 | 11.712 |
| 1 | 3.625 | 3.888 |

Noise is catastrophic; pooled memory is benign. The mechanism delivers
*something* — it is simply not the rule. Being far better than noise is not
evidence of content, which is the shape of error HANDOFF §5 catalogues.

## VII.3 Two reasons, one of them structural

**Anisotropy, worse than BGE.** V.2 diagnosed superposition collapse as a
geometry problem and measured it on sentence embeddings. The same measurement
on the model's own pooled key vectors, per layer:

| space | mean off-diagonal cosine between unrelated memories |
|---|---|
| SmolLM2 keys, layer 0 | **+0.984** |
| SmolLM2 keys, mid | +0.964 |
| SmolLM2 keys, last | +0.967 |
| SmolLM2 keys, mean over 30 layers | **+0.969** |
| BGE as shipped (V.2) | +0.649 |
| BGE whitened (V.2) | −0.001 |

A frozen decoder's key space is *far* narrower a cone than BGE's. V.2's fix
does not transfer: whitening the keys changes every `q·k` the frozen query
projections were trained to produce, so the transform that rescues capacity
also destroys the scores that would use it.

**The decode step does not exist.** This is the harder objection and it is
algebra, not measurement — the V.1 style. V.2 decodes a superposition with
`X @ state`: logits of the held state against *the store's individual rows*.
A single K/V slot in a frozen LM has no store to score against. Softmax over
one element is 1, so the memory contributes `1 · V_super` — one fixed vector
per layer and head, identical for every query. A superposed slot cannot be
query-addressed, because the operation that addresses it was the MHN's own
logit computation, and that is exactly what injecting into attention throws
away.

V.3 already found that settling destroys a superposition. VII.3 is the mirror
image: **holding** a superposition works only while you retain the decoder,
and "put it in the model's K/V" is precisely the move that discards it.

## VII.4 What V.4's table should say

V.4 as published:

| rules applied | as text | as K/V | ratio |
|---|---|---|---|
| 100 | 3600 tokens | 1 slot | 3600× |

Measured, the honest version is two separate claims:

| claim | verdict |
|---|---|
| context **tokens** → 0 | **holds exactly** (ΔNLL −1.49e-07) |
| KV **slots** → 1 | **fails**: 610× compression recovers −0.5% of the benefit |

The saving is real but it is the prefix-KV-cache saving — you stop re-sending
tokens, you do not stop storing them. That is worth having and it is not 3600×;
at these ratios it is 1×. **V.4's "ratio" column conflates a token cost with a
memory cost and should be retracted.**

## VII.5 The forced-choice metric resolved nothing — reported anyway

The second metric asked whether the gold rule out-scores its traps (superseded
versions, wrong-scope siblings). It sits at chance in every arm, **including
`text`, where the full rulebook is in the prompt**:

| arm | forced choice (n=16) |
|---|---|
| `none` | 0.500 |
| `text` | 0.500 |
| `kv_full` | 0.500 |
| `kv_super_all` | 0.562 |
| noise, 1 slot | 0.562 |

With n=16 the standard error is ≈0.125, so every cell here is one number.
A 135M model cannot tell `exp.approval.v3` from `exp.approval.v1` even with
perfect context, so this metric cannot discriminate *arms* — it discriminates
model capacity, which was not the question. It is reported because omitting a
metric that failed to separate anything is how the "checks that flatter the
mechanism" pattern in HANDOFF §5 gets started. Every VII.1–VII.4 conclusion
rests on recitation NLL alone.

**Other limits of this part.** One model, one size, one corpus; the rulebook is
hand-written synthetic data, which §5 warns against. Mean-pooling is the
crudest possible KV compression — a *trained* compressor (Gist tokens, AutoCompressor,
KV-cache distillation) is a different and untested question. What VII.2 rules
out is the specific untrained mechanism this project proposed: normalised
summation, `ModernHopfieldNetwork.superpose`, transplanted into attention.

## VII.6 Status of the three claims, revised

| claim | RESULTS.md | after Part VII |
|---|---|---|
| memory as K/V inside attention | proved exactly; demo blocked | **demo run.** Token claim exact; slot claim false |
| N rules → one vector | holds after whitening: 8 lossless | **holds in BGE space only.** In a frozen decoder's key space: 0% at every ratio, and unwhitenable |
| schema absorption (CLS) | not yet tested | still not tested |

The Part IV conclusion survives and strengthens: the valuable component is the
**write-time ingestion gate** (0.88 → 0.00 stale), not the retrieval layer and
not the substrate. Part V found the MHN is an attention head; Part VII finds
that the reverse embedding — putting the memory inside an attention head — buys
the token saving of a KV cache and none of the compression the design was
pitched on.
