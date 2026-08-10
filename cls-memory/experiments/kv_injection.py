"""Does a memory survive as a K/V pair inside a real decoder LM?

RESULTS.md V.1 proves `softmax(beta X xi) X` is bit-identically one attention
head, and draws the consequence: memories "are a K/V pair the model can attend
to directly, at a cost of zero context tokens". V.4 prices that at 100 rules ->
1 slot, a 3600x saving. Both were argued on BGE sentence vectors; the status
table records the end-to-end demo as blocked, HuggingFace being unreachable.

It is reachable now, so this script runs the demo on SmolLM2-135M-Instruct
against the 26-rule corpus in `rulebook.py`, over arms that separate the two
claims the consequence actually bundles:

  none          no memory                                  0 tokens, 0 slots
  text          rules in the prompt                        N tokens, N slots
  kv_full       same rules prefilled, tokens withheld      0 tokens, N slots
  kv_pooled     each rule mean-pooled to one slot          0 tokens, R slots
  kv_super_all  every rule summed into ONE slot            0 tokens, 1 slot
  kv_super_k    k rules (gold included) summed into one    0 tokens, 1 slot
  kv_random     one slot of noise                          floor control

`text` vs `kv_full` prices the context-token claim; `kv_pooled` and the
`kv_super_*` arms price the compression claim, which is the one V.4 sells.

Two metrics, neither of which asks a 135M model to reason:

  recitation NLL   mean token NLL of the gold rule's own text. Did the content
                   arrive at all? Lower is better; the `none` arm is the prior.
  forced choice    does gold out-score its traps (superseded versions, wrong
                   scope siblings)? Did *discriminative* content arrive?

Run: PYTHONPATH=. .venv/bin/python experiments/kv_injection.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from torch import Tensor

from experiments import rulebook

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SEED = 0
PROMPT = "Question: {q}\nThe applicable company policy states: "
DEFAULT_K = 8
"""V.2 reports k=8 as the lossless point for whitened BGE vectors."""


@dataclass(frozen=True)
class KV:
    """One memory as the model's own key/value tensors, per layer.

    keys[i] and values[i] are (1, kv_heads, slots, head_dim) for layer i.
    """

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]

    @property
    def slots(self) -> int:
        return self.keys[0].shape[2]

    def pooled(self, slots: int = 1) -> "KV":
        """Mean-pool the sequence axis down to `slots` contiguous segments."""

        def _pool(x: Tensor) -> Tensor:
            groups = torch.tensor_split(x, min(slots, x.shape[2]), dim=2)
            return torch.cat([g.mean(dim=2, keepdim=True) for g in groups], dim=2)

        return KV(tuple(_pool(k) for k in self.keys), tuple(_pool(v) for v in self.values))


def noise_like(reference: KV) -> KV:
    """Gaussian K/V matched to a real memory's shape and per-layer scale.

    The floor for every compressed arm: same slot count, same norms, no content.
    """
    generator = torch.Generator().manual_seed(SEED)
    return KV(
        tuple(torch.randn(m.shape, generator=generator) * m.std() for m in reference.keys),
        tuple(torch.randn(m.shape, generator=generator) * m.std() for m in reference.values),
    )


def concat(parts: list[KV]) -> KV:
    return KV(
        tuple(torch.cat([p.keys[i] for p in parts], dim=2) for i in range(len(parts[0].keys))),
        tuple(torch.cat([p.values[i] for p in parts], dim=2) for i in range(len(parts[0].values))),
    )


def superpose(parts: list[KV]) -> KV:
    """Normalised sum into a single slot -- `ModernHopfieldNetwork.superpose`.

    The sum is rescaled to the mean norm of its members. A raw sum of k vectors
    grows like sqrt(k) and would move the state outside the range of key norms
    the frozen model was trained against, which is a scale artifact and not the
    capacity question V.2 asks.
    """

    def _sum(tensors: list[Tensor]) -> Tensor:
        stack = torch.stack(tensors, dim=0)  # (k, 1, heads, 1, dim)
        total = stack.sum(dim=0)
        scale = stack.norm(dim=-1, keepdim=True).mean(dim=0)
        return total / total.norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale

    pooled = [p.pooled() for p in parts]
    n_layers = len(pooled[0].keys)
    return KV(
        tuple(_sum([p.keys[i] for p in pooled]) for i in range(n_layers)),
        tuple(_sum([p.values[i] for p in pooled]) for i in range(n_layers)),
    )


class Injector:
    """Prefills text into the model's own K/V space and answers against it."""

    def __init__(self, model_name: str = MODEL) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
        self.n_layers = self.model.config.num_hidden_layers

    def tokens(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids

    @torch.no_grad()
    def prefill(self, text: str) -> KV:
        cache = self.model(self.tokens(text), use_cache=True).past_key_values
        length = cache.get_seq_length()
        return KV(
            tuple(cache.layers[i].keys[:, :, :length].clone() for i in range(self.n_layers)),
            tuple(cache.layers[i].values[:, :, :length].clone() for i in range(self.n_layers)),
        )

    def _cache(self, memory: KV | None):
        """A DynamicCache holding `memory`, built by overwriting a real prefill."""
        if memory is None:
            return None
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.model.config)
        for i in range(self.n_layers):
            cache.update(memory.keys[i], memory.values[i], i)
        return cache

    @torch.no_grad()
    def nll(self, prompt: str, target: str, memory: KV | None = None) -> float:
        """Mean token NLL of `target` continuing `prompt`, attending to `memory`."""
        prompt_ids = self.tokens(prompt)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([prompt_ids, target_ids], dim=1)
        offset = memory.slots if memory is not None else 0
        positions = torch.arange(offset, offset + ids.shape[1]).unsqueeze(0)
        logits = self.model(
            ids,
            past_key_values=self._cache(memory),
            position_ids=positions,
            use_cache=False,
        ).logits
        n = target_ids.shape[1]
        logprobs = torch.log_softmax(logits[0, -n - 1 : -1].float(), dim=-1)
        return float(-logprobs.gather(1, target_ids[0].unsqueeze(1)).mean())


def anisotropy_by_layer(memories: list[KV]) -> list[float]:
    """Mean pairwise cosine between distinct memories' pooled keys, per layer.

    The V.2 diagnosis, transplanted: BGE vectors sit at +0.649 and superposition
    collapses; random unit vectors sit at 0.000 and it does not.
    """
    pooled = [m.pooled() for m in memories]
    out = []
    for i in range(len(pooled[0].keys)):
        x = torch.stack([p.keys[i].flatten() for p in pooled])
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = x @ x.T
        off = ~torch.eye(len(x), dtype=torch.bool)
        out.append(float(cos[off].mean()))
    return out


def choose_k(situation, all_ids: list[str], k: int) -> list[str]:
    """k rule ids containing the situation's gold, so a superposed mixture is a
    fair test of capacity (gold is always present, never crowded out by luck).

    Pulled out of `build_arms` so ticket 13's learned-read-in harness can drive
    the same k-selection when it trains and evaluates against `kv_super_k`'s
    exact mixtures, instead of re-deriving this rule and risking drift.
    """
    chosen = list(situation.gold)
    for rid in all_ids:
        if len(chosen) >= k:
            break
        if rid not in chosen:
            chosen.append(rid)
    return chosen


def build_arms(
    inj: Injector,
    memories: dict[str, KV],
    k: int,
    pools: list[int],
    rules: list | None = None,
) -> dict:
    """name -> (memory for a situation, prompt-token cost).

    `text` and `kv_full` are deliberately the *same token sequence*, split
    differently: one supplies the rules as prompt, the other prefills them and
    withholds the tokens. Any gap between them is the price of the mechanism.

    `rules` defaults to the full `rulebook.RULES` (unchanged behaviour for
    every existing caller). It exists so a smoke-scale caller -- ticket 13's
    learned-read-in harness -- can pass a small, self-consistent subset and
    get a fast, comparable curve instead of Part VII's full ~4-minute run.
    """
    rules = rules if rules is not None else rulebook.RULES
    all_ids = [r.rule_id for r in rules]
    text = rulebook_text(rules)
    full_kv = inj.prefill(text)
    super_all = superpose([memories[i] for i in all_ids])

    def super_k(situation) -> KV:
        return superpose([memories[i] for i in choose_k(situation, all_ids, k)])

    arms: dict = {
        "none": (lambda s: None, 0),
        "text": (lambda s: None, inj.tokens(text).shape[1]),
        "kv_full": (lambda s: full_kv, 0),
    }
    for n in pools:
        pooled = concat([memories[i].pooled(n) for i in all_ids])
        arms[f"kv_pool{n}"] = (lambda s, kv=pooled: kv, 0)
        arms[f"  noise/{n}"] = (lambda s, kv=noise_like(pooled): kv, 0)
    arms["kv_super_all"] = (lambda s: super_all, 0)
    arms[f"kv_super_k{k}"] = (super_k, 0)
    arms["  noise/1slot"] = (lambda s, kv=noise_like(super_all): kv, 0)
    return arms


def rulebook_text(rules: list | None = None) -> str:
    """`rules` defaults to the full `rulebook.RULES` -- see `build_arms`."""
    rules = rules if rules is not None else rulebook.RULES
    return "\n".join(r.text for r in rules) + "\n"


def score_arm(
    inj: Injector,
    by_id: dict,
    situations: list,
    name: str,
    memory_of,
    prompt_cost: int,
    full_text: str,
) -> dict:
    """Recitation NLL + forced choice for one arm -- Part VII's metric, exactly
    the loop `main` below runs, pulled out so ticket 13's learned-read-in
    harness can score its own arms with the identical code path instead of a
    reimplementation that could silently drift from the published numbers.
    """
    scored = [s for s in situations if s.traps]
    nlls, correct, margins = [], 0, []
    for situation in situations:
        memory = memory_of(situation)
        prompt = PROMPT.format(q=situation.question)
        if name == "text":
            prompt = full_text + prompt
        gold_nll = inj.nll(prompt, by_id[situation.gold[0]].text, memory)
        nlls.append(gold_nll)
        traps = [inj.nll(prompt, by_id[t].text, memory) for t in situation.traps]
        if traps:
            correct += int(gold_nll < min(traps))
            margins.append(min(traps) - gold_nll)
    probe = memory_of(situations[0])
    return {
        "recite_nll": sum(nlls) / len(nlls),
        "forced_choice": correct / len(scored) if scored else float("nan"),
        "margin": sum(margins) / len(margins) if margins else float("nan"),
        "prompt_tokens": prompt_cost,
        "slots": probe.slots if probe is not None else prompt_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--pools", default="1,2,4,8")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    inj = Injector(args.model)
    by_id = rulebook.by_id()
    pools = [int(p) for p in args.pools.split(",")]
    full_text = rulebook_text()

    print(f"model {args.model}  layers {inj.n_layers}  rules {len(rulebook.RULES)}")
    memories = {r.rule_id: inj.prefill(r.text) for r in rulebook.RULES}
    mean_slots = sum(m.slots for m in memories.values()) / len(memories)
    print(
        f"mean rule = {mean_slots:.1f} slots; "
        f"whole rulebook = {inj.tokens(full_text).shape[1]} tokens"
    )

    aniso = anisotropy_by_layer(list(memories.values()))
    print("\nkey anisotropy (mean off-diagonal cosine between pooled rule keys)")
    print(
        f"  layer 0 {aniso[0]:+.3f}   mid {aniso[len(aniso) // 2]:+.3f}   "
        f"last {aniso[-1]:+.3f}   mean {sum(aniso) / len(aniso):+.3f}"
    )

    scored = [s for s in rulebook.SITUATIONS if s.traps]
    print(f"\n{len(rulebook.SITUATIONS)} situations, {len(scored)} with traps\n")
    print(f"  {'arm':<14} {'nll':>7} {'fc':>6} {'margin':>8} {'tokens':>7} {'slots':>6}")

    arms = build_arms(inj, memories, args.k, pools)
    results: dict[str, dict] = {}
    for name, (memory_of, prompt_cost) in arms.items():
        # Scored through `score_arm`, which is this loop, extracted. Ticket 13's
        # learned-read-in harness scores its arms with the same function, and
        # that is only worth anything if `main` -- the thing that produced the
        # published Part VII numbers -- goes through it too. Two copies of a
        # metric are two metrics.
        results[name.strip()] = score_arm(
            inj, by_id, rulebook.SITUATIONS, name, memory_of, prompt_cost, full_text
        )
        r = results[name.strip()]
        print(
            f"  {name:<14} {r['recite_nll']:7.3f} {r['forced_choice']:6.3f} "
            f"{r['margin']:+8.3f} {r['prompt_tokens']:7d} {r['slots']:6d}"
        )

    gap = results["text"]["recite_nll"] - results["kv_full"]["recite_nll"]
    print(
        f"\ntext vs kv_full on the same {results['text']['prompt_tokens']} tokens: "
        f"dNLL {gap:+.3e}  (prompt cost {results['text']['prompt_tokens']} -> 0)"
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"anisotropy": aniso, "arms": results}, handle, indent=2)


if __name__ == "__main__":
    main()
