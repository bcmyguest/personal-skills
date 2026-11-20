"""Ticket 13 -- a LEARNED read-in for KV superposition, and the correction
its result motivates.

VII.2 (`experiments/kv_injection.py`) found that compressing rules into a
single K/V slot recovers none of the zero-context-token benefit. Three things
make that negative weaker than it reads, per ticket 13:

  1. the frozen decoder's query projections were never trained to read a
     memory slot -- they were trained to read real prefill K/V, and VII.3
     measured that key space as a +0.969 mean-cosine cone, far narrower than
     anything `superpose`'s untrained normalise-and-rescale-sum was aimed at;
  2. the compression itself was a fixed rule (mean-pool, or normalise-sum),
     never a *learned* one;
  3. V.2's fix for superposition collapse -- whitening -- cannot be applied
     inside a frozen model without changing every `q.k` score the frozen
     query projections were trained to produce.

This module addresses (1) and (2) directly: `LearnedReadIn` is a small,
gradient-trained per-layer affine map from a held superposition (still built
by `kv_injection.superpose` -- the *hold* step is unchanged) into whatever
K/V geometry minimises the frozen decoder's recitation loss. It cannot touch
(3): the decoder's queries stay frozen by construction, so this is a test of
how far a learned *read-in* alone can close VII.2's gap, not a retraction of
VII.3's whitening argument.

Two claims, kept structurally separate (ticket 13's third checkbox)
------------------------------------------------------------------
`main()` below prints two headed, non-adjacent blocks:

  * COMPRESSION -- the re-run curve: same recitation-NLL / forced-choice /
    margin metric as `kv_injection.score_arm` (imported, not reimplemented,
    so there is no way for this harness's numbers to silently diverge from
    Part VII's), the same matched-noise controls (`kv_injection.noise_like`,
    same call), now including a `kv_learned_all` arm and its own matched
    noise floor.
  * ZERO-CONTEXT-TOKENS -- VII.1's `text` vs `kv_full` gap, re-confirmed on
    whatever rule/situation subset this run used. This claim does not depend
    on compression or on the learned projection at all; printing it in its
    own block stops a reader from attributing the (likely still-limited)
    compression numbers to a claim that was already exact before this ticket
    existed, and stops the reverse mistake of reading the token claim as
    evidence for the compression claim.

Smoke vs full runs
-------------------
Training backpropagates through the whole frozen 135M-parameter decoder (only
`LearnedReadIn`'s own ~246k parameters get an optimizer step, but every layer
between the injected memory and the loss must still build its autograd graph
each step). That is CPU-feasible at smoke scale and is NOT something this
module runs at full scale on its own: `--situations` and `--steps` default
small enough to finish in about a minute, and are plain CLI arguments so a
full-curve run (all 26 rules, all 16 situations, more steps) is the
maintainer's call, not something this script does automatically.

Run (smoke): PYTHONPATH=. uv run python -u experiments/kv_learned_readin.py

By default `--holdout 0` trains and evaluates `kv_learned_all` on the SAME
situations. Measured at smoke scale that drives recitation NLL from
`kv_super_all`'s ~3.3 down near 0.4 -- which is memorization of 3 situations
by a 246k-parameter map, not evidence the compression mechanism generalizes.
Pass `--holdout N` (with `--situations` large enough to leave real training
data) to get the number that actually answers ticket 13's question: NLL on
situations the projection never trained against, next to the untrained
`kv_super_all` baseline on the same split. See the GENERALIZATION block in
`main`'s output.

CONCEPTUAL CORRECTION (ticket 13's fourth checkbox)
----------------------------------------------------
RESULTS.md V.4 prices "100 rules -> 1 slot" at 3600x and frames it as a
capacity result. **That is not a Hopfield-network capacity claim, and it
should not have been priced as though it were one.**

Hopfield/modern-Hopfield capacity (Ramsauer et al. 2021, "Hopfield Networks
is All You Need", arXiv:2008.02217 -- the paper `cls_memory/hippocampus.py`
already implements) concerns storing N patterns as N separate rows and
retrieving ONE of them by settling a query onto its attractor. Ramsauer's
separation bound (`ModernHopfieldNetwork.separation`) governs exactly that
operation: `step`/`retrieve` in this repo. It says nothing about summing N
patterns into a single vector and reading several of them back out of the
sum -- `superpose`/`decode`, a different pair of operations, as
`hippocampus.py`'s own docstrings now state (they did not, before RESULTS.md
Part V; every earlier experiment in this project used the settled state for
both jobs and got V.3's collapse for it).

The claim that actually governs "fold many vectors into one and read several
back out" belongs to vector-symbolic architectures (VSA) / holographic
reduced representations (HRR):

  * Tony A. Plate, "Holographic Reduced Representations", IEEE Transactions
    on Neural Networks, 1995. Circular-convolution binding and
    vector-sum superposition, with the crosstalk noise that bounds how many
    bound pairs one vector can hold. High confidence in this citation
    (title, author, venue, year); I have not verified the volume/issue/page
    numbers and do not quote them.
  * Pentti Kanerva, "Hyperdimensional Computing: An Introduction to
    Computing in Distributed Representation with High-Dimensional Random
    Vectors", Cognitive Computation, 2009. The "bundling" (sum) operator and
    the near-orthogonality high-dimensional random codes need for bundled
    items to stay separable -- the same requirement V.2's whitening
    rediscovered empirically on BGE vectors, and VII.3 found violated, worse,
    in a frozen decoder's own key space. High confidence in this citation;
    again, no page numbers quoted.
  * E. Paxon Frady, Denis Kleyko, and Friedrich T. Sommer have published
    capacity analyses specifically for superposition/bundling codes in
    VSA/hyperdimensional systems (individually and with Bruno Olshausen).
    I am NOT confident enough in a specific title, year, or venue to cite one
    of their papers precisely here, and the ticket that spawned this module
    is explicit that an uncertain identifier must not be invented -- so this
    is a pointer to the right authors to look up, not a citation.

Read against this literature rather than Ramsauer's, the capacity for
reliably un-binding N items from one superposed vector sits far below what
V.4's "100 rules -> 1 slot, 3600x" implied, and requires the summed codes to
be close to orthogonal. V.4's cost table conflated a token-cost saving (real,
exact, VII.1) with a memory/capacity claim (VSA-governed, and -- per VII.2 --
measured here at effectively 0% recovered at 610x compression in a frozen
decoder's own key space). It should be republished as two rows, not one, and
grounded in the literature that actually bounds the second row.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

import torch
from torch import Tensor

# The decoder LM must already be on disk -- see the module docstring's "Smoke
# vs full runs" note and the README/HANDOFF's HuggingFace-reachability
# history. Setting this before `transformers` touches the network turns a
# silent "sure, I'll fetch it" into a loud, immediate failure if the cache
# used by `experiments/kv_injection.py` does not have the model this run
# asks for -- which is what ticket 13 asked this harness to do instead of
# assuming availability.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from experiments import rulebook  # noqa: E402
from experiments.kv_injection import (  # noqa: E402
    MODEL,
    PROMPT,
    SEED,
    Injector,
    KV,
    build_arms,
    noise_like,
    rulebook_text,
    score_arm,
    superpose,
)
from experiments.threads import add_threads_arg, pin_threads  # noqa: E402


def smoke_subset(n_situations: int) -> tuple[list[rulebook.Rule], list[rulebook.Situation]]:
    """A minimal self-consistent slice of the rulebook: the first
    `n_situations` situations, plus every rule they cite (gold and traps).

    Slicing situations without slicing rules to match would KeyError inside
    `superpose`/`build_arms` the moment a situation's gold or trap rule fell
    outside the truncated rule set. This keeps the two truncations
    consistent so the smoke run is small but not broken.
    """
    situations = rulebook.SITUATIONS[:n_situations]
    if not situations:
        raise ValueError("n_situations must be >= 1")
    needed = {rid for s in situations for rid in (*s.gold, *s.traps)}
    rules = [r for r in rulebook.RULES if r.rule_id in needed]
    return rules, situations


class LearnedReadIn(torch.nn.Module):
    """Learned per-layer, per-head-shared affine maps from a held
    superposition into the K/V geometry the frozen decoder's queries expect.

    Part VII's compressed arms combined memories with a FIXED rule -- mean-
    pool to n slots, or `kv_injection.superpose`'s normalise-and-rescale-sum
    to one. This class replaces that fixed rule with a trained one, for the
    single-slot case specifically (mirroring `kv_super_all`): one
    `torch.nn.Linear(head_dim, head_dim)` per layer for keys, one for values,
    SHARED across attention heads (applied to each (kv_heads, head_dim) row
    identically). Sharing across heads keeps the parameter count small enough
    to train on CPU in the couple of minutes this harness budgets --
    `n_layers * 2 * (head_dim**2 + head_dim)` parameters, 249,600 for
    SmolLM2-135M's 30 layers / head_dim 64 -- and matches the structure GQA
    attention already imposes: every query head within a KV group reads the
    same K/V row, so per-head maps would have no signal to differ on for a
    single superposed slot.

    Initialised near-identity (`eye_` weight, zero bias) so training starts
    from Part VII's untrained baseline rather than from noise: at step 0,
    `LearnedReadIn(base) == base` up to floating point, i.e. exactly
    `kv_super_all`.
    """

    def __init__(self, n_layers: int, head_dim: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.key_proj = torch.nn.ModuleList(
            torch.nn.Linear(head_dim, head_dim) for _ in range(n_layers)
        )
        self.value_proj = torch.nn.ModuleList(
            torch.nn.Linear(head_dim, head_dim) for _ in range(n_layers)
        )
        for lin in (*self.key_proj, *self.value_proj):
            torch.nn.init.eye_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, superposed: KV) -> KV:
        """Map a held superposition through the learned per-layer maps.

        Shape in == shape out (each layer's key/value keeps its
        `(1, kv_heads, slots, head_dim)` shape); only the content moves.
        """
        if len(superposed.keys) != self.n_layers:
            raise ValueError(
                f"LearnedReadIn built for {self.n_layers} layers, got a KV "
                f"with {len(superposed.keys)}"
            )
        keys = tuple(self.key_proj[i](superposed.keys[i]) for i in range(self.n_layers))
        values = tuple(self.value_proj[i](superposed.values[i]) for i in range(self.n_layers))
        return KV(keys, values)


def detach_kv(kv: KV) -> KV:
    """A KV with every tensor detached -- for handing a trained module's
    output to `kv_injection.score_arm`, which must not build a graph."""
    return KV(tuple(k.detach() for k in kv.keys), tuple(v.detach() for v in kv.values))


def train_readin(
    readin: LearnedReadIn,
    loss_fn: Callable[[LearnedReadIn], Tensor],
    steps: int,
    lr: float = 0.05,
) -> list[float]:
    """Adam over `readin`'s own parameters only, for `steps` steps.

    Deliberately decoupled from any specific loss: `loss_fn(readin) ->
    Tensor` is the whole contract, so tests can drive this with a cheap
    synthetic target (no decoder LM, no download) and `main` below can drive
    it with the real recitation loss against the frozen SmolLM2 forward pass.
    Whatever `loss_fn` closes over (a frozen model, a fixed target) never
    gets an optimizer step here -- only `readin.parameters()` does.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    optimizer = torch.optim.Adam(readin.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(readin)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def differentiable_nll(inj: Injector, prompt: str, target: str, memory: KV) -> Tensor:
    """`Injector.nll`'s arithmetic, minus its `@torch.no_grad()`.

    Training needs gradients to flow from this loss back through `memory`
    into `LearnedReadIn`'s parameters; `Injector.nll` cannot be reused as-is
    because its decorator strips exactly the gradient this needs. Everything
    below this line is otherwise identical to `Injector.nll`, on purpose --
    see that method for why each step is there.
    """
    prompt_ids = inj.tokens(prompt)
    target_ids = inj.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([prompt_ids, target_ids], dim=1)
    offset = memory.slots
    positions = torch.arange(offset, offset + ids.shape[1]).unsqueeze(0)
    # `_cache` is Injector's private cache-builder; reused rather than
    # duplicated so a memory built here is packaged for the model identically
    # to every other arm `kv_injection.py` scores.
    logits = inj.model(
        ids,
        past_key_values=inj._cache(memory),
        position_ids=positions,
        use_cache=False,
    ).logits
    n = target_ids.shape[1]
    logprobs = torch.log_softmax(logits[0, -n - 1 : -1].float(), dim=-1)
    return -logprobs.gather(1, target_ids[0].unsqueeze(1)).mean()


def recitation_loss(
    inj: Injector,
    memories: dict[str, KV],
    by_id: dict,
    situations: list,
    rules: list,
) -> Callable[[LearnedReadIn], Tensor]:
    """Mean recitation NLL of each training situation's gold rule, read
    through `readin` applied to the FIXED all-rules superposition -- the
    direct trained counterpart of `kv_super_all`, not `kv_super_k` (training
    a per-situation-varying input would need a fresh superposition per step,
    which is a bigger CPU bill for a smoke harness with no payoff for this
    ticket's question: does a learned map help the single fixed slot at all).
    """
    all_ids = [r.rule_id for r in rules]
    base = superpose([memories[i] for i in all_ids])

    def loss_fn(readin: LearnedReadIn) -> Tensor:
        memory = readin(base)
        total = torch.zeros(())
        for situation in situations:
            prompt = PROMPT.format(q=situation.question)
            target = by_id[situation.gold[0]].text
            total = total + differentiable_nll(inj, prompt, target, memory)
        return total / len(situations)

    return loss_fn


def print_table(title: str, results: dict[str, dict], ref_slots: int) -> None:
    print(f"\n=== {title} ===")
    print(
        f"  {'arm':<16} {'nll':>7} {'fc':>6} {'margin':>8} "
        f"{'tokens':>7} {'slots':>6} {'compression':>12}"
    )
    for name, r in results.items():
        ratio = f"{ref_slots / r['slots']:.1f}x" if r["slots"] else "-"
        print(
            f"  {name:<16} {r['recite_nll']:7.3f} {r['forced_choice']:6.3f} "
            f"{r['margin']:+8.3f} {r['prompt_tokens']:7d} {r['slots']:6d} {ratio:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--situations",
        type=int,
        default=3,
        help="how many rulebook situations (from the start of the list) to use for "
        "both training and eval -- smoke default is small on purpose; scale this up "
        "yourself for a full run (max 16).",
    )
    parser.add_argument("--k", type=int, default=4, help="rules per kv_super_k mixture")
    parser.add_argument(
        "--pools", default="1", help="comma-separated pool sizes for kv_pool arms"
    )
    parser.add_argument("--steps", type=int, default=15, help="LearnedReadIn training steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="hold out the LAST N of the --situations slice from training and report "
        "kv_learned_all's recitation NLL on train vs held-out separately. Default 0 "
        "trains and evaluates on the SAME situations -- a deliberate overfitting "
        "smoke check that the mechanism runs end to end, NOT a generalization claim. "
        "Set this >0 (and raise --situations to leave enough training data) for a run "
        "that actually tests whether the learned map generalizes.",
    )
    parser.add_argument("--json", default="")
    add_threads_arg(parser)
    args = parser.parse_args()
    threads = pin_threads(args.threads)
    print(f"torch threads: {threads}")

    torch.manual_seed(args.seed)

    try:
        inj = Injector(args.model)
    except OSError as exc:
        raise SystemExit(
            f"Could not load {args.model} with HF_HUB_OFFLINE=1 -- the weights are not "
            "in the local Hugging Face cache. This harness refuses to download them "
            "silently (ticket 13's instruction); fetch the model yourself first if you "
            "want to run this for real, e.g.:\n"
            f"  HF_HUB_OFFLINE=0 uv run python -u -c "
            f"\"from transformers import AutoModelForCausalLM; "
            f"AutoModelForCausalLM.from_pretrained('{args.model}')\"\n"
            f"Original error: {exc}"
        ) from exc

    by_id = rulebook.by_id()
    pools = [int(p) for p in args.pools.split(",")]
    rules, situations = smoke_subset(args.situations)
    all_ids = [r.rule_id for r in rules]
    full_text = rulebook_text(rules)

    print(
        f"model {args.model}  layers {inj.n_layers}  "
        f"rules {len(rules)}/{len(rulebook.RULES)}  "
        f"situations {len(situations)}/{len(rulebook.SITUATIONS)} (smoke subset)"
    )
    memories = {r.rule_id: inj.prefill(r.text) for r in rules}

    if args.holdout:
        if args.holdout >= len(situations):
            raise SystemExit(
                f"--holdout {args.holdout} must be smaller than the "
                f"--situations slice ({len(situations)})"
            )
        train_situations = situations[: len(situations) - args.holdout]
        held_out_situations = situations[len(situations) - args.holdout :]
    else:
        train_situations = situations
        held_out_situations = []

    cfg = inj.model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    readin = LearnedReadIn(inj.n_layers, head_dim)
    loss_fn = recitation_loss(inj, memories, by_id, train_situations, rules)

    print(f"\ntraining LearnedReadIn: {sum(p.numel() for p in readin.parameters())} params, "
          f"{args.steps} steps, lr={args.lr}")
    history = train_readin(readin, loss_fn, steps=args.steps, lr=args.lr)
    print(f"  loss  step 0: {history[0]:.4f}  ->  step {len(history) - 1}: {history[-1]:.4f}")

    with torch.no_grad():
        base = superpose([memories[i] for i in all_ids])
        learned_memory = detach_kv(readin(base))

    arms = build_arms(inj, memories, args.k, pools, rules=rules)
    arms["kv_learned_all"] = (lambda s, kv=learned_memory: kv, 0)
    arms["  noise/learned"] = (lambda s, kv=noise_like(learned_memory): kv, 0)

    results = {
        name.strip(): score_arm(inj, by_id, situations, name, memory_of, prompt_cost, full_text)
        for name, (memory_of, prompt_cost) in arms.items()
    }

    ref_slots = results["kv_full"]["slots"]
    compression_names = [n for n in results if n not in ("none", "text", "kv_full")]
    print_table(
        "COMPRESSION (Part VII's metric, same matched-noise controls, kv_learned_all added)",
        {n: results[n] for n in ["none", *compression_names]},
        ref_slots,
    )

    if held_out_situations:
        # The generalization check --holdout enables: kv_learned_all's NLL on
        # situations it never trained against, next to kv_super_all's on the
        # same split (untrained, so this is the "did learning help at all,
        # on data it didn't see" baseline). If the trained arm's held-out
        # number is close to its train number and still beats kv_super_all,
        # that is a real compression improvement. If held-out regresses
        # toward (or past) kv_super_all, the training-set number is
        # memorization, not a fix to the mechanism VII.2 measured.
        learned_train = score_arm(
            inj, by_id, train_situations, "kv_learned_all",
            lambda s: learned_memory, 0, full_text,
        )
        learned_held = score_arm(
            inj, by_id, held_out_situations, "kv_learned_all",
            lambda s: learned_memory, 0, full_text,
        )
        base_train = score_arm(
            inj, by_id, train_situations, "kv_super_all", lambda s: base, 0, full_text,
        )
        base_held = score_arm(
            inj, by_id, held_out_situations, "kv_super_all", lambda s: base, 0, full_text,
        )
        print(
            f"\n=== GENERALIZATION ({len(train_situations)} train / "
            f"{len(held_out_situations)} held-out situations) ==="
        )
        print(f"  {'arm':<20} {'train nll':>10} {'held-out nll':>13}")
        print(f"  {'kv_learned_all':<20} {learned_train['recite_nll']:10.3f} "
              f"{learned_held['recite_nll']:13.3f}")
        print(f"  {'kv_super_all (untrained)':<20} {base_train['recite_nll']:10.3f} "
              f"{base_held['recite_nll']:13.3f}")

    # Kept visibly separate on purpose: this claim does not depend on
    # compression or on LearnedReadIn at all (see module docstring).
    gap = results["text"]["recite_nll"] - results["kv_full"]["recite_nll"]
    print("\n=== ZERO-CONTEXT-TOKENS (VII.1, re-confirmed, independent of the above) ===")
    print(
        f"  text {results['text']['prompt_tokens']} tokens -> kv_full 0 tokens, "
        f"same {results['text']['slots']} slots: dNLL {gap:+.3e}"
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "threads": threads,
                    "training_loss": history,
                    "arms": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
