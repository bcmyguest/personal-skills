"""Energy-based memory tracking, via the MHN <-> diffusion equivalence.

The equivalence
---------------
Give the Hopfield energy a Gibbs measure at inverse temperature beta:

    p(xi) ∝ exp(-beta * E(xi))

Substituting E from `hippocampus` and using beta*x_i.xi - beta/2*||xi||^2 =
-beta/2 * (||xi - x_i||^2 - ||x_i||^2):

    exp(-beta E(xi)) ∝ sum_i w_i * exp(-beta/2 * ||xi - x_i||^2)
                                  * exp(beta/2 * (||x_i||^2 - M^2))

When patterns are unit-normalised the trailing factor is a constant, and the
Gibbs measure of the MHN is *exactly* a Gaussian mixture centred on the stored
memories:

    p(xi) = sum_i w_i * N(xi; x_i, sigma^2 I),      sigma^2 = 1 / beta

That is precisely the noised marginal p_sigma of a diffusion model whose data
distribution is the empirical distribution of the memory set. Hence:

  * beta is not a free knob -- it *is* the inverse noise level. Low beta =
    early (noisy) diffusion time = broad schema-level basins; high beta = late
    diffusion time = sharp episodic attractors.

  * The score is available in closed form:

        score(xi) = grad log p_sigma(xi) = -beta * grad E(xi)
                  = beta * (X^T softmax(beta X xi + log w) - xi)
                  = (m(xi) - xi) / sigma^2

  * By Tweedie/Miyasawa, the optimal denoiser is the posterior mean

        E[x | xi] = xi + sigma^2 * score(xi) = X^T softmax(beta X xi + log w)

    which is *identically* the Modern Hopfield update. One MHN retrieval step
    is one exact denoising step of a diffusion model over the memory set.
    (Ambrogioni 2023, arXiv:2309.17290, "In search of dispersed memories:
    Generative diffusion models are associative memory networks"; Hoover et al.
    2023, arXiv:2309.16750, "Memory in Plain Sight" -- diffusion models are
    associative memories whose energy is the negative log-density of the
    noised data.)

Why this is useful here, rather than decorative
-----------------------------------------------
1. `log_density` gives a *calibrated, normalised* score for "does the
   organisation actually remember this?" -- the raw Hopfield energy is only
   defined up to a constant, but under the mixture reading the normaliser is
   known in closed form, so the number is comparable across memory sets of
   different sizes and across time.

2. `basin_depth` measures energy at a query relative to the energy at the
   nearest stored memory. Near zero = the query sits inside a real attractor
   (trustworthy recall). Large = the query is out-of-distribution and whatever
   we return is a confabulated mixture, not a memory. This is the retrieval
   confidence signal.

3. `langevin_replay` samples the memory distribution using the exact score,
   giving a principled generator for hippocampal replay during consolidation
   -- samples come from the *smoothed* memory distribution, so the cortex is
   trained on the neighbourhood of each episode rather than memorising it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .hippocampus import ModernHopfieldNetwork


def beta_to_sigma(beta: float) -> float:
    """Inverse temperature -> equivalent diffusion noise std."""
    return 1.0 / math.sqrt(beta)


def sigma_to_beta(sigma: float) -> float:
    """Diffusion noise std -> equivalent inverse temperature."""
    return 1.0 / (sigma * sigma)


def score(mhn: ModernHopfieldNetwork, xi: Tensor, beta: float | None = None) -> Tensor:
    """grad_xi log p_sigma(xi), in closed form. No autograd required."""
    b = mhn._beta(beta)
    return b * (mhn.step(xi, beta) - xi)


def denoise(mhn: ModernHopfieldNetwork, xi: Tensor, beta: float | None = None) -> Tensor:
    """Tweedie posterior mean E[x | xi] -- identical to one MHN update."""
    return mhn.step(xi, beta)


def log_density(
    mhn: ModernHopfieldNetwork, xi: Tensor, beta: float | None = None
) -> Tensor:
    """Properly normalised log p_sigma(xi) of the Gaussian mixture.

    log p = logsumexp_i( log w_i - ||xi - x_i||^2 / (2 sigma^2) )
            - (d/2) log(2 pi sigma^2)

    Valid as written when patterns are unit-normalised; otherwise it differs
    from the true density by the per-pattern norm factors documented above.
    """
    if mhn.is_empty:
        raise RuntimeError("log_density undefined for an empty memory")
    b = mhn._beta(beta)
    d = mhn.dim
    prior = mhn.log_prior - torch.logsumexp(mhn.log_prior, dim=0)
    sq = torch.cdist(xi.reshape(-1, d), mhn.patterns).pow(2)
    lse = torch.logsumexp(prior - 0.5 * b * sq, dim=-1)
    out = lse - 0.5 * d * math.log(2 * math.pi / b)
    return out.reshape(xi.shape[:-1]) if xi.dim() > 1 else out.squeeze(0)


@dataclass
class BasinReport:
    """Diagnostic for one query against the memory landscape."""

    energy: float
    nearest_energy: float
    depth: float
    """energy - nearest_energy, in units of the energy function. >0 always;
    small means the cue is inside a genuine basin."""
    log_density: float
    top_similarity: float
    nearest_index: int

    @property
    def is_confabulation(self) -> bool:
        """Heuristic: the cue is far outside every stored basin, so a returned
        'memory' is an interpolation the organisation never actually recorded."""
        return self.depth > 0.5 or self.top_similarity < 0.3


def basin_depth(
    mhn: ModernHopfieldNetwork, xi: Tensor, beta: float | None = None
) -> BasinReport:
    """How well does the memory landscape actually explain this query?"""
    sims = xi @ mhn.patterns.T
    idx = int(sims.argmax())
    e_query = float(mhn.energy(xi, beta))
    e_near = float(mhn.energy(mhn.patterns[idx], beta))
    return BasinReport(
        energy=e_query,
        nearest_energy=e_near,
        depth=e_query - e_near,
        log_density=float(log_density(mhn, xi, beta)),
        top_similarity=float(sims[idx]),
        nearest_index=idx,
    )


@torch.no_grad()
def langevin_replay(
    mhn: ModernHopfieldNetwork,
    n_samples: int,
    *,
    steps: int = 8,
    step_size: float = 0.05,
    beta: float | None = None,
    init: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the memory distribution by annealed Langevin dynamics.

        xi <- xi + (eta/2) * score(xi) + sqrt(eta) * z,   z ~ N(0, I)

    Seeded from salience-weighted stored patterns, so replay frequency tracks
    memory strength -- decayed episodes are replayed less and therefore
    consolidate into the cortex more weakly. That is the intended behaviour:
    the forgetting curve should shape what becomes permanent knowledge.
    """
    if mhn.is_empty:
        raise RuntimeError("cannot replay from an empty hippocampus")

    if init is None:
        probs = torch.softmax(mhn.log_prior, dim=0)
        idx = torch.multinomial(probs, n_samples, replacement=True, generator=generator)
        xi = mhn.patterns[idx].clone()
        sigma = beta_to_sigma(mhn._beta(beta))
        xi = xi + sigma * torch.randn(xi.shape, generator=generator)
    else:
        xi = init.clone()

    for _ in range(steps):
        noise = torch.randn(xi.shape, generator=generator)
        xi = xi + 0.5 * step_size * score(mhn, xi, beta) + math.sqrt(step_size) * noise
    return xi
