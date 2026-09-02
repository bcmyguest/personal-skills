"""Whitening and superposition, measured on real BGE embeddings.

These tests exist to hold three measured claims from RESULTS.md Part V in place:

  1. BGE embeddings are anisotropic (+0.649) and whitening fixes it (-0.001);
  2. superposition capacity depends entirely on that -- per-item recall at k=8
     is 0.062 as shipped and 0.999 whitened;
  3. settling a mixture destroys it at every beta, which is the mistake every
     experiment in this project made before Part V.

They are deliberately run on real text. HANDOFF.md §5 records that synthetic
data has flattered every mechanism here, and it would make test 1 vacuous:
random Gaussian vectors are isotropic by construction, so a whitener that did
nothing at all would pass. Each test asserts *both ends* of a gap for the same
reason -- a good number on its own does not show the transform did anything.
"""

from __future__ import annotations

import math
import os
import warnings

import pytest
import torch

from cls_memory import (
    HopfieldConfig,
    MemorySystemConfig,
    ModernHopfieldNetwork,
    OrganizationalMemory,
    WhiteningConfig,
    Whitener,
    anisotropy,
)
from cls_memory.embeddings import HashingEmbedder
from cls_memory.whitening import WhitenedEmbedder

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------- fixtures

# These tests are what actually run RESULTS.md Part V's real-embedding claims.
# `pytest.importorskip`/`pytest.skip` would let the suite exit green while
# quietly not exercising those claims -- see issue 01. Default to a hard
# failure so an incomplete environment is *visibly* different from a complete
# one (red suite, nonzero exit code), not just a shorter one you'd have to
# read the skip list to notice. Set this env var to explicitly acknowledge
# running without the real-embedding runtime (e.g. an unrelated quick dev
# loop) and fall back to the old skip behaviour.
_ALLOW_MISSING_EMBEDDINGS_ENV = "CLS_MEMORY_ALLOW_MISSING_EMBEDDINGS"


def _require_real_embeddings(reason: str) -> None:
    """Fail loudly unless the caller has explicitly opted into skipping."""
    if os.environ.get(_ALLOW_MISSING_EMBEDDINGS_ENV):
        pytest.skip(f"{reason} (explicitly allowed via {_ALLOW_MISSING_EMBEDDINGS_ENV}=1)")
    pytest.fail(
        f"{reason}. This test backs a claim in RESULTS.md Part V and must not "
        f"silently skip. Install the real-embedding runtime (see README.md "
        f"'Status'/HANDOFF.md '0. Orientation': `uv sync --extra test-embeddings`) "
        f"or set {_ALLOW_MISSING_EMBEDDINGS_ENV}=1 to explicitly run without it.",
        pytrace=False,
    )


@pytest.fixture(scope="module")
def bge_turns() -> torch.Tensor:
    """788 real LoCoMo turns through BGE-small-v1.5 (~20s, encoded once)."""
    for module_name in ("onnxruntime", "tokenizers"):
        try:
            __import__(module_name)
        except ImportError as exc:
            _require_real_embeddings(f"{module_name} is not installed: {exc}")
    try:
        from experiments import locomo
        from experiments.recall_ablation import BGEEmbedder
    except ImportError as exc:  # pragma: no cover - experiments are optional
        _require_real_embeddings(f"experiments package unavailable: {exc}")
    try:
        conversations = locomo.load()[:2]
    except FileNotFoundError as exc:  # pragma: no cover - data not fetched
        _require_real_embeddings(f"LoCoMo data unavailable: {exc}")
    try:
        embedder = BGEEmbedder()
    except Exception as exc:  # pragma: no cover - model weights not present
        _require_real_embeddings(f"BGE weights unavailable offline: {exc}")
    turns = [t.memory_text for c in conversations for t in c.turns]
    x = embedder.encode(turns)
    assert x.shape[0] > 500, "expected the full LoCoMo turn set"
    return x


@pytest.fixture(scope="module")
def whitened(bge_turns) -> torch.Tensor:
    return Whitener(bge_turns.shape[1]).fit_transform(bge_turns)


def per_item_recall(patterns: torch.Tensor, k: int, *, trials: int, seed: int,
                    settle_beta: float | None = None) -> float:
    """Fraction of k superposed memories that come back in the top k.

    `settle_beta=None` measures the held (unsettled) sum; a value settles the
    state with the Hopfield update first, which is the operation under test in
    `test_settling_destroys_the_superposition`.
    """
    n, d = patterns.shape
    net = ModernHopfieldNetwork(d, HopfieldConfig(beta=settle_beta or 128.0))
    net.write(patterns)
    generator = torch.Generator().manual_seed(seed)
    hits = []
    for _ in range(trials):
        members = torch.randperm(n, generator=generator)[:k]
        state = net.superpose(members)
        if settle_beta is not None:
            for _ in range(8):
                nxt = net.step(state, settle_beta)
                nxt = nxt / nxt.norm().clamp_min(1e-12)
                if float((nxt - state).norm()) < 1e-6:
                    state = nxt
                    break
                state = nxt
        top = set(net.decode(state, k).tolist())
        hits.append(len(top & set(members.tolist())) / k)
    return sum(hits) / len(hits)


# ------------------------------------------------------- 1. the diagnostic


def test_whitening_makes_real_embeddings_isotropic(bge_turns, whitened):
    """RESULTS.md V.2: +0.649 as shipped, -0.001 whitened.

    Both ends are asserted. Checking only the whitened number would pass for a
    transform that did nothing if the corpus happened to be isotropic already,
    which is exactly why this runs on BGE output and not on torch.randn.
    """
    raw = anisotropy(bge_turns)
    assert raw > 0.5, f"BGE is supposed to be anisotropic; measured {raw:+.3f}"
    white = anisotropy(whitened)
    assert abs(white) < 0.05, f"whitened code still anisotropic: {white:+.3f}"


def test_anisotropy_detects_a_cone_it_is_given():
    """The diagnostic itself can fail: a known cone must read high, and an
    isotropic control near zero. Without this, test 1 rests on an unverified
    measuring stick."""
    generator = torch.Generator().manual_seed(0)
    isotropic = torch.randn(512, 64, generator=generator)
    assert abs(anisotropy(isotropic)) < 0.05

    axis = torch.zeros(64)
    axis[0] = 1.0
    cone = isotropic / isotropic.norm(dim=-1, keepdim=True) * 0.2 + axis
    assert anisotropy(cone) > 0.9


# ------------------------------------------------- 2. superposition capacity


def test_superposition_at_k8_needs_whitening(bge_turns, whitened):
    """RESULTS.md V.2: per-item recall at k=8 is 0.062 as shipped, 0.999
    whitened. The gap is the claim, so the gap is what is asserted."""
    white = per_item_recall(whitened, 8, trials=60, seed=0)
    raw = per_item_recall(bge_turns, 8, trials=60, seed=0)
    assert white >= 0.95, f"whitened k=8 recall {white:.3f}, expected ~0.999"
    assert raw < 0.3, f"unwhitened k=8 recall {raw:.3f}, expected ~0.062"
    assert white - raw > 0.6, f"gap {white - raw:.3f} too small to be the effect"


def test_capacity_degrades_gracefully_and_not_before_k8(whitened):
    """Monotone-ish decay, not a cliff: 1.000 at k=2 and k=4, ~0.999 at k=8,
    ~0.974 at k=16. A sum of k near-orthogonal unit vectors keeps each component
    at cosine ~1/sqrt(k), so the shape of the curve is the mechanism showing."""
    scores = {k: per_item_recall(whitened, k, trials=40, seed=k) for k in (2, 4, 8, 16)}
    assert scores[2] >= 0.99 and scores[4] >= 0.99, scores
    assert scores[8] >= 0.95, scores
    assert scores[16] >= 0.90, scores
    assert scores[16] <= scores[8] + 1e-6, f"capacity should not improve with k: {scores}"


def test_superposition_is_not_settled(whitened):
    """`superpose` must return the raw normalised sum, bit-for-bit.

    Asserted against the arithmetic rather than against a recall score: a future
    'improvement' that slipped one Hopfield step into `superpose` would move the
    capacity numbers a little and this by 1e-1."""
    net = ModernHopfieldNetwork(whitened.shape[1], HopfieldConfig(beta=128.0))
    net.write(whitened)
    members = [3, 17, 42, 101]
    expected = net.patterns[members].sum(dim=0)
    expected = expected / expected.norm()
    got = net.superpose(members)
    assert torch.allclose(got, expected, atol=1e-6)
    settled = net.step(got)
    settled = settled / settled.norm()
    assert float((settled - got).norm()) > 0.1, "settling was a no-op; check beta"


# ------------------------------------------------------- 3. the anti-regression


@pytest.mark.parametrize("k", [4, 8])
def test_settling_destroys_the_superposition(whitened, k):
    """RESULTS.md V.3: at k=4 the held sum scores 1.000 and the settled state
    0.253; at k=8, 1.000 against 0.149. This is the operation mix-up every
    experiment before Part V made, so it gets an explicit regression test."""
    held = per_item_recall(whitened, k, trials=40, seed=7)
    settled = per_item_recall(whitened, k, trials=40, seed=7, settle_beta=128.0)
    assert held >= 0.95, f"held sum at k={k} scored {held:.3f}"
    assert settled < 0.5, f"settled state at k={k} scored {settled:.3f}, expected ~0.2"
    assert held - settled > 0.4, f"k={k}: held {held:.3f} vs settled {settled:.3f}"


def test_no_beta_holds_a_mixture(whitened):
    """Low beta collapses to the global centroid, high beta to one attractor.
    Checking a sweep rather than one value, because 'we just had the temperature
    wrong' is the obvious rescue and it is not available."""
    d = whitened.shape[1]
    for beta in (1.0 / math.sqrt(d), 2.0, 8.0, 128.0):
        score = per_item_recall(whitened, 8, trials=20, seed=11, settle_beta=beta)
        assert score < 0.5, f"beta={beta:g} held the mixture at {score:.3f}"


# ------------------------------------------------------------- 4. persistence


def test_whitener_round_trips_through_state_dict(bge_turns):
    """Restoring into a freshly constructed whitener must reproduce the transform
    exactly. `ModernHopfieldNetwork` shipped a state_dict bug that silently
    restored padding as live memories; the same shape-changing-buffer hazard is
    present here, so this asserts values, not shapes."""
    dim = bge_turns.shape[1]
    fitted = Whitener(dim).fit(bge_turns[:600])
    held_out = bge_turns[600:]
    before = fitted.transform(held_out)

    restored = Whitener(dim)
    assert not restored.is_fitted
    missing, unexpected = restored.load_state_dict(fitted.state_dict(), strict=True)
    assert not missing and not unexpected
    assert restored.is_fitted
    after = restored.transform(held_out)
    assert torch.equal(before, after), (
        f"max deviation {float((before - after).abs().max()):.2e}"
    )
    # allclose, not equal: a (1, d) matmul takes a different BLAS path from an
    # (n, d) one and differs in the last float32 ulp (~1e-7 measured). That is
    # batching, not state.
    assert torch.allclose(restored.transform(held_out[0]), before[0], atol=1e-6)


def test_state_dict_rejects_a_mismatched_dim(bge_turns):
    fitted = Whitener(bge_turns.shape[1]).fit(bge_turns[:200])
    with pytest.raises(RuntimeError, match="expected"):
        Whitener(bge_turns.shape[1] // 2).load_state_dict(fitted.state_dict())


# ------------------------------------------------------------------ 5. decode


def test_decode_recovers_the_members_of_a_superposition(whitened):
    """The whole point: eight known memories go into one 384-d vector and all
    eight come back out, against 788 stored candidates."""
    net = ModernHopfieldNetwork(whitened.shape[1], HopfieldConfig(beta=128.0))
    net.write(whitened)
    members = [5, 61, 130, 222, 341, 480, 559, 700]
    state = net.superpose(members)
    assert state.shape == (whitened.shape[1],)
    assert set(net.decode(state, len(members)).tolist()) == set(members)


def test_decode_ranks_by_logits_not_by_settled_output(whitened):
    """`decode` is the store's own logit computation stopped before the softmax,
    so its ranking must match X @ state for unit-norm patterns under a uniform
    prior -- and must NOT match what settling produces."""
    net = ModernHopfieldNetwork(whitened.shape[1], HopfieldConfig(beta=128.0))
    net.write(whitened)
    members = [11, 90, 204, 333]
    state = net.superpose(members)
    by_cosine = torch.topk(net.patterns @ state, 4).indices
    assert torch.equal(net.decode(state, 4), by_cosine)
    from_settled = net.decode(net.retrieve(state).state, 4)
    assert set(from_settled.tolist()) != set(members)


def test_decode_and_superpose_reject_bad_input(whitened):
    net = ModernHopfieldNetwork(whitened.shape[1], HopfieldConfig(beta=128.0))
    net.write(whitened[:32])
    with pytest.raises(ValueError):
        net.decode(net.superpose([1, 2]), 0)
    with pytest.raises(ValueError):
        net.decode(net.superpose([1, 2]), 33)
    with pytest.raises(IndexError):
        net.superpose([1, 999])
    with pytest.raises(ValueError):
        net.superpose([])
    with pytest.raises(ValueError):
        # Antipodal members cancel: there is no mixture left to hold.
        net.superpose(torch.stack([whitened[0], -whitened[0]]))


def test_superpose_accepts_raw_vectors_not_in_the_store(whitened):
    net = ModernHopfieldNetwork(whitened.shape[1], HopfieldConfig(beta=128.0))
    net.write(whitened)
    rows = whitened[[2, 4, 6]] * torch.tensor([[1.0], [7.0], [0.1]])
    # Rows are unit-normalised first, so an arbitrarily long vector cannot take
    # over the mixture: the result must match superposing the stored indices.
    assert torch.allclose(net.superpose(rows), net.superpose([2, 4, 6]), atol=1e-6)


# ------------------------------------------------------- the fitted transform


def test_transform_does_not_depend_on_the_batch_it_is_called_with(bge_turns):
    """The defect this class exists to prevent.

    `experiments/superposition.py` re-fits an SVD on whatever matrix it is
    handed, so a query whitened alone lands in a different space from the
    documents and every cosine between them is meaningless."""
    whitener = Whitener(bge_turns.shape[1]).fit(bge_turns)
    alone = whitener.transform(bge_turns[7])
    in_a_pair = whitener.transform(bge_turns[6:8])[1]
    in_the_corpus = whitener.transform(bge_turns)[7]
    # atol=1e-6 is float32 batching noise; a refit-per-call whitener disagrees
    # by ~1e-1 here, since a 2-row corpus has a rank-1 covariance.
    assert torch.allclose(alone, in_a_pair, atol=1e-6)
    assert torch.allclose(alone, in_the_corpus, atol=1e-6)
    refit_on_the_pair = Whitener(bge_turns.shape[1]).fit(bge_turns[6:8]).transform(
        bge_turns[7]
    )
    assert float((refit_on_the_pair - alone).abs().max()) > 1e-2, (
        "the control failed: refitting per call should NOT agree with the "
        "fitted transform, or this test proves nothing"
    )


def test_whitened_vectors_are_unit_norm(whitened):
    norms = whitened.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_whitener_guards_degenerate_fits():
    dim = 16
    with pytest.raises(RuntimeError, match="not fitted"):
        Whitener(dim).transform(torch.randn(dim))
    with pytest.raises(ValueError, match="sample"):
        Whitener(dim).fit(torch.randn(1, dim))
    with pytest.raises(ValueError, match="dim"):
        Whitener(dim).fit(torch.randn(32, dim + 1))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.randn(32, dim)
        bad[0, 0] = float("nan")
        Whitener(dim).fit(bad)
    with pytest.raises(ValueError, match="variance"):
        Whitener(dim).fit(torch.ones(32, dim))
    with pytest.raises(ValueError, match="positive"):
        Whitener(dim, WhiteningConfig(floor=0.0))


def test_fitting_below_full_rank_warns_and_still_works():
    """Fewer samples than dimensions: the covariance is singular, so the
    transform projects onto the fitted subspace. That is a real narrowing and
    the caller is told about it rather than getting silently amplified noise."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(20, 64, generator=generator)
    x = x / x.norm(dim=-1, keepdim=True)
    whitener = Whitener(64)
    with pytest.warns(UserWarning, match="rank-deficient"):
        whitener.fit(x)
    out = whitener.transform(x)
    assert torch.isfinite(out).all()
    assert whitener.components == 20


# ------------------------------------------------------------ the config knob


def test_whitening_is_off_by_default_and_changes_nothing():
    """RESULTS.md Parts I-IV were all measured with whitening off. The default
    must therefore leave the embedding path untouched, byte for byte."""
    assert WhiteningConfig().enabled is False
    assert MemorySystemConfig().whitening.enabled is False
    system = OrganizationalMemory(embedder=HashingEmbedder(dim=64))
    assert system.whitener is None
    assert not isinstance(system.embedder, WhitenedEmbedder)


def test_enabling_whitening_actually_whitens_the_system():
    """A config knob that nothing reads is worse than no knob. This checks the
    wiring end to end: bootstrap fits the whitener, and the embeddings the
    system stores and queries with are the whitened ones."""
    config = MemorySystemConfig()
    config.whitening = WhiteningConfig(enabled=True)
    config.cortex.epochs = 2
    corpus = [
        f"incident {i}: the {noun} was {verb} during the {when} maintenance window"
        for i, (noun, verb, when) in enumerate(
            [("database", "restored", "friday"), ("cache", "flushed", "sunday"),
             ("index", "rebuilt", "monday"), ("cluster", "drained", "tuesday")] * 12
        )
    ]
    system = OrganizationalMemory(config, embedder=HashingEmbedder(dim=64))
    assert isinstance(system.embedder, WhitenedEmbedder)
    assert not system.whitener.is_fitted
    system.bootstrap(corpus, epochs=2)
    assert system.whitener.is_fitted

    plain = HashingEmbedder(dim=64).encode(corpus)
    encoded = system.embedder.encode(corpus)
    assert not torch.allclose(encoded, plain, atol=1e-3)
    assert torch.allclose(encoded, system.whitener.transform(plain), atol=1e-6)
    assert anisotropy(encoded) < anisotropy(plain)

    # And the whole loop still works with it on. remember_rule, because the
    # novelty gate correctly refuses text this close to the bootstrap corpus.
    system.remember_rule("incident 99: the wombat was launched during the eclipse")
    assert len(system) == 1
    top = system.recall("wombat launched eclipse").top
    assert top is not None and "wombat" in top.record.text


# ------------------------------------------------------ WhitenedEmbedder.encode_query


class _AsymmetricStub:
    """A stub embedder whose document and query paths are deliberately
    distinguishable, the way BGE's query-instruction prefix distinguishes its
    query path from its document path.

    `encode` and `encode_query` differ only in coordinate 0 (0.0 for
    documents, 1.0 for queries); everything else is a deterministic function
    of the text so two calls to the same path are still comparable.
    """

    dim = 8

    def _base(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            out[i, 1] = float((len(t) % 5) + 1)
            out[i, 2] = float((sum(map(ord, t)) % 7) + 1)
        return out

    def encode(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        out = self._base(list(texts))
        out[:, 0] = 0.0
        return out

    def encode_query(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        out = self._base(list(texts))
        out[:, 0] = 1.0
        return out


def test_whitened_embedder_exposes_encode_query():
    wrapped = WhitenedEmbedder(_AsymmetricStub())
    assert hasattr(wrapped, "encode_query")


def test_whitened_encode_query_is_not_the_document_path():
    """The regression this exists to catch: `getattr(embedder, "encode_query",
    embedder.encode)` -- the pattern every call site in `experiments/` uses --
    must find a *real* query encoder on `WhitenedEmbedder`, not silently fall
    back to `encode`. If `encode_query` were ever reimplemented as `return
    self.encode(texts)` (dropping the delegation to the wrapped embedder's own
    query path), this test fails because the two outputs would be identical.
    """
    corpus = [f"document number {i} about widgets" for i in range(16)]
    wrapped = WhitenedEmbedder(_AsymmetricStub()).fit(corpus)
    assert wrapped.whitener.is_fitted

    same_text = ["a widget question"]
    doc_side = wrapped.encode(same_text)
    query_side = wrapped.encode_query(same_text)

    assert not torch.allclose(doc_side, query_side)

    # And it isn't just "different by construction" -- it must be the SAME
    # fitted whitener transform applied to the wrapped embedder's own
    # encode_query output, i.e. documents and queries land in the same space.
    inner = _AsymmetricStub()
    expected = wrapped.whitener.transform(inner.encode_query(same_text))
    assert torch.allclose(query_side, expected, atol=1e-6)

    # The getattr fallback pattern used throughout experiments/recall_ablation.py
    # must resolve to the real query encoder, not to encode().
    resolved = getattr(wrapped, "encode_query", wrapped.encode)
    assert torch.allclose(resolved(same_text), query_side)


def test_whitened_encode_query_falls_back_to_encode_when_wrapped_embedder_lacks_it():
    """A wrapped embedder with no query/document distinction (e.g. HashingEmbedder)
    must still work: `encode_query` falls back to the wrapped embedder's `encode`,
    then applies the same whitener transform as the document path."""
    corpus = [f"incident {i} report" for i in range(16)]
    wrapped = WhitenedEmbedder(HashingEmbedder(dim=32)).fit(corpus)
    text = ["a fresh incident"]
    assert torch.allclose(wrapped.encode(text), wrapped.encode_query(text))


def test_whitened_encode_query_passes_through_unfitted_and_warns_once():
    """Same pass-through + warn-once contract as `encode`, exercised on the
    query path specifically, since that path used not to exist at all."""
    wrapped = WhitenedEmbedder(_AsymmetricStub())
    assert not wrapped.whitener.is_fitted

    inner = _AsymmetricStub()
    text = ["unfitted query"]
    with pytest.warns(UserWarning, match="not fitted"):
        out = wrapped.encode_query(text)
    assert torch.allclose(out, inner.encode_query(text))

    # warn-once: a second call must not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wrapped.encode_query(text)


# --------------------------------------- prototype/library equivalence (ticket 06)
#
# `experiments/superposition.py` used to carry private copies of the whitening,
# superposition and decode operations, so RESULTS.md V.5/VI.2/VI.3 claimed to
# have been re-measured "through the library classes rather than the prototype"
# while actually running the duplicates. The experiment now calls the library.
# These tests pin the two implementations together so they cannot silently
# diverge again -- they assert VALUES, not that a function exists.


def _prototype_whiten(x, *, fit_on=None, floor=1e-2):
    """The exact SVD whitening `experiments/superposition.py` used to define."""
    basis = x if fit_on is None else fit_on
    mean = basis.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(basis - mean, full_matrices=False)
    scale = (s / math.sqrt(basis.shape[0])).clamp_min(floor)
    out = ((x - mean) @ vh.T / scale) @ vh
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _corpus(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Anisotropic on purpose: a shared offset puts every row in a narrow cone,
    # which is the regime whitening exists to fix. Isotropic Gaussians would
    # make the comparison vacuous (HANDOFF §5).
    x = torch.randn(n, d, generator=g) + 3.0 * torch.randn(1, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def test_library_whitener_matches_the_prototype_transform():
    """In-sample (transductive) fit: `Whitener` must reproduce the prototype."""
    x = _corpus(256, 32)
    library = Whitener(32, WhiteningConfig(floor=1e-2)).fit(x).transform(x)
    assert torch.allclose(library, _prototype_whiten(x), atol=1e-5)


def test_library_whitener_matches_the_prototype_on_a_held_out_fit():
    """Fit on one corpus, apply to another -- the discipline RESULTS.md V.2's
    held-out capacity numbers depend on. A whitener that silently refitted on
    the transformed batch would pass the in-sample test above and fail here."""
    fit_pool = _corpus(512, 32, seed=1)
    target = _corpus(128, 32, seed=2)
    library = Whitener(32, WhiteningConfig(floor=1e-2)).fit(fit_pool).transform(target)
    assert torch.allclose(library, _prototype_whiten(target, fit_on=fit_pool), atol=1e-5)


def test_library_superpose_and_decode_match_the_prototype_cosine_path():
    """`superpose` + `decode` must rank exactly as the prototype's normalised
    mean + cosine top-k did. This is the identity V.2's capacity table rests on:
    for unit-norm patterns under a uniform prior the attention logits order the
    store identically to cosine, so porting the experiment to the library must
    not move a single published cell."""
    patterns = torch.nn.functional.normalize(_corpus(200, 32, seed=3), dim=-1)
    mhn = ModernHopfieldNetwork(32, HopfieldConfig())
    mhn.write(patterns)

    g = torch.Generator().manual_seed(7)
    for k in (2, 4, 8, 16):
        for _ in range(10):
            members = torch.randperm(200, generator=g)[:k].tolist()

            state = mhn.superpose(members)
            prototype_state = patterns[members].mean(dim=0)
            prototype_state = prototype_state / prototype_state.norm().clamp_min(1e-12)
            assert torch.allclose(state, prototype_state, atol=1e-6)

            cos = patterns @ prototype_state
            library_top = mhn.decode(state, k).tolist()
            prototype_top = torch.topk(cos, k).indices.tolist()

            # Compared by SIMILARITY, not by index. `decode` ranks by
            # `beta*(X xi - ||x||^2/2) + log w`; with beta=128 that amplifies
            # float rounding, so two candidates whose cosines differ by ~1e-8
            # can swap places against a direct matmul. Measured here: indices
            # 60 and 189 at cosine 0.969203949 vs 0.969204009. Asserting index
            # equality would make this test fail on an effectively exact tie,
            # while asserting the selected similarities agree still catches a
            # real divergence (a genuinely worse memory being selected).
            assert torch.allclose(
                cos[library_top].sort(descending=True).values,
                cos[prototype_top].sort(descending=True).values,
                atol=1e-6,
            )


def test_prototype_and_library_agree_on_measured_capacity():
    """End to end: the per-item recall number V.2 publishes is the same whether
    it is computed through the library or the prototype. Checks the metric, not
    just the intermediate tensors."""
    patterns = torch.nn.functional.normalize(_corpus(300, 48, seed=5), dim=-1)
    mhn = ModernHopfieldNetwork(48, HopfieldConfig())
    mhn.write(patterns)

    def recall(use_library: bool, k: int, trials: int = 25) -> float:
        g = torch.Generator().manual_seed(11)
        hits = 0
        for _ in range(trials):
            members = torch.randperm(300, generator=g)[:k].tolist()
            if use_library:
                top = mhn.decode(mhn.superpose(members), k).tolist()
            else:
                state = patterns[members].mean(dim=0)
                state = state / state.norm().clamp_min(1e-12)
                top = torch.topk(patterns @ state, k).indices.tolist()
            hits += len(set(top) & set(members)) / k
        return hits / trials

    for k in (4, 8):
        assert recall(True, k) == pytest.approx(recall(False, k), abs=1e-9)
