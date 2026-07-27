"""Tests for the v8 RelationalHead (CPU-only, no mocks needed, no network).

The two properties this file exists to defend are the ones carried over from
``SlotResonanceFusion`` and called out in CLAUDE.md as design claims:

1. Object evidence fades along ONE graded path shared by dream staleness and
   train-time modality dropout — continuous in the weight, exactly annihilating
   at 0, never a binary mask and never a separate "is dream" flag.
2. The last-executed-action token is never faded.

Everything else here is shape/plumbing hygiene: config-driven dims (no
hardcoded numbers), gradient reachability of every input, permutation
equivariance over the object set, batch independence, and the param budget.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from microvla.config import DEFAULT_CONFIG
from microvla.relational import RelationalHead
from microvla.relational.relational_head import _TYPE_INDEX

CFG = DEFAULT_CONFIG

#: This module's share of the 9M trainable budget (it inherits fusion's 5.0M
#: hard cap, but the v8 ledger sizes it at 2.4M so the HRM backbone fits).
REL_TARGET = 2_400_000
REL_HARD_CAP = 5_000_000

#: Floating-point noise floor of one forward pass, measured, not guessed.
#: Reordering the tokens of a 25-token attention changes the summation order,
#: so two runs that are mathematically identical still differ by ~2.4e-6 in
#: float32 (measured on the object-permutation case, where exact invariance is
#: guaranteed by construction). Every "input X must change the output" test
#: below therefore asserts an effect SIZE against this floor instead of
#: ``not allclose(atol=1e-6)``: at 1e-6 the assertion is satisfied by the noise
#: alone and passes on an implementation where X is genuinely ignored (verified
#: — collapsing the three text type embeddings into one makes the text order
#: provably irrelevant, and a 1e-6 threshold still called it "changed"). The
#: real effects measured here are 3e-2..9e-1, four orders above the floor, so
#: 1e-3 separates them with a wide margin.
FP_NOISE = 2.4e-6
MIN_EFFECT = 1e-3


def _gen(seed: int) -> torch.Generator:
    """A private RNG.

    Every random draw in this file goes through a local generator and module
    construction happens inside ``torch.random.fork_rng`` — the global RNG is
    neither read nor written, so this file cannot shift the sampling of any
    test that runs after it.
    """
    return torch.Generator().manual_seed(seed)


def _inputs(batch: int = 2, seed: int = 0, cfg=CFG, weight: float = 1.0) -> dict:
    """Builds one full input dict in the canonical (standardized-ish) space."""
    g = _gen(seed)
    return {
        "next_emb": torch.randn(batch, cfg.vis_dim, generator=g),
        "obj_emb": torch.randn(batch, cfg.max_objects, cfg.vis_dim, generator=g),
        "obj_center": torch.rand(batch, cfg.max_objects, 2, generator=g),
        "obj_weight": torch.full((batch, cfg.max_objects), weight),
        "text_tokens": torch.randn(batch, cfg.n_text_tokens, cfg.text_dim, generator=g),
        "last_action": torch.rand(batch, cfg.num_servos, generator=g) * 2.0 - 1.0,
    }


def _head(seed: int = 0, cfg=CFG) -> RelationalHead:
    """Builds a head in eval mode (no train-time evidence fade in the way)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = RelationalHead(cfg)
    head.eval()
    return head


class TestShapesAndPlumbing:
    def test_output_shape(self):
        head = _head()
        out = head(**_inputs(batch=2))
        assert out.shape == (2, CFG.rel_tokens, CFG.rel_dim)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("batch", [1, 3, 8])
    def test_batch_sizes(self, batch):
        head = _head()
        out = head(**_inputs(batch=batch, seed=batch))
        assert out.shape == (batch, CFG.rel_tokens, CFG.rel_dim)

    def test_last_action_defaults_to_zeros(self):
        """``None`` must mean "episode start", i.e. an all-zero command."""
        head = _head()
        inputs = _inputs(batch=2, seed=3)
        explicit = dict(inputs)
        explicit["last_action"] = torch.zeros(2, CFG.num_servos)
        implicit = dict(inputs)
        implicit["last_action"] = None
        assert torch.equal(head(**explicit), head(**implicit))

    def test_dims_come_from_config_not_hardcoded(self):
        """Every dim must track cfg — a hardcoded 384/8/12 would fail here."""
        cfg = dataclasses.replace(
            CFG, rel_dim=128, rel_tokens=5, rel_heads=4, rel_layers=1, max_objects=3
        )
        head = _head(cfg=cfg)
        out = head(**_inputs(batch=2, cfg=cfg))
        assert out.shape == (2, 5, 128)
        # Token layout: queries | latent | 3 text | 3 objects | action.
        assert head.n_tokens == 5 + 1 + cfg.n_text_tokens + 3 + 1
        assert head.object_slice.stop - head.object_slice.start == cfg.max_objects
        assert head.action_index == head.n_tokens - 1

    def test_rejects_a_text_token_count_it_cannot_type(self):
        """Each text position owns a type embedding; a 4th would borrow OBJECT's."""
        cfg = dataclasses.replace(CFG, n_text_tokens=4)
        with pytest.raises(ValueError, match="ordered text"):
            RelationalHead(cfg)

    def test_all_slots_empty_is_finite(self):
        """A frame where the detector found nothing at all must not blow up."""
        head = _head()
        inputs = _inputs(batch=2, seed=11, weight=0.0)
        out = head(**inputs)
        assert torch.isfinite(out).all()


class TestGradients:
    def test_gradient_reaches_every_input(self):
        head = _head(seed=1)
        inputs = _inputs(batch=2, seed=2, weight=0.6)
        for tensor in inputs.values():
            tensor.requires_grad_(True)

        head(**inputs).pow(2).sum().backward()

        for name, tensor in inputs.items():
            assert tensor.grad is not None, f"no gradient path to {name}"
            assert torch.isfinite(tensor.grad).all(), f"non-finite grad for {name}"
            assert tensor.grad.abs().sum() > 0.0, f"zero gradient for {name}"

    def test_center_reaches_output_through_the_pairwise_bias(self):
        """The object-object geometry bias must actually be wired into attention.

        Zeroing ``geom_proj`` removes the per-object center path, leaving
        ``rel_bias`` (displacement -> attention logit) as the ONLY route from a
        center to the output. If the additive bias were silently dropped (a
        3D ``attn_mask`` is easy to lose), moving every object would become a
        no-op and this test fails.

        Asserted as an effect size (see ``MIN_EFFECT``): dropping the mask
        leaves a residual difference of pure float32 reordering noise, which
        an ``atol=1e-6`` inequality would happily accept.
        """
        head = _head(seed=4)
        with torch.no_grad():
            head.geom_proj.weight.zero_()
            head.geom_proj.bias.zero_()

        inputs = _inputs(batch=2, seed=5)
        moved = dict(inputs)
        moved["obj_center"] = torch.rand(2, CFG.max_objects, 2, generator=_gen(51))
        effect = (head(**inputs) - head(**moved)).abs().max().item()
        assert effect > MIN_EFFECT, (
            f"moving every object changed the read-out by {effect:.2e}, which is "
            f"at the {FP_NOISE:.1e} noise floor — the pairwise bias is not wired in"
        )


class TestEvidenceFade:
    """Property 1: one graded path, continuous, exact at 0."""

    def test_zero_weight_object_cannot_reach_the_output(self):
        """A weight-0 slot is inert: its embedding and center are unreadable."""
        head = _head(seed=6)
        inputs = _inputs(batch=2, seed=7)
        inputs["obj_weight"][:, 0] = 0.0

        replaced = dict(inputs)
        replaced["obj_emb"] = inputs["obj_emb"].clone()
        replaced["obj_emb"][:, 0] = torch.randn(2, CFG.vis_dim, generator=_gen(52)) * 10.0
        replaced["obj_center"] = inputs["obj_center"].clone()
        replaced["obj_center"][:, 0] = torch.rand(2, 2, generator=_gen(53))

        assert torch.equal(head(**inputs), head(**replaced))

    def test_zero_weight_token_is_exactly_the_type_embedding(self):
        """Content annihilated, slot preserved — a faded token is not masked out.

        Keeping the (content-free) slot in the sequence is what makes the fade
        continuous: masking would make weight 0 a different computation graph
        rather than the limit of a stale one.
        """
        head = _head(seed=8)
        inputs = _inputs(batch=2, seed=9)
        inputs["obj_weight"][:, 2] = 0.0
        tokens = head._build_tokens(
            inputs["next_emb"],
            inputs["obj_emb"],
            inputs["obj_center"],
            inputs["obj_weight"],
            inputs["text_tokens"],
            inputs["last_action"],
        )
        faded = tokens[:, head.object_slice.start + 2]
        expected = head.type_emb[_TYPE_INDEX["object"]].expand_as(faded)
        assert torch.equal(faded, expected)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_object_fade_is_monotone(self, seed):
        """1.0 -> 0.0 attenuates monotonically, not just at the endpoint.

        Measured from BOTH ends: as the weight drops, the output moves
        monotonically away from the full-evidence output and monotonically
        toward the no-evidence output. Endpoint-only checks would pass for a
        binary mask; this sweep would not.
        """
        head = _head(seed=seed)
        inputs = _inputs(batch=1, seed=seed + 20)
        sweep = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0)

        outs = []
        for w in sweep:
            weights = torch.ones(1, CFG.max_objects)
            weights[:, 0] = w
            outs.append(head(**{**inputs, "obj_weight": weights}))

        to_absent = [(o - outs[-1]).norm().item() for o in outs]
        to_full = [(o - outs[0]).norm().item() for o in outs]
        assert all(a > b for a, b in zip(to_absent, to_absent[1:])), to_absent
        assert all(a < b for a, b in zip(to_full, to_full[1:])), to_full
        assert to_absent[-1] == 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_dream_rollout_fade_is_monotone(self, seed):
        """The real dream regime: every held box decays together.

        The weights walked here are exactly what the JEPA loop emits across a
        15-tick dream — ``confidence * staleness_decay**k`` — so this is the
        deployment path, not a synthetic sweep.
        """
        head = _head(seed=seed)
        inputs = _inputs(batch=1, seed=seed + 30)
        confidence = 0.85

        outs = []
        for k in range(CFG.dream_ticks_per_real + 1):
            weights = torch.full(
                (1, CFG.max_objects), confidence * CFG.staleness_decay**k
            )
            outs.append(head(**{**inputs, "obj_weight": weights}))
        absent = head(**{**inputs, "obj_weight": torch.zeros(1, CFG.max_objects)})

        to_absent = [(o - absent).norm().item() for o in outs]
        assert all(a > b for a, b in zip(to_absent, to_absent[1:])), to_absent

    def test_modality_dropout_only_fires_in_train_mode(self):
        head = _head(seed=10)
        weights = torch.full((64, CFG.max_objects), 0.9)

        head.eval()
        assert torch.equal(head._fade_evidence(weights), weights)

        head.train()
        with torch.random.fork_rng(devices=[]):
            faded = head._fade_evidence(weights)
        assert (faded <= weights + 1e-6).all(), "the fade must never brighten evidence"
        assert (faded < weights - 1e-6).any(), "no sample faded at p=0.3 over B=64"

    def test_modality_dropout_fades_all_objects_together(self):
        """One draw per SAMPLE, shared across K — a dream stales every box at once."""
        head = _head(seed=11)
        head.train()
        weights = torch.full((32, CFG.max_objects), 0.9)
        with torch.random.fork_rng(devices=[]):
            faded = head._fade_evidence(weights)
        per_sample_spread = (faded.max(dim=1).values - faded.min(dim=1).values).abs()
        assert torch.allclose(per_sample_spread, torch.zeros_like(per_sample_spread))

    def test_forward_actually_applies_the_fade_in_train_mode(self):
        """The fade must be wired into ``forward``, not only into the helper.

        Every other dropout test in this class pokes ``_fade_evidence``
        directly, so all of them stay green on a ``forward`` that quietly uses
        the caller's raw weights (verified by mutation: replacing
        ``self._fade_evidence(obj_weight)`` with ``obj_weight`` survived the
        whole file). That mutation would delete the training-inference
        alignment the dream regime is built on — the network would only ever
        see full-evidence weights at train time and meet decayed ones at 30 Hz.
        With ``modality_dropout=1.0`` every sample is faded by a uniform draw,
        so train-mode output must differ from eval-mode output and must be
        resampled between calls.
        """
        cfg = dataclasses.replace(CFG, modality_dropout=1.0)
        head = _head(seed=29, cfg=cfg)
        inputs = _inputs(batch=6, seed=30, cfg=cfg, weight=0.9)

        unfaded = head(**inputs)  # eval mode
        head.train()
        # The fade draws from the global RNG (that is the module's own
        # contract, matching fusion); fork so this file still leaves the global
        # stream exactly as it found it.
        with torch.random.fork_rng(devices=[]):
            draws = [head(**inputs) for _ in range(4)]

        assert all(not torch.equal(d, unfaded) for d in draws), (
            "train-mode forward matched the eval-mode output exactly — "
            "forward() is not applying the evidence fade"
        )
        assert any(not torch.equal(a, b) for a, b in zip(draws, draws[1:])), (
            "repeated train-mode forwards were identical — the fade is not resampled"
        )

    def test_train_mode_fade_only_ever_dims_the_evidence(self):
        """The train-time draw must walk DOWN the same curve dreams walk down.

        A fade that could brighten (or that landed anywhere other than on the
        weight axis) would put train-time inputs off the deployment manifold.
        Every draw must sit no further from the no-evidence output than the
        un-faded input does.
        """
        cfg = dataclasses.replace(CFG, modality_dropout=1.0)
        head = _head(seed=31, cfg=cfg)
        inputs = _inputs(batch=1, seed=32, cfg=cfg, weight=0.9)
        absent = head(**{**inputs, "obj_weight": torch.zeros(1, cfg.max_objects)})
        full_distance = (head(**inputs) - absent).norm().item()

        head.train()
        with torch.random.fork_rng(devices=[]):
            for _ in range(16):
                faded = (head(**inputs) - absent).norm().item()
                assert faded <= full_distance + 1e-5, (faded, full_distance)

    def test_modality_dropout_rate_zero_is_a_no_op(self):
        cfg = dataclasses.replace(CFG, modality_dropout=0.0)
        head = _head(seed=12, cfg=cfg)
        head.train()
        inputs = _inputs(batch=4, seed=13, cfg=cfg, weight=0.7)
        assert torch.equal(head(**inputs), head(**inputs))


class TestActionTokenNeverFades:
    """Property 2: fusion's 8th token, carried over."""

    def test_action_token_is_bit_identical_across_evidence_levels(self):
        head = _head(seed=14)
        inputs = _inputs(batch=2, seed=15)
        rows = []
        for w in (1.0, 0.5, 0.0):
            weights = torch.full((2, CFG.max_objects), w)
            tokens = head._build_tokens(
                inputs["next_emb"],
                inputs["obj_emb"],
                inputs["obj_center"],
                weights,
                inputs["text_tokens"],
                inputs["last_action"],
            )
            rows.append(tokens[:, head.action_index])
        assert torch.equal(rows[0], rows[1])
        assert torch.equal(rows[0], rows[2])

    def test_action_still_moves_the_output_with_no_visual_evidence(self):
        """Total perception blackout must not silence the controller's own state."""
        head = _head(seed=16)
        inputs = _inputs(batch=2, seed=17, weight=0.0)
        other = dict(inputs)
        other["last_action"] = -inputs["last_action"]
        effect = (head(**inputs) - head(**other)).abs().max().item()
        assert effect > MIN_EFFECT, (
            f"flipping the last action moved the read-out by only {effect:.2e} "
            f"with no visual evidence — the action token is being silenced too"
        )

    def test_modality_dropout_never_touches_the_action_token(self):
        """Train mode fades boxes; the action token must be untouched anyway."""
        head = _head(seed=18)
        inputs = _inputs(batch=2, seed=19)
        head.train()
        with torch.random.fork_rng(devices=[]):
            baseline = head._build_tokens(
                inputs["next_emb"],
                inputs["obj_emb"],
                inputs["obj_center"],
                head._fade_evidence(inputs["obj_weight"]),
                inputs["text_tokens"],
                inputs["last_action"],
            )[:, head.action_index]
            for _ in range(8):
                tokens = head._build_tokens(
                    inputs["next_emb"],
                    inputs["obj_emb"],
                    inputs["obj_center"],
                    head._fade_evidence(inputs["obj_weight"]),
                    inputs["text_tokens"],
                    inputs["last_action"],
                )
                assert torch.equal(tokens[:, head.action_index], baseline)


class TestInvariances:
    def test_deterministic_in_eval_mode(self):
        head = _head(seed=21)
        inputs = _inputs(batch=3, seed=22, weight=0.4)
        first = head(**inputs)
        for _ in range(3):
            assert torch.equal(head(**inputs), first)

    def test_batch_independence(self):
        """No cross-sample leakage: attention is over tokens, not over the batch."""
        head = _head(seed=23)
        inputs = _inputs(batch=4, seed=24, weight=0.8)
        batched = head(**inputs)
        for i in range(4):
            single = head(**{k: v[i : i + 1] for k, v in inputs.items()})
            assert torch.allclose(batched[i : i + 1], single, atol=1e-5)

    def test_permutation_equivariance_over_the_object_set(self):
        """Proposal order carries no information — the detector re-orders it.

        Objects share one type embedding and get no index embedding, so
        permuting (emb, center, weight) together must leave the read-out
        unchanged. If someone adds a per-slot positional embedding, this fails.
        """
        head = _head(seed=25)
        inputs = _inputs(batch=2, seed=26)
        inputs["obj_weight"] = torch.rand(2, CFG.max_objects, generator=_gen(54))
        perm = torch.randperm(CFG.max_objects, generator=_gen(55))

        permuted = dict(inputs)
        permuted["obj_emb"] = inputs["obj_emb"][:, perm]
        permuted["obj_center"] = inputs["obj_center"][:, perm]
        permuted["obj_weight"] = inputs["obj_weight"][:, perm]

        assert torch.allclose(head(**inputs), head(**permuted), atol=1e-5)

    def test_text_order_matters(self):
        """source/target are ordered roles: swapping them must change the read-out.

        The head has no positional encoding — the three DISTINCT text type
        embeddings are the only thing that makes "move can to ball" differ from
        "move ball to can". Collapse them to one and the sequence becomes a
        set, so the swap is an exact no-op up to float reordering; that is
        precisely the mutation an ``atol=1e-6`` inequality fails to catch
        (measured: 2.3e-6 residual, i.e. indistinguishable from noise), so the
        assertion is on the effect size.
        """
        head = _head(seed=27)
        inputs = _inputs(batch=2, seed=28)
        swapped = dict(inputs)
        swapped["text_tokens"] = inputs["text_tokens"][:, [0, 2, 1]]
        effect = (head(**inputs) - head(**swapped)).abs().max().item()
        assert effect > MIN_EFFECT, (
            f"swapping the source and target phrases moved the read-out by "
            f"{effect:.2e} — the text roles are not distinguishable"
        )


class TestBudget:
    def test_param_count_under_target(self, capsys):
        head = RelationalHead(CFG)
        n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        with capsys.disabled():
            print(f"\nRelationalHead: {n_params:,d} params ({n_params / 1e6:.3f}M)")
        assert n_params <= REL_TARGET, (
            f"RelationalHead has {n_params:,d} params, over its {REL_TARGET:,d} "
            f"share of the {CFG.trainable_param_budget:,d} trainable budget"
        )
        assert n_params <= REL_HARD_CAP

    def test_runtime_state_is_not_stored_in_parameters(self):
        """The head is stateless — no anchors, no buffers except the constants.

        The drift/HRM path owns temporal state; a relational head that
        accumulated any would silently break the JEPA loop's replay and the
        deployment-exact rollouts in stage A.
        """
        head = RelationalHead(CFG)
        buffer_names = {name for name, _ in head.named_buffers()}
        assert buffer_names == {"fourier_freqs"}
        assert not any(b.requires_grad for _, b in head.named_buffers())
