"""The trainer and the deployment loop must assemble the SAME planner call.

Seven defects in this stack have been the same shape: two sides of a train/deploy
pair disagreeing about what a value means, each side individually correct and
individually tested (paper.md 4t, 4u, 4v). Two of them lived here, in the gap
between ``train/train_batched.py``'s per-step assembly and
``microvla/jepa/loop.py``'s:

* **defect 6** — a real-tick detection MISS: the loop held the last-known box at
  ``miss_decay ** age`` while the corpus bakes weight 0 at the (0.5, 0.5)
  fallback, so the policy read "the object is there, where it used to be" on
  exactly the ticks training taught "no evidence". Measured deployment weights
  0.156 / 0.109 / 0.077 on ticks the corpus zeroed.
* **defect 7** — the staleness factor recovered as
  ``box_weight.max() / max(source.conf, target.conf)``, which is ``0 / 1e-6 = 0``
  when BOTH roles miss, zeroing every class-agnostic proposal.

Neither was visible to any existing test, because ``tests/test_jepa_loop.py``
pinned the loop to the loop's behaviour and the trainer's tests pinned the
trainer to its own. The only thing that finds this class of bug is running both
sides on the same input and diffing, which is what this module does — on CPU,
with mocks, in the normal suite, so the next divergence fails the build instead
of costing a training run and three false root causes.

The action token is teacher-forced here on purpose. That it is NOT teacher-forced
at deployment is a real and separately-tracked asymmetry (defect 8, exposure
bias, paper.md 4v), but it is a property of the training protocol rather than a
disagreement about assembly — holding it fixed is what isolates the assembly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.jepa.loop import JEPALoop
from microvla.perception.text_encoder import MockTaskEncoder
from microvla.perception.yolo_world import BoxObs, Perception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.mock_trm import MockTRM

CFG = DEFAULT_CONFIG
T = 6
MISS_AT = 3          # the tick whose detection fails — where defect 6 lived


@pytest.fixture
def episode():
    """One synthetic episode with a real-tick MISS in the middle."""
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g)
    bw = torch.rand(T, 2, generator=g) * 0.5 + 0.4
    bw[MISS_AT] = 0.0                    # both roles miss: defects 6 AND 7
    ctr_s = torch.rand(T, 2, generator=g)
    ctr_t = torch.rand(T, 2, generator=g)
    # A miss is baked as the (0.5, 0.5) fallback, not as a stale center.
    ctr_s[MISS_AT] = 0.5
    ctr_t[MISS_AT] = 0.5
    return {
        "frame_embs": r(T, CFG.vis_dim),
        "source_box_embs": r(T, CFG.vis_dim),
        "target_box_embs": r(T, CFG.vis_dim),
        "source_centers": ctr_s,
        "target_centers": ctr_t,
        "box_weights": bw,
        "pwm_targets": torch.tanh(r(T, CFG.plan_steps, CFG.num_servos)),
        "proprio": torch.cat([r(T, 9), torch.ones(T, 1)], dim=-1),
        "text_tokens": r(3, CFG.text_dim),
    }


@pytest.fixture
def modules():
    torch.manual_seed(0)
    return dict(fusion=SlotResonanceFusion(CFG).eval(),
                drift=AnchoredDriftEncoder(CFG).eval(),
                trm=MockTRM(CFG).eval(),
                planner=ChronoQueryPlanner(CFG).eval())


def _trainer_inputs(episode, modules):
    """What train/train_batched.py hands the planner, per timestep."""
    from train.train_batched import _boxes, real_paths

    batch = {k: v.unsqueeze(0) for k, v in episode.items() if k != "text_tokens"}
    batch["text_tokens"] = episode["text_tokens"].unsqueeze(0)
    out = []
    with torch.no_grad():
        fused_all, delta_all = real_paths(batch, modules["fusion"], modules["drift"],
                                          CFG, ablate=False)
        for t in range(T):
            sbe, tbe, sc, tc, bw = _boxes(batch, t, 1.0, CFG, False)
            out.append({
                "fused": fused_all[t],
                "state_delta": delta_all[t],
                "geometry": torch.cat([sc, tc, bw], dim=-1),
                "current_emb": batch["frame_embs"][:, t],
            })
    return out


def _deploy_inputs(episode, modules, cfg=CFG):
    """What microvla/jepa/loop.py hands the planner, on the same episode."""
    class Replay:
        def __init__(self):
            self.i = 0

        def set_role_prompts(self, source, target=None):
            pass

        def set_classes(self, names):
            pass

        def perceive(self, _frame):
            i = min(self.i, T - 1)
            self.i += 1
            box = lambda e, c, j: BoxObs(
                emb=episode[e][i], center=episode[c][i], xyxy=torch.zeros(4),
                confidence=float(episode["box_weights"][i][j]))
            return Perception(frame_emb=episode["frame_embs"][i],
                              source=box("source_box_embs", "source_centers", 0),
                              target=box("target_box_embs", "target_centers", 1))

    loop = JEPALoop(cfg, MockTaskEncoder(cfg.text_dim), Replay(),
                    modules["fusion"], modules["drift"], modules["trm"],
                    modules["planner"])
    loop.set_task("move can to ball")
    loop.perception = Replay()      # fresh counter: set_task must not consume one
    tt = episode["text_tokens"]
    loop._task.command_emb, loop._task.source_emb, loop._task.target_emb = (
        tt[0], tt[1], tt[2])

    seen = []
    hook = modules["planner"].register_forward_pre_hook(
        lambda _m, _a, kw: seen.append({k: v.detach().clone()
                                        for k, v in kw.items() if torch.is_tensor(v)}),
        with_kwargs=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for t in range(T):
        loop.tick(frame, proprio=episode["proprio"][t].numpy())
        # Teacher-force the action token, exactly as stage B does, so the only
        # variable left is the assembly (see the module docstring).
        loop._last_action = episode["pwm_targets"][t, 0]
    hook.remove()
    return seen


class TestPlannerCallParity:
    def test_geometry_matches_including_on_a_missed_detection(self, episode, modules):
        """Defect 6: the loop held a stale box where the corpus bakes zero."""
        a = _trainer_inputs(episode, modules)
        b = _deploy_inputs(episode, modules)
        assert len(b) >= T
        for t in range(T):
            assert torch.allclose(a[t]["geometry"], b[t]["geometry"], atol=1e-5), (
                f"geometry diverged at t={t}\n"
                f"  trainer {a[t]['geometry'].reshape(-1).tolist()}\n"
                f"  deploy  {b[t]['geometry'].reshape(-1).tolist()}\n"
                f"t={MISS_AT} is a MISS: the corpus bakes weight 0 at the "
                f"(0.5, 0.5) fallback, so the loop must not substitute a held box."
            )

    def test_a_miss_reaches_the_planner_as_zero_weight(self, episode, modules):
        """The property behind defect 6, stated directly."""
        b = _deploy_inputs(episode, modules)
        w = b[MISS_AT]["geometry"].reshape(-1)[-2:]
        assert torch.count_nonzero(w) == 0, (
            f"a missed detection carried weight {w.tolist()} into the planner; "
            f"training fed 0 on these ticks."
        )

    def test_fused_and_state_delta_match(self, episode, modules):
        a = _trainer_inputs(episode, modules)
        b = _deploy_inputs(episode, modules)
        for t in range(T):
            for key in ("state_delta", "fused"):
                assert torch.allclose(a[t][key], b[t][key], atol=1e-4), (
                    f"{key} diverged at t={t} — the trainer and the loop built "
                    f"the same call from the same modules and disagreed."
                )


class TestTheParityCheckHasTeeth:
    """A regression test is worth what it catches, so catch the known defect.

    Re-introducing defect 6 (``miss_hold=True``, the loop's old default) must
    make the parity assertion above fail. Without this, a future refactor could
    make ``_deploy_inputs`` silently stop exercising the loop — the tests would
    still pass and would still be measuring nothing, which is precisely the
    failure mode this whole file exists to prevent.
    """

    def test_reintroducing_the_miss_hold_defect_breaks_parity(self, episode, modules):
        import dataclasses

        held = dataclasses.replace(CFG, miss_hold=True)
        a = _trainer_inputs(episode, modules)
        b = _deploy_inputs(episode, modules, cfg=held)
        assert not torch.allclose(a[MISS_AT]["geometry"], b[MISS_AT]["geometry"],
                                  atol=1e-5), (
            "miss_hold=True no longer diverges from the corpus, so the parity "
            "test is not exercising the miss path and would not catch defect 6."
        )
        w = b[MISS_AT]["geometry"].reshape(-1)[-2:]
        assert torch.count_nonzero(w) > 0, "the held box should carry decayed weight"


class TestHRMReceivesEndEffector:
    """The HRM's metric branch must be fed on BOTH sides.

    ``HRMBackbone.forward`` accepts ``eef`` and builds
    ``[eef, eef - anchor, validity]`` through ``eef_proj``, but ``DriftAdapter``
    called ``self.hrm(frame_emb, is_real=True)`` with no eef, and
    ``_eef_features(None, ...)`` returns ZEROS. So the whole metric branch
    contributed a constant, ``eef_proj`` received gradient in no code path, and
    the module designed to act as a learned controller over end-effector error
    was running on vision alone — in training AND at deployment.
    """

    def test_drift_adapter_forwards_eef_to_the_hrm(self):
        import dataclasses

        from microvla.v8 import DriftAdapter

        d = DriftAdapter(CFG)
        emb = torch.randn(2, CFG.vis_dim)
        emb2 = torch.randn(2, CFG.vis_dim)
        eef = torch.randn(2, CFG.waypoint_dim)
        # The first forward after reset() is the ANCHOR tick and returns an
        # exactly-zero code by contract, so the comparison must step past it.
        d.reset(); d(emb); a = d(emb2)                       # metric branch fed nothing
        d.reset(); d(emb, eef=torch.zeros_like(eef)); b = d(emb2, eef=eef)
        assert not torch.allclose(a, b, atol=1e-6), (
            "passing eef changed nothing — the metric branch is still inert"
        )

    def test_eef_proj_receives_gradient(self):
        from microvla.v8 import DriftAdapter

        d = DriftAdapter(CFG)
        with torch.enable_grad():          # independent of ambient grad mode
            d.reset()
            d(torch.randn(2, CFG.vis_dim), eef=torch.zeros(2, CFG.waypoint_dim))
            # A large EEF displacement: the metric branch is one addend among
            # several into a GRU drive, so its gradient is small in absolute
            # terms and a tiny probe can round to zero in float32.
            out = d(torch.randn(2, CFG.vis_dim),
                    eef=torch.full((2, CFG.waypoint_dim), 10.0))
            out.abs().sum().backward()
        g = d.hrm.eef_proj.weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0, (
            "eef_proj got no gradient; the HRM's control branch cannot learn"
        )
