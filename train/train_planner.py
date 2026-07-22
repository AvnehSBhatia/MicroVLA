"""Behavior-cloning training scaffold for the MicroVLA v2 trainable heads.

Runs a few CPU epochs of behavior cloning on synthetic episodes through the
full differentiable path:

    SlotResonanceFusion -> [TRM SLOT: MockTRM stand-in] -> ChronoQueryPlanner

with the AnchoredDriftEncoder providing the state-delta side input. Only the
trainable heads (fusion + drift + planner) are optimized; the MockTRM merely
lets gradients flow through the slot the real ~10M-param TRM will occupy.

``--modality-dropout`` controls ``cfg.modality_dropout`` (the per-sample
Bernoulli probability, inside ``SlotResonanceFusion``, of zeroing the
box/geometry tokens during training) so the dream-mode code path — the SAME
path the JEPA loop's dream ticks use at inference — is exercised while
training the heads.

Usage::

    python train/train_planner.py [--epochs 3] [--episodes 4] [--modality-dropout 0.3]

Checkpoints are written to ``./checkpoints/`` by default.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

# Allow `python train/train_planner.py` from anywhere without installing the
# package: put the repo root (parent of this file's directory) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder  # noqa: E402
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig  # noqa: E402
from microvla.fusion.slot_fusion import SlotResonanceFusion  # noqa: E402
from microvla.planner.chrono_planner import ChronoQueryPlanner  # noqa: E402
from microvla.trm.mock_trm import MockTRM  # noqa: E402
from train.dataset import EpisodeDataset, make_synthetic_episode, save_episode  # noqa: E402
from train.losses import planner_bc_loss, smoothness_loss, total_planner_loss  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the training scaffold."""
    parser = argparse.ArgumentParser(
        description="Behavior-cloning smoke training for MicroVLA v2 heads (CPU)."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--episodes", type=int, default=4, help="Synthetic episodes.")
    parser.add_argument("--episode-len", type=int, default=16, help="Timesteps per episode.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--smooth-weight", type=float, default=0.1, help="Smoothness loss weight."
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic-data base seed.")
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=DEFAULT_CONFIG.modality_dropout,
        help=(
            "Per-sample Bernoulli probability (inside SlotResonanceFusion) of "
            "zeroing the box/geometry tokens during training -- the same code "
            "path JEPA dream ticks use at inference, so this exercises dream "
            "mode while training the heads."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory of .npz episodes; if omitted, synthetic episodes are generated.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' picks Apple-silicon MPS when available, else CPU; or "
        "pass an explicit torch device string ('cpu', 'mps', 'cuda').",
    )
    return parser.parse_args(argv)


def resolve_device(spec: str) -> torch.device:
    """Maps the --device flag to a torch.device.

    Args:
        spec: 'auto' (MPS if available, else CPU) or an explicit device.

    Returns:
        The resolved device.
    """
    if spec != "auto":
        return torch.device(spec)
    # ROCm PyTorch presents AMD GPUs (e.g. MI300X) through the torch.cuda API,
    # so this branch covers both NVIDIA CUDA and AMD ROCm.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(
    episode: dict[str, torch.Tensor],
    fusion: SlotResonanceFusion,
    drift: AnchoredDriftEncoder,
    trm: MockTRM,
    planner: ChronoQueryPlanner,
) -> torch.Tensor:
    """Runs one episode through the full differentiable path.

    The drift encoder is stateful and sequential, so the episode is processed
    timestep by timestep with batch size 1 (matching runtime).

    Args:
        episode: Episode tensors keyed per ``train.dataset.EPISODE_KEYS``.
        fusion: Slot Resonance Fusion head.
        drift: Anchored Drift Encoder (reset at the start of the episode).
        trm: TRM-slot module (MockTRM stand-in; gradients flow through it).
        planner: Chrono-Query Planner head.

    Returns:
        Predicted plans ``[T, plan_steps, num_servos]`` (with grad).
    """
    text_tokens = episode["text_tokens"].unsqueeze(0)  # [1, 3, text_dim]
    T = episode["frame_embs"].shape[0]

    drift.reset()  # new episode: next frame becomes the drift anchor
    preds = []
    for t in range(T):
        frame_emb = episode["frame_embs"][t].unsqueeze(0)  # [1, vis_dim]
        source_box_emb = episode["source_box_embs"][t].unsqueeze(0)  # [1, vis_dim]
        target_box_emb = episode["target_box_embs"][t].unsqueeze(0)  # [1, vis_dim]
        source_center = episode["source_centers"][t].unsqueeze(0)  # [1, 2]
        target_center = episode["target_centers"][t].unsqueeze(0)  # [1, 2]
        box_weight = episode["box_weights"][t].unsqueeze(0)  # [1, 2]

        # Previously EXECUTED action (teacher-forced): row 0 of the previous
        # timestep's target plan; zeros at episode start. This mirrors the
        # JEPA loop, where fusion's action token carries plan[0] of the last
        # emitted plan.
        if t == 0:
            last_action = episode["pwm_targets"].new_zeros(1, episode["pwm_targets"].shape[-1])
        else:
            last_action = episode["pwm_targets"][t - 1, 0].unsqueeze(0)

        # Grounded (real-perception) training data; the dream regime is still
        # exercised via fusion's own train-time evidence fade
        # (cfg.modality_dropout), the SAME weighting path JEPA dream ticks
        # use at inference with staleness-decayed weights.
        fused = fusion(
            text_tokens,
            frame_emb,
            source_box_emb,
            target_box_emb,
            source_center,
            target_center,
            box_weight=box_weight,
            last_action=last_action,
        )  # [1, 32, 5]
        state_delta = drift(frame_emb)  # [1, 256]
        # ------------------------------------------------------------------
        # TRM SLOT: MockTRM stands in for the real ~10M-param TRM
        # (TRM.py::RecursiveTRM). Swap in any TRMBase implementation here;
        # nothing else changes. NOTE the training order that actually works:
        # 1) train the TRM self-supervised on unlabeled video first
        #    (TRM_SPEC.md rollout training), THEN 2) train fusion + planner
        #    jointly with the trained TRM in this slot. Heads trained through
        #    the frozen random MockTRM below are scaffolding only — their
        #    weights will NOT transfer to a real TRM.
        # ------------------------------------------------------------------
        next_emb = trm(fused, state_delta, frame_emb)  # [1, 512]
        plan = planner(next_emb)  # [1, plan_steps, num_servos]
        preds.append(plan.squeeze(0))
    return torch.stack(preds, dim=0)  # [T, plan_steps, num_servos]


def main(argv: list[str] | None = None) -> None:
    """Trains the heads for a few epochs and saves checkpoints."""
    args = parse_args(argv)
    cfg: MicroVLAConfig = dataclasses.replace(
        DEFAULT_CONFIG, modality_dropout=args.modality_dropout
    )
    device = resolve_device(args.device)
    print(f"training on device: {device}")

    # Data: use the provided directory, or generate synthetic episodes.
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(tempfile.mkdtemp(prefix="microvla_synthetic_"))
        for i in range(args.episodes):
            episode = make_synthetic_episode(args.episode_len, cfg, seed=args.seed + i)
            save_episode(data_dir / f"episode_{i:03d}.npz", episode)
        print(f"Generated {args.episodes} synthetic episodes in {data_dir}")
    dataset = EpisodeDataset(data_dir)

    # Trainable heads.
    fusion = SlotResonanceFusion(cfg).to(device)
    drift = AnchoredDriftEncoder(cfg).to(device)
    planner = ChronoQueryPlanner(cfg).to(device)

    # ----------------------------------------------------------------------
    # TRM SLOT: the real ~10M-param TRM (see microvla/trm/TRM_SPEC.md) is NOT
    # built or trained in this repo. MockTRM is a frozen pass-through stub so
    # the fusion->TRM->planner path is end-to-end differentiable today.
    # ----------------------------------------------------------------------
    trm = MockTRM(cfg).to(device)
    for p in trm.parameters():
        p.requires_grad_(False)

    fusion.train()
    drift.train()
    planner.train()

    optimizer = torch.optim.Adam(
        [*fusion.parameters(), *drift.parameters(), *planner.parameters()],
        lr=args.lr,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_bc = 0.0
        epoch_smooth = 0.0
        epoch_total = 0.0
        for idx in range(len(dataset)):
            episode = {k: v.to(device) for k, v in dataset[idx].items()}
            preds = run_episode(episode, fusion, drift, trm, planner)
            targets = episode["pwm_targets"]

            loss = total_planner_loss(preds, targets, smooth_weight=args.smooth_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_bc += planner_bc_loss(preds, targets).item()
                epoch_smooth += smoothness_loss(preds).item()
                epoch_total += loss.item()

        n = len(dataset)
        print(
            f"epoch {epoch:>3d} | total {epoch_total / n:.4f} | "
            f"bc {epoch_bc / n:.4f} | smooth {epoch_smooth / n:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "cfg": dataclasses.asdict(cfg),
            "fusion": fusion.state_dict(),
            "drift": drift.state_dict(),
            "planner": planner.state_dict(),
        }
        path = checkpoint_dir / f"heads_epoch{epoch:03d}.pt"
        torch.save(checkpoint, path)
        print(f"saved checkpoint -> {path}")


if __name__ == "__main__":
    main()
