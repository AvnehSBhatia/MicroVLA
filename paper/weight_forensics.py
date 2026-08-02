"""Static weight forensics over MicroVLA checkpoints.

Reads checkpoints/full_stageB_rec_fix.pt (deployed policy) and
checkpoints/full_stageA_v8_s0.pt (a stage-A world model), computes a
systematic battery of per-tensor and per-module diagnostics, writes every
figure to paper/visuals/ and a machine-generated findings ledger to
paper/forensics_static.json + paper/forensics_static.md.

Pure CPU, deterministic, no model construction — state-dict math only.
Run: .venv/bin/python paper/weight_forensics.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "paper" / "visuals"
VIS.mkdir(parents=True, exist_ok=True)

CKPT_B = ROOT / "checkpoints" / "full_stageB_rec_fix.pt"
CKPT_A = ROOT / "checkpoints" / "full_stageA_v8_s0.pt"

MODULES = ("fusion", "drift", "trm", "planner", "tqsa", "relational")

findings: list[dict] = []


def F(fid: str, module: str, kind: str, text: str, **data):
    findings.append({"id": fid, "module": module, "kind": kind,
                     "text": text, **data})


def tstats(t: torch.Tensor) -> dict:
    x = t.detach().float().reshape(-1).numpy()
    n = x.size
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / (sd + 1e-12)
    return {
        "numel": n, "mean": mu, "std": sd,
        "absmax": float(np.abs(x).max()),
        "sparsity_1e3": float((np.abs(x) < 1e-3).mean()),
        "kurtosis": float((z ** 4).mean() - 3.0),
        "skew": float((z ** 3).mean()),
        "l2": float(np.linalg.norm(x)),
    }


def spectral(w: torch.Tensor) -> dict | None:
    """SVD diagnostics for a 2D weight matrix."""
    if w.dim() != 2 or min(w.shape) < 4:
        return None
    s = torch.linalg.svdvals(w.detach().float()).numpy()
    s = s[s > 1e-12]
    if s.size < 4:
        return None
    p = (s ** 2) / (s ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-16)).sum()))
    stable_rank = float((s ** 2).sum() / (s[0] ** 2))
    # Hill estimator on the top tail of the ESD (eigenvalues of W^T W).
    lam = np.sort(s ** 2)[::-1]
    k = max(4, min(50, lam.size // 4))
    hill = float(1.0 + k / np.log(lam[:k] / lam[k - 1] + 1e-16).sum()) \
        if lam[k - 1] > 0 else float("nan")
    return {
        "shape": list(w.shape), "rank_full": int(min(w.shape)),
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
        "cond": float(s[0] / s[-1]),
        "eff_rank": eff_rank, "stable_rank": stable_rank,
        "eff_rank_frac": eff_rank / min(w.shape),
        "hill_alpha": hill,
        "spectrum": s[:64].tolist(),
    }


def neuron_util(w: torch.Tensor) -> dict | None:
    if w.dim() != 2:
        return None
    rn = w.detach().float().norm(dim=1).numpy()
    med = np.median(rn)
    return {
        "rows": int(w.shape[0]),
        "dead_rows_1pct": int((rn < 0.01 * med).sum()),
        "weak_rows_10pct": int((rn < 0.10 * med).sum()),
        "row_norm_cv": float(rn.std() / (rn.mean() + 1e-12)),
        "row_norm_max_ratio": float(rn.max() / (med + 1e-12)),
    }


def load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    B = load(CKPT_B)
    A = load(CKPT_A)

    # ---------------------------------------------------------------- census
    census = {}
    for m in MODULES:
        sd = B.get(m) or {}
        census[m] = {
            "tensors": len(sd),
            "params": int(sum(t.numel() for t in sd.values())),
        }
    total = sum(v["params"] for v in census.values())
    F("F-001", "all", "census",
      f"Deployed trainable state: {total/1e6:.3f}M parameters across "
      f"{sum(v['tensors'] for v in census.values())} tensors in 6 modules; "
      f"TRM holds {census['trm']['params']/total:.1%} of them.",
      census=census, total=total)

    fig, ax = plt.subplots(figsize=(8, 4))
    ms = list(census)
    ax.bar(ms, [census[m]["params"] / 1e6 for m in ms], color="#4477aa")
    ax.set_ylabel("params (M)")
    ax.set_title("Parameter census by module (full_stageB_rec_fix)")
    for i, m in enumerate(ms):
        ax.text(i, census[m]["params"] / 1e6, str(census[m]["tensors"]) + "t",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(VIS / "census_params.png", dpi=150); plt.close(fig)

    # ------------------------------------------------- per-tensor statistics
    per_tensor = {}
    fid = 2
    for m in MODULES:
        for name, t in (B.get(m) or {}).items():
            st = tstats(t)
            key = f"{m}.{name}"
            per_tensor[key] = st
            if st["sparsity_1e3"] > 0.5 and st["numel"] > 256:
                F(f"F-{fid:03d}", m, "sparsity",
                  f"`{key}` is {st['sparsity_1e3']:.0%} near-zero "
                  f"(|w|<1e-3, n={st['numel']}) — under-used capacity or "
                  f"heavy regularization.", tensor=key, **st); fid += 1
            if abs(st["kurtosis"]) > 20 and st["numel"] > 256:
                F(f"F-{fid:03d}", m, "tails",
                  f"`{key}` kurtosis {st['kurtosis']:.1f} — extreme "
                  f"heavy-tailed weights; a few connections dominate.",
                  tensor=key, **st); fid += 1
            if st["absmax"] > 10 * (st["std"] + 1e-9) and st["numel"] > 256:
                F(f"F-{fid:03d}", m, "outlier",
                  f"`{key}` has |w|max {st['absmax']:.3f} at {st['absmax']/(st['std']+1e-9):.0f}σ "
                  f"— outlier weights that will dominate int8 ranges.",
                  tensor=key, **st); fid += 1

    # std-vs-tensor overview figure per module
    for m in MODULES:
        keys = [k for k in per_tensor if k.startswith(m + ".")]
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(10, max(2, 0.22 * len(keys))))
        stds = [per_tensor[k]["std"] for k in keys]
        ax.barh([k.split(m + ".", 1)[1][:48] for k in keys], stds,
                color="#66ccee")
        ax.set_xlabel("weight std"); ax.set_title(f"{m}: per-tensor std")
        ax.tick_params(axis="y", labelsize=6)
        fig.tight_layout(); fig.savefig(VIS / f"std_{m}.png", dpi=150); plt.close(fig)

    # --------------------------------------------------------- spectral pass
    spec = {}
    for m in MODULES:
        for name, t in (B.get(m) or {}).items():
            if t.dim() == 2 and min(t.shape) >= 8:
                sp = spectral(t)
                if sp:
                    spec[f"{m}.{name}"] = sp
    # findings: lowest effective-rank fractions (rank collapse) and best-conditioned
    ranked = sorted(spec.items(), key=lambda kv: kv[1]["eff_rank_frac"])
    for key, sp in ranked[:12]:
        F(f"F-{fid:03d}", key.split('.')[0], "rank",
          f"`{key}` {sp['shape']} uses an effective rank of "
          f"{sp['eff_rank']:.1f}/{sp['rank_full']} "
          f"({sp['eff_rank_frac']:.0%}) — "
          + ("severe rank collapse; this layer is nearly a low-rank "
             "bottleneck." if sp["eff_rank_frac"] < 0.25 else
             "compressible; a low-rank factorization would keep behavior."),
          tensor=key, **{k: v for k, v in sp.items() if k != "spectrum"})
        fid += 1
    conds = sorted(spec.items(), key=lambda kv: -kv[1]["cond"])[:6]
    for key, sp in conds:
        F(f"F-{fid:03d}", key.split('.')[0], "conditioning",
          f"`{key}` condition number {sp['cond']:.1e} — gradient flow "
          f"through this map is anisotropic by {math.log10(sp['cond']):.1f} "
          f"orders of magnitude.", tensor=key,
          **{k: v for k, v in sp.items() if k != "spectrum"})
        fid += 1
    alphas = {k: v["hill_alpha"] for k, v in spec.items()
              if np.isfinite(v["hill_alpha"])}
    if alphas:
        good = {k: a for k, a in alphas.items() if 2.0 <= a <= 6.0}
        F(f"F-{fid:03d}", "all", "esd",
          f"Heavy-tail (Hill) exponents span {min(alphas.values()):.2f}–"
          f"{max(alphas.values()):.2f}; {len(good)}/{len(alphas)} matrices "
          f"sit in the 2–6 'well-trained' band reported for converged "
          f"networks.", alphas={k: round(a, 2) for k, a in alphas.items()})
        fid += 1

    # spectrum grid figures per module
    for m in MODULES:
        keys = [k for k in spec if k.startswith(m + ".")][:12]
        if not keys:
            continue
        cols = 4
        rows = (len(keys) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows),
                                 squeeze=False)
        for i, k in enumerate(keys):
            ax = axes[i // cols][i % cols]
            s = np.array(spec[k]["spectrum"])
            ax.semilogy(s / s[0], lw=1)
            ax.set_title(k.split(".", 1)[1][:30], fontsize=6)
            ax.tick_params(labelsize=5)
        for j in range(len(keys), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"{m}: normalized singular spectra", fontsize=10)
        fig.tight_layout()
        fig.savefig(VIS / f"spectra_{m}.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------ neuron utilization
    dead_tab = {}
    for m in MODULES:
        for name, t in (B.get(m) or {}).items():
            nu = neuron_util(t)
            if nu and nu["rows"] >= 16:
                dead_tab[f"{m}.{name}"] = nu
    worst = sorted(dead_tab.items(), key=lambda kv: -kv[1]["weak_rows_10pct"])[:10]
    for key, nu in worst:
        if nu["weak_rows_10pct"] > 0:
            F(f"F-{fid:03d}", key.split('.')[0], "neurons",
              f"`{key}`: {nu['dead_rows_1pct']} dead and "
              f"{nu['weak_rows_10pct']}/{nu['rows']} weak output rows "
              f"(<10% median norm); row-norm CV {nu['row_norm_cv']:.2f}.",
              tensor=key, **nu)
            fid += 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ks = [k for k, _ in worst]
    ax.bar(range(len(ks)), [dead_tab[k]["weak_rows_10pct"] for k in ks],
           color="#ee6677")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k[:28] for k in ks], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("# weak rows (<10% median)")
    ax.set_title("Neuron under-utilization (top offenders)")
    fig.tight_layout(); fig.savefig(VIS / "dead_neurons.png", dpi=150); plt.close(fig)

    # ------------------------------------------- stage-A ↔ stage-B comparison
    deltas = {}
    for m in MODULES:
        a_sd, b_sd = A.get(m), B.get(m)
        if not a_sd or not b_sd:
            continue
        common = [k for k in b_sd if k in a_sd and a_sd[k].shape == b_sd[k].shape]
        if not common:
            continue
        rel = {}
        for k in common:
            x = a_sd[k].detach().float(); y = b_sd[k].detach().float()
            rel[k] = float((y - x).norm() / (x.norm() + 1e-12))
        deltas[m] = rel
    module_mean = {m: float(np.mean(list(r.values()))) for m, r in deltas.items()}
    F(f"F-{fid:03d}", "all", "lineage",
      "Cross-checkpoint relative Frobenius distance (rec_fix vs v8_s0 stage "
      "A), mean per module: "
      + ", ".join(f"{m}={v:.3f}" for m, v in module_mean.items())
      + ". Near-zero means shared lineage/frozen; O(1) means independently "
        "trained tensors.", module_mean=module_mean)
    fid += 1
    if deltas:
        fig, ax = plt.subplots(figsize=(8, 4))
        ms = list(module_mean)
        ax.bar(ms, [module_mean[m] for m in ms], color="#228833")
        ax.set_ylabel("mean relative ΔFro vs stage-A ckpt")
        ax.set_title("Which modules moved (rec_fix vs v8_s0)")
        fig.tight_layout(); fig.savefig(VIS / "stage_delta.png", dpi=150); plt.close(fig)

    # ----------------------------------------------------- module fingerprints
    # Radar-ish summary per module: std, eff-rank-frac, sparsity, kurtosis.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for i, m in enumerate(MODULES):
        keys = [k for k in per_tensor if k.startswith(m + ".")]
        sk = [k for k in spec if k.startswith(m + ".")]
        ax = axes[i // 3][i % 3]
        vals = [
            np.mean([per_tensor[k]["std"] for k in keys]) if keys else 0,
            np.mean([spec[k]["eff_rank_frac"] for k in sk]) if sk else 0,
            np.mean([per_tensor[k]["sparsity_1e3"] for k in keys]) if keys else 0,
            np.tanh(np.mean([per_tensor[k]["kurtosis"] for k in keys]) / 20) if keys else 0,
        ]
        ax.bar(["std", "effRank", "sparse", "kurt~"], vals,
               color=["#4477aa", "#228833", "#ccbb44", "#ee6677"])
        ax.set_title(m); ax.set_ylim(0, 1.0)
    fig.suptitle("Module fingerprints (mean per-tensor diagnostics)")
    fig.tight_layout(); fig.savefig(VIS / "module_fingerprints.png", dpi=150); plt.close(fig)

    # --------------------------------------------------- quantization readiness
    qerr = {}
    for m in MODULES:
        for name, t in (B.get(m) or {}).items():
            if t.numel() < 256 or t.dim() < 2:
                continue
            x = t.detach().float()
            scale = x.abs().max() / 127.0
            q = torch.clamp((x / scale).round(), -127, 127) * scale
            qerr[f"{m}.{name}"] = float(((q - x).norm() / (x.norm() + 1e-12)))
    worst_q = sorted(qerr.items(), key=lambda kv: -kv[1])[:8]
    F(f"F-{fid:03d}", "all", "quant",
      "Simulated per-tensor symmetric int8: median relative error "
      f"{np.nanmedian([v for v in qerr.values() if np.isfinite(v)]):.4f}; worst layers: "
      + ", ".join(f"{k}={v:.3f}" for k, v in worst_q[:4])
      + ". Outlier-dominated ranges (see outlier findings) are the "
        "Pi-deployment quantization risk.", worst=dict(worst_q))
    fid += 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ks = [k for k, _ in worst_q]
    ax.bar(range(len(ks)), [qerr[k] for k in ks], color="#aa3377")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k[:28] for k in ks], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("relative int8 error")
    ax.set_title("Quantization-sensitive layers (symmetric per-tensor int8)")
    fig.tight_layout(); fig.savefig(VIS / "quant_error.png", dpi=150); plt.close(fig)

    # --------------------------------------------------------------- heatmaps
    for key in ["planner", "trm", "relational"]:
        sd = B.get(key) or {}
        two_d = [(n, t) for n, t in sd.items() if t.dim() == 2 and t.numel() > 4096]
        two_d.sort(key=lambda kv: -kv[1].numel())
        if two_d:
            n, t = two_d[0]
            w = t.detach().float().numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            v = np.percentile(np.abs(w), 99)
            im = ax.imshow(w, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{key}.{n} {list(t.shape)}", fontsize=8)
            fig.tight_layout()
            fig.savefig(VIS / f"heatmap_{key}.png", dpi=150); plt.close(fig)

    # ----------------------------------------------------------------- output
    out = {
        "census": census, "per_tensor": per_tensor,
        "spectral": {k: {kk: vv for kk, vv in v.items() if kk != "spectrum"}
                     for k, v in spec.items()},
        "neurons": dead_tab, "stage_delta": deltas, "quant_rel_err": qerr,
        "findings": findings,
    }
    (ROOT / "paper" / "forensics_static.json").write_text(
        json.dumps(out, indent=1))

    lines = ["# Static weight forensics — machine-generated ledger", ""]
    for f_ in findings:
        lines.append(f"- **{f_['id']}** [{f_['module']}/{f_['kind']}] {f_['text']}")
    (ROOT / "paper" / "forensics_static.md").write_text("\n".join(lines) + "\n")
    print(f"findings: {len(findings)}  figures: {len(list(VIS.glob('*.png')))}")
    print("tensors:", len(per_tensor), " spectra:", len(spec))


if __name__ == "__main__":
    main()
