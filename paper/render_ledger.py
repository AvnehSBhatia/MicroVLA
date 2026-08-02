"""Render the exhaustive per-tensor findings ledger from forensics JSON.

Every tensor contributes one census finding; every 2D matrix adds a spectral
finding; every analyzed layer adds neuron-utilization, quantization, and
stage-delta findings where measured. Output: paper/forensics_ledger.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / "forensics_static.json").read_text())
dyn = json.loads((ROOT / "forensics_dynamic.json").read_text())

lines = ["# Exhaustive weight-forensics ledger (machine-generated)",
         "",
         "One numbered finding per tensor per instrument. Source of truth: "
         "`forensics_static.json` / `forensics_dynamic.json`; figures in "
         "`visuals/`.", ""]
n = 0

lines.append("## L1. Per-tensor census (mean / std / |max| / near-zero / kurtosis)")
for k, s in d["per_tensor"].items():
    n += 1
    lines.append(
        f"- **L-{n:03d}** `{k}` [{s['numel']}]: mean {s['mean']:+.4f}, "
        f"std {s['std']:.4f}, |max| {s['absmax']:.3f}, near-zero "
        f"{s['sparsity_1e3']:.1%}, kurtosis {s['kurtosis']:+.1f}, "
        f"skew {s['skew']:+.2f}")

lines.append("")
lines.append("## L2. Spectral diagnostics (effective rank / stable rank / conditioning / tail)")
for k, s in d["spectral"].items():
    n += 1
    lines.append(
        f"- **L-{n:03d}** `{k}` {s['shape']}: σmax {s['sigma_max']:.3f}, "
        f"eff-rank {s['eff_rank']:.1f}/{s['rank_full']} "
        f"({s['eff_rank_frac']:.0%}), stable-rank {s['stable_rank']:.1f}, "
        f"cond {s['cond']:.1e}, Hill α {s['hill_alpha']:.2f}")

lines.append("")
lines.append("## L3. Neuron utilization (output-row norms)")
for k, s in d["neurons"].items():
    n += 1
    lines.append(
        f"- **L-{n:03d}** `{k}`: {s['rows']} rows, dead {s['dead_rows_1pct']}, "
        f"weak {s['weak_rows_10pct']}, norm CV {s['row_norm_cv']:.2f}, "
        f"max/median {s['row_norm_max_ratio']:.2f}")

lines.append("")
lines.append("## L4. Symmetric int8 quantization error (relative Frobenius)")
for k, v in sorted(d["quant_rel_err"].items(), key=lambda kv: -kv[1]):
    n += 1
    lines.append(f"- **L-{n:03d}** `{k}`: {v:.4f}")

lines.append("")
lines.append("## L5. Cross-checkpoint deltas (rec_fix vs v8_s0, relative Frobenius)")
for m, rel in d["stage_delta"].items():
    for k, v in rel.items():
        n += 1
        lines.append(f"- **L-{n:03d}** `{m}.{k}`: Δ {v:.3f}")

lines.append("")
lines.append("## L6. Curated findings (static)")
for f in d["findings"]:
    n += 1
    lines.append(f"- **L-{n:03d}** ({f['id']}) {f['text']}")
lines.append("")
lines.append("## L7. Curated findings (dynamic)")
for f in dyn["findings"]:
    n += 1
    lines.append(f"- **L-{n:03d}** ({f['id']}) {f['text']}")

(ROOT / "forensics_ledger.md").write_text("\n".join(lines) + "\n")
print("ledger entries:", n)
