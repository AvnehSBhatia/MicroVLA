# Running MicroVLA training on a GPU box (MI300X / ROCm)

The Mac trains at batch=1 (slow). On a real GPU use `train/train_batched.py`,
which batches episodes and hard-caps VRAM. On an MI300X, Stage A goes from
hours to minutes.

## 1. Push the code (from the Mac, once)

```bash
git push -u origin main          # repo is ~1.5 MB, instant
```

## 2. Get the dataset onto the box

The 703 MB dataset is **not** in git (raw binaries don't belong in history).
Move it directly — pick one:

```bash
# Option A — rsync straight into the clone (cleanest, no extra disk):
rsync -avz data/ USER@BOX:~/MicroVLA/data/

# Option B — tar + scp:
tar czf microvla-data.tar.gz data/ && scp microvla-data.tar.gz USER@BOX:~/MicroVLA/
#   then on the box:  tar xzf microvla-data.tar.gz
```

(If you'd rather it be literally push→clone→run with no separate transfer, ask
and I'll commit the dataset — every file is <1 MB so it pushes fine, it just
makes the repo ~700 MB.)

## 3. On the box

```bash
git clone https://github.com/AvnehSBhatia/MicroVLA.git && cd MicroVLA
python -m venv .venv && source .venv/bin/activate

# ROCm PyTorch — match your ROCm version (MI300X needs ROCm >= 6.0):
pip install torch numpy --index-url https://download.pytorch.org/whl/rocm6.2

# (data/ is already here from step 2)
```

Sanity-check the GPU is seen (ROCm exposes AMD GPUs through the torch.cuda API):
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True   AMD Instinct MI300X
```

## 4. Train

```bash
python train/train_batched.py \
    --data-dir data/bridge --data-dir data/libero \
    --device cuda --batch-size 64 --max-vram-gb 50 \
    --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \
    --stage-b-epochs 3
```

- `--max-vram-gb 50` hard-caps the process at 50 GB (of the 192 GB); it will
  OOM inside that cap rather than eating the whole card. The epoch line prints
  `peakVRAM` so you can tune `--batch-size` — with 50 GB you can likely push
  batch-size to 128–256 (the model is tiny; the graph is the cost).
- Same objective, early stopping, and best-checkpoint logic as the Mac run, so
  results are directly comparable (Stage A should reach ~+20% over persistence).
- Outputs: `checkpoints/full_stageA.pt` (best world model) and
  `full_stageB.pt` (+ planner). Copy these back with `norm_stats.json` from the
  dataset dir — the plan outputs are meaningless without the normalizer.

## 5. Ablations (each is one more run, cheap on the MI300X)

```bash
# no evidence-fade (paper E6):  add  --ablate ... (see train_full flags) via train_batched --ablate-grounding
python train/train_batched.py ... --ablate-grounding --tag noground   # E7: frame-only
```

## Notes

- `--device auto` also works on the box (it now prefers cuda/ROCm, then MPS,
  then CPU).
- This box also unblocks the **LIBERO closed-loop eval** (robosuite needs
  Linux — see `eval/SIM_SETUP.md`), so it's worth setting up the sim extra here
  too once training lands.
