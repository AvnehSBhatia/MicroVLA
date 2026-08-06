"""Are deployment box embeddings off-manifold relative to the training corpus?

For each object: collect the corpus's stored source_box_embs, and the
deployment embeddings logged during rollouts, then measure how far deployment
sits from training. Two statistics, because either alone is easy to misread:

  cos_to_mean   -- cosine of each deployment emb to the corpus MEAN direction.
                   Sensitive to a systematic shift.
  nn_cos        -- cosine to the NEAREST single corpus emb. Sensitive to
                   'nothing in training looks like this', which a mean can hide
                   when the corpus is multi-modal.

Compared BETWEEN objects (cream vs soup), never against an absolute threshold:
the question is whether cream's deployment is further from its own corpus than
soup's is from its own.
"""
import glob, json, sys
import numpy as np

def corpus_embs(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        if f.endswith("_state.npz"):
            continue
        with np.load(f) as z:
            if "source_box_embs" in z:
                e = np.asarray(z["source_box_embs"], dtype=np.float64)
                w = np.asarray(z["box_weights"], dtype=np.float64) if "box_weights" in z else None
                if w is not None and w.ndim > 1:
                    w = w[:, 0]
                keep = (w > 0) if w is not None else np.ones(len(e), bool)
                out.append(e[keep])
    return np.concatenate(out) if out else np.zeros((0, 512))

def deploy_embs(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("src_box_emb"):
                out.append(r["src_box_emb"])
    return np.asarray(out, dtype=np.float64)

def unit(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)

for name, corp, dep in (("soup", "data/teacher_rand/*.npz", "eval_results/emb_task0/*telemetry.jsonl"),
                        ("cream", "data/cream_rand/*.npz", "eval_results/emb_task1/*telemetry.jsonl"),
                        ("butter", "data/butter_rand/*.npz", "eval_results/emb_task6/*telemetry.jsonl")):
    C = corpus_embs(corp); D = deploy_embs(dep)
    if not len(C) or not len(D):
        print(f"{name}: corpus {len(C)}, deployment {len(D)} -- skipping"); continue
    Cu, Du = unit(C), unit(D)
    mu = unit(C.mean(axis=0, keepdims=True))
    cos_mean = (Du @ mu.T).ravel()
    # nearest-neighbour cosine, deployment -> corpus
    nn = (Du @ Cu.T).max(axis=1)
    # baseline: corpus -> corpus (leave-one-out-ish, via the same NN on a split)
    half = len(Cu) // 2
    nn_self = (Cu[half:] @ Cu[:half].T).max(axis=1) if half > 1 else np.array([np.nan])
    print(f"\n{name}: corpus n={len(C)}, deployment n={len(D)}")
    print(f"  deployment cos to corpus MEAN : {cos_mean.mean():.4f}  (std {cos_mean.std():.4f})")
    print(f"  deployment NN-cos to corpus   : {nn.mean():.4f}  (std {nn.std():.4f})")
    print(f"  corpus->corpus NN-cos baseline: {np.nanmean(nn_self):.4f}   <- how close training is to itself")
    print(f"  GAP (baseline - deployment NN): {np.nanmean(nn_self) - nn.mean():+.4f}")
print("\nEMBDRIFT_DONE", flush=True)
