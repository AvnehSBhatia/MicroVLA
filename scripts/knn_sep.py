import sys
from pathlib import Path
import numpy as np

def load(dirs, conf_floor=0.02):
    eps = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            if f.stem.endswith("_state"): continue
            with np.load(f) as data:
                if "source_box_embs" not in data: continue
                emb = np.asarray(data["source_box_embs"], dtype=np.float64)
                conf = np.asarray(data["box_weights"], dtype=np.float64)[:, 0]
            k = conf >= conf_floor
            if k.any():
                e = emb[k]; eps.append(e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-8))
    return eps

data = {
    "cream": load(["data/cream_rand"]),
    "butter": load(["data/butter_rand","data/butter_rand2"]),
    "soup": load(["data/teacher_rand_full"]),
}
names = list(data)
allt = np.concatenate([np.concatenate(v) for v in data.values()])
g = allt.mean(0); g /= np.linalg.norm(g)+1e-8
def cen(x):
    y = x - (x @ g)[:, None]*g
    return y/(np.linalg.norm(y,axis=1,keepdims=True)+1e-8)

for K in (1, 5, 15):
    correct = total = 0
    per = {n: [0,0] for n in names}
    for n in names:
        for i, ep in enumerate(data[n]):
            bank, lab = [], []
            for m in names:
                for j, e in enumerate(data[m]):
                    if m == n and j == i: continue
                    bank.append(e); lab += [m]*len(e)
            B = cen(np.concatenate(bank)); L = np.asarray(lab)
            # subsample bank for speed, stratified
            idx = np.arange(len(B))[::3]
            B, L = B[idx], L[idx]
            Q = cen(ep)
            S = Q @ B.T
            top = np.argpartition(-S, K, axis=1)[:, :K]
            votes = L[top]
            pred = np.asarray([max(set(v), key=list(v).count) for v in votes])
            c = int((pred == n).sum()); correct += c; total += len(pred)
            per[n][0] += c; per[n][1] += len(pred)
    print(f"k={K}: leave-episode-out accuracy {correct}/{total} = {correct/total:.3f} "
          + " ".join(f"{n}={per[n][0]/max(1,per[n][1]):.2f}" for n in names))
