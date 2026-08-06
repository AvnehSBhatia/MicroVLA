"""Is the appearance drift a CAUSE or a CONSEQUENCE of the trajectory error?

A machine that servos to a wrong xy necessarily sees the object from unusual
viewpoints, so late-episode drift proves nothing -- the same circularity that
invalidated the uv measurement. Early ticks are different: at episode start the
arm is at its home pose and has not yet acted on any bad prediction, so drift
present THERE cannot have been produced by the error.

Splits deployment embeddings by tick_index and reports the gap in each band.
"""
import glob, json
import numpy as np

def corpus_embs(pat):
    out=[]
    for f in sorted(glob.glob(pat)):
        if f.endswith("_state.npz"): continue
        with np.load(f) as z:
            if "source_box_embs" in z:
                e=np.asarray(z["source_box_embs"],dtype=np.float64)
                w=np.asarray(z["box_weights"],dtype=np.float64) if "box_weights" in z else None
                if w is not None and w.ndim>1: w=w[:,0]
                out.append(e[(w>0) if w is not None else np.ones(len(e),bool)])
    return np.concatenate(out) if out else np.zeros((0,512))

def unit(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-9)

for name, corp, dep in (("soup","data/teacher_rand/*.npz","eval_results/emb_task0/*telemetry.jsonl"),
                        ("butter","data/butter_rand/*.npz","eval_results/emb_task6/*telemetry.jsonl"),
                        ("cream","data/cream_rand/*.npz","eval_results/emb_task1/*telemetry.jsonl")):
    C=corpus_embs(corp); Cu=unit(C)
    half=len(Cu)//2
    base=float((Cu[half:]@Cu[:half].T).max(axis=1).mean())
    rows=[]
    for f in sorted(glob.glob(dep)):
        for l in open(f):
            if not l.strip(): continue
            r=json.loads(l)
            if r.get("src_box_emb"): rows.append((int(r.get("tick_index",0)), r["src_box_emb"]))
    if not rows: print(f"{name}: none"); continue
    print(f"\n{name}: corpus self-NN baseline {base:.4f}")
    for lo,hi,lbl in ((0,100,"ticks 0-100 (BEFORE error accumulates)"),
                      (100,300,"ticks 100-300"),
                      (300,10**9,"ticks 300+ (after)")):
        sel=[e for t,e in rows if lo<=t<hi]
        if not sel: 
            print(f"   {lbl:<38s} n=0"); continue
        D=unit(np.asarray(sel,dtype=np.float64))
        nn=float((D@Cu.T).max(axis=1).mean())
        print(f"   {lbl:<38s} n={len(sel):3d}  NN-cos {nn:.4f}  gap {base-nn:+.4f}")
print("\nEMBEARLY_DONE", flush=True)
