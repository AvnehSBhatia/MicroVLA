"""Does _sim_object_pos actually return the TASK'S TARGET object?

It picks "the first non-basket-looking body" from obj_body_id. That is an
assumption about dict ordering, not a lookup of the task's target. Every
eef_obj_dist_* number in the paper depends on it. Verify, per task, against
the BDDL's declared target.
"""
import sys
sys.path.insert(0, "/root/MicroVLA")
from eval._libero_compat import prepare_libero
prepare_libero()
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark as lb

bench = lb.get_benchmark_dict()["libero_object"]()
for tid in (0, 1, 6):
    task = bench.get_task(tid)
    bddl = bench.get_task_bddl_file_path(tid)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64,
                             camera_widths=64, camera_names=["robot0_eye_in_hand"])
    env.reset()
    inner = getattr(env, "env", env)
    body_ids = getattr(inner, "obj_body_id", None) or {}
    names = list(body_ids.keys())
    pick = next((n for n in names
                 if not any(t in n.lower() for t in ("basket", "bin", "plate", "tray"))),
                names[0] if names else None)
    # the task's declared target, from the task name
    lang = getattr(task, "language", "")
    print(f"\ntask {tid}: {lang}")
    print(f"  obj_body_id order : {names}")
    print(f"  _sim_object_pos picks: {pick}")
    tgt = getattr(task, "name", "")
    print(f"  task name          : {tgt}")
    ok = pick and pick.split("_1")[0].replace("_", " ") in lang.lower()
    print(f"  ==> picked body matches the commanded noun? {'YES' if ok else '*** NO ***'}")
    env.close()
print("\nVERIFY_DONE", flush=True)
