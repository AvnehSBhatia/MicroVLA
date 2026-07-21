#!/bin/zsh
# Overnight conversion driver: LIBERO (30 hdf5 shards) then BridgeData V2
# (128 RLDS shards), sequentially, each under the shared disk budget.
# Raw shards are streamed: downloaded, converted, deleted. Logs to logs/.
# Per-shard failures are skipped and journaled by the pipeline itself
# (resume = re-run this script; completed shards are never redone).
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/bridge/_rlds_meta

PY=.venv/bin/python
BUDGET=6   # GB tracked by the pipeline (out dirs + workdirs); total repo
           # footprint incl. venv (~2.3 GB) stays under the 10 GB cap.
STATUS=0

echo "=== LIBERO (object+spatial+goal) ==="
if ! $PY -m preprocess.shard_pipeline libero_shards.txt data/libero \
    --dataset libero --budget-gb $BUDGET --workdir .shard_tmp_libero \
    --device mps 2>&1 | tee logs/convert_libero.log | grep -E "INFO shard|shard done|FAILED|finalized|resuming"; then
  echo "LIBERO pipeline exited nonzero (see logs/convert_libero.log)"; STATUS=1
fi

echo "=== BridgeData V2 (128 RLDS shards) ==="
if ! $PY -m preprocess.shard_pipeline bridge_shards.txt data/bridge \
    --dataset bridge_rlds --rlds-meta data/bridge/_rlds_meta \
    --budget-gb $BUDGET --workdir .shard_tmp_bridge \
    --device mps 2>&1 | tee logs/convert_bridge.log | grep -E "INFO shard|shard done|FAILED|finalized|resuming"; then
  echo "Bridge pipeline exited nonzero (see logs/convert_bridge.log)"; STATUS=1
fi

echo "=== conversion summary ==="
du -sh data/libero data/bridge 2>/dev/null
$PY - <<'EOF'
import json, pathlib
for name in ("data/libero", "data/bridge"):
    m = pathlib.Path(name) / "manifest.json"
    if m.exists():
        eps = json.loads(m.read_text())["episodes"]
        print(f"{name}: {len(eps)} episodes, {sum(e['T'] for e in eps)} real frames")
    else:
        print(f"{name}: NOT finalized (re-run scripts/convert_all.sh to resume)")
EOF
exit $STATUS
