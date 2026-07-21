#!/bin/zsh
# Overnight conversion driver: LIBERO (30 hdf5 shards) then BridgeData V2
# (128 RLDS shards), sequentially, each under the shared disk budget.
# Raw shards are streamed: downloaded, converted, deleted. Logs to logs/.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs data/bridge/_rlds_meta

PY=.venv/bin/python
BUDGET=6   # GB tracked by the pipeline (out dirs + workdirs); total repo
           # footprint incl. venv (~2.3 GB) stays under the 10 GB cap.

echo "=== LIBERO (object+spatial+goal) ==="
$PY -m preprocess.shard_pipeline libero_shards.txt data/libero \
    --dataset libero --budget-gb $BUDGET --workdir .shard_tmp_libero \
    --device mps 2>&1 | tee logs/convert_libero.log | grep -E "INFO shard|shard done|finalized" || exit 1

echo "=== BridgeData V2 (128 RLDS shards) ==="
$PY -m preprocess.shard_pipeline bridge_shards.txt data/bridge \
    --dataset bridge_rlds --rlds-meta data/bridge/_rlds_meta \
    --budget-gb $BUDGET --workdir .shard_tmp_bridge \
    --device mps 2>&1 | tee logs/convert_bridge.log | grep -E "INFO shard|shard done|finalized" || exit 1

echo "=== conversion complete ==="
du -sh data/libero data/bridge
$PY - <<'EOF'
import json
for name in ("data/libero", "data/bridge"):
    m = json.load(open(f"{name}/manifest.json"))
    eps = m["episodes"]
    print(f"{name}: {len(eps)} episodes, {sum(e['T'] for e in eps)} real frames")
EOF
