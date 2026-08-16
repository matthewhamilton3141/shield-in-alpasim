#!/bin/bash
# Validation + hero: full shielded surround+semantic stack vs unshielded baseline, n=3, on the two
# curated scenes. Confirms (a) a clean scene stays clean under the shield, (b) a crash scene is
# reliably fixed (at-fault -> 0) without stochastic offroad. Shielded arm renders the DEFAULT video.
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=$HOME/shield-data/hf_cache
cd ~/alpasim
N=${N_ROLLOUTS:-3}
RESULTS=~/val_results.csv
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N")
echo "scene,arm,at_fault_mean,offroad_mean,progress_mean,progress_std,status0" > "$RESULTS"
echo "=== VALIDATE start $(date -u) n=$N ==="

agg() {
  python3 - "$1" << "PY" 2>/dev/null || echo ",,,,MISSING"
import json,sys
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
def g(k):
    v=m.get(k); return round(v,3) if isinstance(v,(int,float)) else v
print(",".join(str(x) for x in [g("collision_at_fault"),g("offroad"),g("progress_clipped_rel"),g("progress_clipped_rel_std"),r["status"]]))
PY
}

for S in clipgt-065dcac9-ee67-4434-a835-c6b816c88e48 clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6; do
  echo "======== $S $(date -u +%H:%M:%S) ========"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  PREP=out_val_prep_$S; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam wizard.run_method=NONE \
     wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E "s/.*at //")
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"
  echo "  USDZ -> $CONT"

  OUT=out_val_unsh_$S; rm -rf "$OUT"
  env SHIELD_OBSTACLE_SOURCE=gt \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam eval.video.video_layouts="[]" \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(agg "$OUT/aggregate/results-summary.json"); echo "  unshielded: $M"; echo "$S,unshielded,$M" >> "$RESULTS"

  OUT=out_val_shielded_$S; rm -rf "$OUT"
  env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_SEMANTIC=1 SHIELD_DEPTH_BATCH=1 \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround eval.video.video_layouts="[DEFAULT]" \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(agg "$OUT/aggregate/results-summary.json"); echo "  shielded:   $M"; echo "$S,shielded,$M" >> "$RESULTS"
done
echo "=== VALIDATE DONE $(date -u) ==="
column -t -s, "$RESULTS"
