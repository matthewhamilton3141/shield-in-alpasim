#!/bin/bash
# Tier 1 degradation curve: for each curated scene, shielded_vavam_surround with GT obstacles vs
# camera(surround+semantic) obstacles, n=5. Same driver/policy, only the obstacle source differs ->
# isolates how much the certificate degrades under learned perception. No video (fast).
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=$HOME/shield-data/hf_cache
cd ~/alpasim
N=${N_ROLLOUTS:-5}
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N" eval.video.video_layouts="[]")
RESULTS=~/tier1_results.csv
echo "scene,arm,at_fault,at_fault_std,offroad,progress,progress_std,mean_obstacles,status0" > "$RESULTS"
echo "=== TIER1 start $(date -u) n=$N ==="

agg() {
  python3 - "$1" "$2" << "PY" 2>/dev/null || echo ",,,,,,MISSING"
import json,sys,re,glob,statistics as st
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
def g(k):
    v=m.get(k); return round(v,3) if isinstance(v,(int,float)) else v
mo=""
t=open(sys.argv[2]).read() if glob.glob(sys.argv[2]) else ""
nob=[int(x) for x in re.findall(r"n_obstacles.: (\d+)",t)]
mo=round(st.mean(nob),1) if nob else ""
print(",".join(str(x) for x in [g("collision_at_fault"),g("collision_at_fault_std"),g("offroad"),g("progress_clipped_rel"),g("progress_clipped_rel_std"),mo,r["status"]]))
PY
}

i=0
while read -r S; do
  [ -z "$S" ] && continue
  i=$((i+1))
  echo "======== [$i/10] $S $(date -u +%H:%M:%S) ========"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  PREP=out_t1_prep_$i; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam wizard.run_method=NONE \
     wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E "s/.*at //")
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! no USDZ for $S; skip"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"

  # GT obstacle arm
  OUT=out_t1_gt_$i; rm -rf "$OUT"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=gt \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(agg "$OUT/aggregate/results-summary.json" "$OUT.log"); echo "$S,gt,$M" >> "$RESULTS"; echo "  gt:     $M"

  # camera(surround+semantic) obstacle arm
  OUT=out_t1_cam_$i; rm -rf "$OUT"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=camera SHIELD_SEMANTIC=1 SHIELD_DEPTH_BATCH=1 \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(agg "$OUT/aggregate/results-summary.json" "$OUT.log"); echo "$S,camera,$M" >> "$RESULTS"; echo "  camera: $M"
done < ~/tier1_scenes.txt
echo "=== TIER1 DONE $(date -u) ==="
column -t -s, "$RESULTS"
