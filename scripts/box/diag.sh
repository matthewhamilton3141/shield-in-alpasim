#!/bin/bash
# Diagnostic A/B: same driver (shielded_vavam_surround, armed), only the shield obstacle source
# differs -- GT actors vs camera(surround+semantic). Isolates perception-induced over-braking from
# genuinely-required braking. Camera arm dumps per-cycle BEV (perceived discs vs real GT discs) +
# fisheye frames. Both render the DEFAULT video (GT arm = clean hero candidate).
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=$HOME/shield-data/hf_cache
cd ~/alpasim
N=${N_ROLLOUTS:-3}
S=clipgt-065dcac9-ee67-4434-a835-c6b816c88e48
CONT=/mnt/nre-data/all-usdzs/017e570f-e223-441e-b85e-36b2819cd8d2.usdz
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N")
RESULTS=~/diag_results.csv
echo "arm,at_fault,offroad,progress,progress_std,status0,mean_obstacles,intv_cycles" > "$RESULTS"
echo "=== DIAG start $(date -u) n=$N ==="

report() {  # $1=arm $2=out_dir
  A=$(python3 - "$2/aggregate/results-summary.json" << "PY" 2>/dev/null || echo ",,,,MISSING"
import json,sys
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
def g(k):
    v=m.get(k); return round(v,3) if isinstance(v,(int,float)) else v
print(",".join(str(x) for x in [g("collision_at_fault"),g("offroad"),g("progress_clipped_rel"),g("progress_clipped_rel_std"),r["status"]]))
PY
)
  O=$(python3 - "$2.log" << "PY" 2>/dev/null || echo ","
import re,sys,statistics as st
t=open(sys.argv[1]).read()
nob=[int(x) for x in re.findall(r"n_obstacles.: (\d+)",t)]
nv=[int(x) for x in re.findall(r"n_interventions.: (\d+)",t)]
mo=round(st.mean(nob),1) if nob else 0
ic=f"{sum(1 for x in nv if x>0)}/{len(nv)}" if nv else "0/0"
print(f"{mo},{ic}")
PY
)
  echo "$1,$A,$O" >> "$RESULTS"; echo "  $1: $A | obs/intv $O"
}

# --- GT obstacle arm (privileged actors) ---
OUT=out_diag_gt; rm -rf "$OUT"
docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
echo "---- GT arm $(date -u +%H:%M:%S) ----"
env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=gt \
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround eval.video.video_layouts="[DEFAULT]" \
  wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
report GT "$OUT"

# --- camera obstacle arm (surround + semantic) with BEV debug ---
OUT=out_diag_cam; rm -rf "$OUT"
docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
echo "---- CAMERA arm $(date -u +%H:%M:%S) ----"
env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=camera SHIELD_SEMANTIC=1 SHIELD_DEPTH_BATCH=1 \
    SHIELD_DEBUG_DIR=/mnt/output/shield_debug \
    SHIELD_DEBUG_CAMERAS=camera_front_wide_120fov,camera_cross_left_120fov,camera_rear_left_70fov \
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround eval.video.video_layouts="[DEFAULT]" \
  wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
report CAMERA "$OUT"

echo "=== DIAG DONE $(date -u) ==="
column -t -s, "$RESULTS"
