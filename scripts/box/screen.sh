#!/bin/bash
# Tier 1 pre-screen: score each candidate scene by unshielded VaVAM on-route behavior (front cam,
# n=1, inert shield -- no SHIELD_SCENE_USDZ). High progress + offroad=0 => the ego stays in the
# reconstructed corridor => a clean scene worth keeping for the degradation sweep + hero video.
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=$HOME/shield-data/hf_cache
cd ~/alpasim
RESULTS=~/screen_results.csv
echo "scene,collision_at_fault,offroad,progress,dist_m,gt_dist_m,status" > "$RESULTS"
echo "=== SCREEN start $(date -u) ; $(df -h / | tail -1) ==="
i=0
while read -r S; do
  [ -z "$S" ] && continue
  i=$((i+1))
  echo "---- [$i/20] $S $(date -u +%H:%M:%S) ----"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  OUT=out_screen_$i; rm -rf "$OUT"
  env SHIELD_OBSTACLE_SOURCE=gt \
    uv run alpasim_wizard deploy=local topology=1gpu driver=shielded_vavam \
    runtime.simulation_config.n_rollouts=1 eval.video.video_layouts="[]" \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(python3 - "$OUT/aggregate/results-summary.json" << "PY" 2>/dev/null || echo ",,,,,MISSING"
import json,sys
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
print(",".join(str(x) for x in [m["collision_at_fault"],m["offroad"],round(m["progress_clipped_rel"],3),round(m["dist_traveled_m"],1),round(m["gt_dist_traveled_m"],1),r["status"]]))
PY
)
  echo "$S,$M" >> "$RESULTS"
  echo "   -> $M"
done < ~/screen_scenes.txt
echo "=== SCREEN DONE $(date -u) ==="
column -t -s, "$RESULTS"
