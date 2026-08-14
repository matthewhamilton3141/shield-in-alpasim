#!/bin/bash
# A/B the rear-obstacle filter (finding-2 fix): shielded VaVAM with SHIELD_REAR_FILTER 0 vs 1,
# one scene at a time, both arms armed with the scene's ground-truth geometry. Both arms are
# shielded, so this isolates the filter's effect on progress/collisions (VaVAM's run-to-run
# stochasticity hits both arms). Scenes must already be downloaded (run scene_sweep.sh first).
#
# Run on the box:  bash ~/shield-in-alpasim/scripts/shield_rear_ab.sh clipgt-... clipgt-...
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd ~/alpasim
RESULTS=~/rear_ab_results.csv
COMMON=(deploy=local topology=1gpu runtime.simulation_config.n_rollouts=1 eval.video.video_layouts='[]')
echo "scene,rear_filter,collision_at_fault,collision_rear,offroad,dist_m,gt_dist_m,progress,score,status" > "$RESULTS"

metric() {
  python3 - "$1" <<'PY' 2>/dev/null || echo ",,,,,,,MISSING"
import json,sys
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
print(",".join(str(x) for x in [m["collision_at_fault"],m["collision_rear"],m["offroad"],
      round(m["dist_traveled_m"],1),round(m["gt_dist_traveled_m"],1),
      round(m["progress_clipped_rel"],3),round(r["score"],3),r["status"]]))
PY
}

for S in "$@"; do
  echo "======== SCENE $S ========"
  # Cheap config-gen (no GPU) to (re)build the sceneset and resolve this scene's USDZ.
  PREP=out_ab_prep_$S; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam wizard.run_method=NONE \
      wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E 's/.*at //')
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! could not resolve USDZ for $S; skipping"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"
  echo "  USDZ -> $CONT"

  for F in 0 1; do
    OUT=out_ab_f${F}_$S; rm -rf "$OUT"
    SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_REAR_FILTER="$F" \
      uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam \
      wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
    M=$(metric "$OUT/aggregate/results-summary.json")
    echo "  rear_filter=$F: $M"
    echo "$S,$F,$M" >> "$RESULTS"
  done
done

echo "======== A/B DONE ========"
column -t -s, "$RESULTS"
