#!/bin/bash
# A/B any shield toggle env var (0 vs 1) for shielded VaVAM, one scene at a time, both arms
# armed with the scene's ground-truth geometry. Isolates that flag's effect on
# progress/collisions; VaVAM's run-to-run stochasticity hits both arms equally. Scenes must
# already be downloaded (run scene_sweep.sh once first).
#
#   bash ~/shield-in-alpasim/scripts/shield_ab.sh SHIELD_PASSTHROUGH clipgt-... clipgt-...
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd ~/alpasim
VAR="$1"; shift
RESULTS=~/${VAR}_ab_results.csv
N_ROLLOUTS=${N_ROLLOUTS:-1}   # rollouts per arm (>1 averages VaVAM's run-to-run variance)
AB_VALUES="${AB_VALUES:-0 1}" # the two values of $VAR to compare (e.g. "gt camera")
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N_ROLLOUTS" eval.video.video_layouts='[]')
echo "scene,$VAR,collision_at_fault,collision_rear,offroad,dist_m,gt_dist_m,progress,score,status" > "$RESULTS"

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
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  PREP=out_ab_prep_$S; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam wizard.run_method=NONE \
      wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E 's/.*at //')
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! could not resolve USDZ for $S; skipping"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"
  echo "  USDZ -> $CONT"

  for V in $AB_VALUES; do
    OUT=out_ab_${VAR}_${V}_$S; rm -rf "$OUT"
    env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" "$VAR=$V" \
      uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam \
      wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
    M=$(metric "$OUT/aggregate/results-summary.json")
    echo "  $VAR=$V: $M"
    echo "$S,$V,$M" >> "$RESULTS"
  done
done

echo "======== A/B DONE ========"
column -t -s, "$RESULTS"
