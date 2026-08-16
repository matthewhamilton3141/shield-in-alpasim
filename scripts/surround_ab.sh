#!/bin/bash
# A/B: front-only camera perception vs the real ftheta SURROUND rig, one scene at a time, both
# arms camera-perception + GT-armed (the GT is only for the debug field / rear filter; perception
# is from the cameras). Isolates what the surround rig buys over the single front camera — does it
# fix the lateral/rear cases (02eadd92) without over-braking the clean ones? VaVAM is stochastic,
# so use N_ROLLOUTS>=3 for a real number; n=1 is a directional first read.
#
#   bash ~/shield-in-alpasim/scripts/surround_ab.sh clipgt-... clipgt-...
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd ~/alpasim

RESULTS=~/surround_ab_results.csv
N_ROLLOUTS=${N_ROLLOUTS:-1}
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N_ROLLOUTS" eval.video.video_layouts='[]')
echo "scene,arm,collision_at_fault,collision_rear,offroad,dist_m,gt_dist_m,progress,score,status" > "$RESULTS"

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
  # Resolve this scene's USDZ (container path) via a run-free prep.
  PREP=out_sab_prep_$S; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam wizard.run_method=NONE \
      wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E 's/.*at //')
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! could not resolve USDZ for $S; skipping"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"
  echo "  USDZ -> $CONT"

  # front-only camera perception (single front cam, the prior baseline)
  OUT=out_sab_front_$S; rm -rf "$OUT"
  env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=camera \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(metric "$OUT/aggregate/results-summary.json"); echo "  front:    $M"; echo "$S,front,$M" >> "$RESULTS"

  # real ftheta surround (5-camera 360 rig; SHIELD_OBSTACLE_SOURCE defaults to camera in its config)
  OUT=out_sab_surround_$S; rm -rf "$OUT"
  env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" \
    uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam_surround \
    wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(metric "$OUT/aggregate/results-summary.json"); echo "  surround: $M"; echo "$S,surround,$M" >> "$RESULTS"
done

echo "======== SURROUND A/B DONE ========"
column -t -s, "$RESULTS"
