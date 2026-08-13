#!/bin/bash
# Scene sweep: unshielded VaVAM vs shielded VaVAM, one scene at a time.
#
# Why one scene at a time (not scenes.limit_to_first_n=N in a single run): the shield arms its
# geometry from a single $SHIELD_SCENE_USDZ_IN_CONTAINER, so a multi-scene shielded run would
# load the wrong scene's actors for all but one. So per scene we (1) run unshielded — which also
# downloads the artifact and builds the sceneset — (2) read that scene's USDZ path off the
# sceneset symlink, then (3) run shielded with it armed. Metrics land in ~/sweep_results.csv.
#
# Requires HF_TOKEN in the environment (scene downloads are gated). Run on the box:
#   tmux new-session -d -s sweep -e HF_TOKEN=<tok> 'bash ~/shield-in-alpasim/scripts/scene_sweep.sh clipgt-... clipgt-... > ~/sweep.log 2>&1'
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd ~/alpasim

CKPT=/mnt/drivers/vavam/VAM_width_768_pretrained_139k.pt
RESULTS=~/sweep_results.csv
COMMON=(deploy=local topology=1gpu runtime.simulation_config.n_rollouts=1 eval.video.video_layouts='[]')
echo "scene,arm,collision_at_fault,collision_rear,offroad,dist_m,gt_dist_m,progress,score,status" > "$RESULTS"

# Pull the eval summary into one CSV row; empty fields if the run produced no summary.
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

  # (1) unshielded — downloads the artifact + builds the sceneset
  OUT=out_sw_van_$S
  rm -rf "$OUT"
  uv run alpasim_wizard "${COMMON[@]}" driver=vavam \
      wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" \
      driver.model.checkpoint_path=$CKPT > "$OUT.log" 2>&1
  M=$(metric "$OUT/aggregate/results-summary.json")
  echo "  unshielded: $M"
  echo "$S,unshielded,$M" >> "$RESULTS"

  # (2) discover this scene's USDZ (container-side path) from the sceneset symlink
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$OUT.log" | tail -1 | sed -E 's/.*at //')
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! could not resolve USDZ for $S; skipping shielded"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"
  echo "  USDZ -> $CONT"

  # (3) shielded, armed with this scene's geometry
  OUT=out_sw_shd_$S
  rm -rf "$OUT"
  SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" uv run alpasim_wizard "${COMMON[@]}" driver=shielded_vavam \
      wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
  M=$(metric "$OUT/aggregate/results-summary.json")
  echo "  shielded:   $M"
  echo "$S,shielded,$M" >> "$RESULTS"
  # Confirm the shield actually loaded this scene's actors (else the number is meaningless).
  grep -oE "Loaded [0-9]+ scene actors" "$OUT.log" | tail -1 | sed 's/^/  /'
done

echo "======== SWEEP DONE ========"
column -t -s, "$RESULTS"
