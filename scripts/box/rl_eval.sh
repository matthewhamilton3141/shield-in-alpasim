#!/bin/bash
# Tier 2 AlpaSim eval — does the shield-trained RL policy transfer to the photoreal sim, and is it
# a crutch there too? For each curated scene, 4 arms (crutch/teacher × shield ON/OFF) over the GT
# obstacle field. driver=shielded (no inner camera model) with $SHIELD_RL_CKPT driving the rollout.
# The checkpoints (results/checkpoints/{crutch,teacher}.pt) are committed, so mounted at /mnt/shield.
#
# Prereqs on the box: `git pull` in ~/shield-in-alpasim (for the RL code + checkpoints), scenes in
# ~/alpasim/data/nre-artifacts. Usage:  N_ROLLOUTS=5 bash scripts/box/rl_eval.sh
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=$HOME/shield-data/hf_cache
cd ~/alpasim
N=${N_ROLLOUTS:-5}
SCENES=${SCENES_FILE:-~/shield-in-alpasim/scripts/box/tier1_scenes.txt}
COMMON=(deploy=local topology=1gpu "runtime.simulation_config.n_rollouts=$N" eval.video.video_layouts="[]")
RESULTS=~/rl_eval_results.csv
echo "scene,arm,test_shield,at_fault,at_fault_std,offroad,progress,progress_std,status0" > "$RESULTS"
echo "=== RL EVAL start $(date -u) n=$N ==="

agg() {
  python3 - "$1" << "PY" 2>/dev/null || echo ",,,,MISSING"
import json,sys
d=json.load(open(sys.argv[1])); m=d["metrics_results"][0]; r=d["rollouts"][0]
def g(k):
    v=m.get(k); return round(v,3) if isinstance(v,(int,float)) else v
print(",".join(str(x) for x in [g("collision_at_fault"),g("collision_at_fault_std"),
      g("offroad"),g("progress_clipped_rel"),g("progress_clipped_rel_std"),r["status"]]))
PY
}

i=0
while read -r S; do
  [ -z "$S" ] && continue
  i=$((i+1))
  echo "======== [$i] $S $(date -u +%H:%M:%S) ========"
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  PREP=out_rl_prep_$i; rm -rf "$PREP"
  uv run alpasim_wizard "${COMMON[@]}" driver=shielded wizard.run_method=NONE \
     wizard.log_dir="$PWD/$PREP" scenes.scene_ids="[$S]" > "$PREP.log" 2>&1
  SSDIR=$(grep -oE "Created sceneset directory at .*" "$PREP.log" | tail -1 | sed -E "s/.*at //")
  USDZ=$(basename "$(readlink "$SSDIR"/*.usdz 2>/dev/null | head -1)")
  HOSTUSDZ=$(find "$HOME/alpasim/data/nre-artifacts" -name "$USDZ" 2>/dev/null | head -1)
  if [ -z "$HOSTUSDZ" ]; then echo "  !! no USDZ for $S; skip"; continue; fi
  CONT="/mnt/nre-data/${HOSTUSDZ#*/nre-artifacts/}"

  for ARM in crutch teacher; do
    CKPT="/mnt/shield/results/checkpoints/${ARM}.pt"
    for SF in 1 0; do   # shield ON then OFF (the crutch test)
      OUT=out_rl_${ARM}_sf${SF}_$i; rm -rf "$OUT"
      docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
      env SHIELD_SCENE_USDZ_IN_CONTAINER="$CONT" SHIELD_OBSTACLE_SOURCE=gt \
          SHIELD_RL_CKPT_IN_CONTAINER="$CKPT" SHIELD_FILTER="$SF" \
        uv run alpasim_wizard "${COMMON[@]}" driver=shielded \
        wizard.log_dir="$PWD/$OUT" scenes.scene_ids="[$S]" > "$OUT.log" 2>&1
      M=$(agg "$OUT/aggregate/results-summary.json")
      echo "$S,$ARM,$SF,$M" >> "$RESULTS"; echo "  $ARM shield=$SF: $M"
    done
  done
done < "$SCENES"
echo "=== RL EVAL DONE $(date -u) ==="
echo "results -> $RESULTS"
