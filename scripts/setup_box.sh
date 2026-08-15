#!/usr/bin/env bash
# One-command bring-up of a fresh GPU box (Lambda / any) for shield-in-alpasim.
#
# Codifies docs/BOX_SETUP.md phases 1-2 (setup + validate) — all CPU, no metered rendering — so a
# fresh instance goes from clean to "shielded registered, tests green, geometry validated" without
# hand-following the guide. Idempotent-ish: safe to re-run.
#
#   HF_TOKEN=<gated-NuRec read token> bash scripts/setup_box.sh
#
# Then the first metered render is BOX_SETUP.md phase 3.
#
# LAMBDA NOTES (different from Brev): Lambda has NO stop-and-preserve — you either run or TERMINATE
# (which wipes the disk). So (1) there's no idle auto-stop safety net: an idle A100 burns ~$30/day —
# terminate the moment you step away; (2) this script gets re-run on every new instance, so put
# data/nre-artifacts + data/drivers on a Lambda PERSISTENT FILESYSTEM to avoid re-downloading the
# gated scenes + VaVAM weights each time (point ALPASIM_DIR's data/ there, or symlink).
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN — a read token for the gated NuRec dataset (see docs/BOX_SETUP.md)}"
SHIELD_DIR=${SHIELD_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ALPASIM_DIR=${ALPASIM_DIR:-$HOME/alpasim}
KITTINAV_DIR=${KITTINAV_DIR:-$HOME/kitti-nav}
ALPASIM_REPO=${ALPASIM_REPO:-https://github.com/NVlabs/alpasim.git}
KITTINAV_REPO=${KITTINAV_REPO:-https://github.com/matthewhamilton3141/kitti-nav.git}
VAVAM_MODEL=${VAVAM_MODEL:-VaVAM-S}
# Lambda persistent filesystem mount (survives instance termination). Set DATA_FS to it (e.g.
# $HOME/shield-data) to keep the gated scenes + VaVAM weights + HF model cache across re-provisions
# — they get symlinked in below. Empty = everything on the ephemeral instance disk (re-downloaded).
DATA_FS=${DATA_FS:-}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "== 0. prerequisites =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v git >/dev/null || { echo "!! git missing"; exit 1; }
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
  echo "!! nvidia-smi found no GPU. On a fresh instance this is sometimes a stuck GPU —"
  echo "   check 'dmesg | grep NVRM'; reimage/relaunch before spending time on setup."
  exit 1
fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-0}" -ge 570 ] || echo "!! driver $DRV < 570 — NuRec (CUDA 12.8) will fail; reimage with a newer driver."

echo "== 1. clone repos (this repo is $SHIELD_DIR) =="
[ -d "$ALPASIM_DIR/.git" ]  || git clone "$ALPASIM_REPO"  "$ALPASIM_DIR"
[ -d "$KITTINAV_DIR/.git" ] || git clone "$KITTINAV_REPO" "$KITTINAV_DIR"

if [ -n "$DATA_FS" ]; then
  echo "== 1b. persist heavy dirs on the filesystem $DATA_FS (survives termination) =="
  mkdir -p "$DATA_FS/nre-artifacts" "$DATA_FS/drivers" "$DATA_FS/hf_cache" "$ALPASIM_DIR/data"
  ln -sfn "$DATA_FS/nre-artifacts" "$ALPASIM_DIR/data/nre-artifacts"  # gated scenes (~1.7 GB each)
  ln -sfn "$DATA_FS/drivers"       "$ALPASIM_DIR/data/drivers"        # VaVAM weights
  # HF cache (depth/seg models + gated scene downloads). Move an existing real dir onto the FS
  # rather than clobber it or nest a symlink inside it.
  HF="$HOME/.cache/huggingface"; mkdir -p "$HOME/.cache"
  if [ -e "$HF" ] && [ ! -L "$HF" ]; then cp -an "$HF/." "$DATA_FS/hf_cache/" 2>/dev/null || true; rm -rf "$HF"; fi
  ln -sfn "$DATA_FS/hf_cache" "$HF"
fi

echo "== 2. alpasim env (Rust toolchain for utils_rs, then uv sync) =="
cd "$ALPASIM_DIR"
# setup_local_env.sh must be SOURCED (it installs Rust into the current shell), not executed.
[ -f ./setup_local_env.sh ] && source ./setup_local_env.sh || true
uv sync --extra all

echo "== 3. install our plugin (--no-deps is REQUIRED: our alpasim_* deps are workspace pkgs) =="
uv pip install -e "$SHIELD_DIR" --no-deps

echo "== 4. wire kitti_nav into the venv (src-layout, no install) =="
SP=$(uv run python -c "import site; print(site.getsitepackages()[0])")
echo "$KITTINAV_DIR/src" > "$SP/kitti_nav.pth"
uv run python -c "import kitti_nav; print('kitti_nav OK:', kitti_nav.__file__)"

echo "== 5. VaVAM assets (public, checksummed) into alpasim/data/drivers/vavam =="
cp "$SHIELD_DIR/data/download_vavam_assets.sh" "$ALPASIM_DIR/data/"
bash "$ALPASIM_DIR/data/download_vavam_assets.sh" --model "$VAVAM_MODEL"

echo "== 6. preflight: HF token -> driver -> disk -> docker/uv -> shielded registered -> tests -> geometry =="
cd "$SHIELD_DIR"
./scripts/preflight.sh || echo "!! preflight reported issues (read above) — fix before the metered run."

cat <<'DONE'

== SETUP DONE ==
Next (BOX_SETUP.md phase 3, the first METERED run):
  cd ~/alpasim
  uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
      wizard.log_dir=$PWD/out_first scenes.limit_to_first_n=1 \
      runtime.simulation_config.n_rollouts=1

REMINDERS
  * Lambda has no auto-stop: TERMINATE when idle (an A100 is ~$30/day left running).
  * Verify 'Loaded N scene actors' in the driver log before trusting a "no interventions" result.
  * Never debug Python on the meter — it's all CPU, do it here (phase 2) or on the Mac.
DONE
