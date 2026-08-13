#!/usr/bin/env bash
# Pre-flight checks for a GPU box, ordered cheapest-failure-first.
#
# Every check here runs on CPU in seconds. The point is that by the time you start a
# rendered rollout — the only genuinely expensive thing — nothing is left that could fail
# for a boring reason. Run it top to bottom; it stops at the first failure.
#
#   ./scripts/preflight.sh                 # checks only
#   SHIELD_SCENE_USDZ=/path/scene.usdz ./scripts/preflight.sh
#
# Nothing here is destructive and nothing here needs the GPU except check 2, which only
# reads the driver version.

set -uo pipefail

ALPASIM_DIR="${ALPASIM_DIR:-$HOME/alpasim}"
SHIELD_DIR="${SHIELD_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KITTI_NAV_DIR="${KITTI_NAV_DIR:-$(dirname "$SHIELD_DIR")/kitti-nav}"

pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# --- 1. credentials: the thing with the longest lead time ---------------------------
step "1. Hugging Face access"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is unset. Scene downloads need it: export HF_TOKEN=<token>"
pass "HF_TOKEN is set"
warn "Token being set does NOT mean the gated dataset is approved. If downloads fail with"
warn "GatedRepoError, request access at huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"

# --- 2. host can run the NRE container ----------------------------------------------
step "2. GPU and driver"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — is this a GPU box?"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
pass "$GPU_NAME, ${VRAM} MiB, driver $DRIVER"
# The NRE container is CUDA 12.8; older drivers die with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
if [ "${DRIVER%%.*}" -lt 570 ]; then
    fail "driver $DRIVER is too old — NRE needs >= 570.x for CUDA 12.8. Reimage; do not debug this."
fi
pass "driver supports CUDA 12.8"

# --- 3. disk: scene artifacts are large ---------------------------------------------
step "3. Disk"
AVAIL_GB="$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')"
[ "$AVAIL_GB" -ge 150 ] || fail "only ${AVAIL_GB} GB free on \$HOME; budget ~200 GB for images + scenes"
pass "${AVAIL_GB} GB free"

# --- 4. toolchain --------------------------------------------------------------------
step "4. Toolchain"
for tool in docker uv; do
    command -v "$tool" >/dev/null || fail "$tool not found (see alpasim docs/ONBOARDING.md)"
done
docker info >/dev/null 2>&1 || fail "docker needs sudo or is not running: sudo usermod -aG docker \$USER, then re-login"
pass "docker runs without sudo"
pass "uv $(uv --version 2>/dev/null | awk '{print $2}')"
command -v cargo >/dev/null || warn "cargo not found — alpasim's utils_rs is a Rust extension; setup_local_env.sh can install it"

# --- 5. checkouts ---------------------------------------------------------------------
step "5. Checkouts"
[ -d "$ALPASIM_DIR" ] || fail "no alpasim checkout at $ALPASIM_DIR (override with ALPASIM_DIR=)"
pass "alpasim at $ALPASIM_DIR"
[ -d "$KITTI_NAV_DIR" ] || fail "no kitti-nav checkout at $KITTI_NAV_DIR — the shield itself lives there"
pass "kitti-nav at $KITTI_NAV_DIR"

# --- 6. our plugin is registered ------------------------------------------------------
step "6. Plugin registration"
cd "$ALPASIM_DIR" || fail "cannot enter $ALPASIM_DIR"
if uv run alpasim-info 2>/dev/null | grep -q shielded; then
    pass "'shielded' is registered under alpasim.models"
else
    fail "'shielded' not listed by alpasim-info. Install the plugin *without* its deps —
        its alpasim_* requirements are workspace packages, not on PyPI:
            cd $ALPASIM_DIR && uv pip install -e $SHIELD_DIR --no-deps
        See docs/BOX_SETUP.md for why."
fi

# --- 7. our own tests, against the box's Python ----------------------------------------
step "7. Unit tests"
if (cd "$SHIELD_DIR" && uv run --project "$ALPASIM_DIR" python -m pytest -q 2>&1 | tail -3); then
    pass "shield-in-alpasim tests pass in the box environment"
else
    warn "tests did not pass under uv; try a plain 'python3 -m pytest -q' in $SHIELD_DIR"
fi

# --- 8. geometry against a real scene, still no renderer -------------------------------
step "8. Scene geometry"
if [ -n "${SHIELD_SCENE_USDZ:-}" ]; then
    (cd "$SHIELD_DIR" && uv run --project "$ALPASIM_DIR" python scripts/check_scene_geometry.py) \
        || fail "scene geometry check failed — fix before rendering anything"
    pass "ground-truth obstacle path validated against a real scene"
else
    warn "SHIELD_SCENE_USDZ unset: skipping the geometry check."
    warn "This is the highest-value check there is and it costs no GPU time. Download one"
    warn "scene first, then re-run with SHIELD_SCENE_USDZ=/path/to/scene.usdz"
fi

step "Pre-flight complete"
cat <<'EOF'
  Next, still cheap — generate configs without simulating:

    uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
        wizard.run_method=NONE wizard.log_dir=$PWD/out_dryrun

  Then the first metered run, one scene, one rollout. See docs/BOX_SETUP.md.
EOF
