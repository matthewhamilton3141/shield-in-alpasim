# shield-in-alpasim — session handoff

Current state, the open decision, and a runbook. The *why* behind the design lives in
[`README.md`](README.md) ("The actual gap"); this file is where to pick up.

---

## ▶ Where things stand

- **`main` = `411aec9`**, working tree clean. Branch `fix/real-alpasim-interface` is merged
  and can be deleted.
- **4 tests green**: `python3 -m pytest -q`. Pure Python, no AlpaSim/GPU needed — kitti-nav
  is found as a sibling checkout via `tests/conftest.py`.
- **kitti-nav `main` = `7a40d29`** (README now leads with `docs/drive_scene.gif` and the
  0-collision result; its shield-in-alpasim section was corrected to match reality).
- **Never run under AlpaSim.** Everything below the AlpaSim boundary is verified by reading
  upstream source, not by executing it. That is the single biggest caveat here.

## What this session changed

The scaffold had been written against an *assumed* AlpaSim API. Checked against
`NVlabs/alpasim` upstream and corrected four things, one a guaranteed runtime crash:

| Was | Actually |
| --- | --- |
| `ModelPrediction(trajectory_xy=…, headings=…)` | Takes `candidate_positions` `(K,T,3)` + `candidate_rotations` `(K,T,3,3)`; ground-plane planners use `ModelPrediction.from_planar()`. The old call would have `TypeError`'d on first inference. |
| `camera_ids = ["front_wide"]` hardcoded | Real IDs are like `camera_front_wide_120fov`, and come from config. `from_config` was accepting `camera_ids`/`context_length` and dropping both. |
| `_headings_from_xy` | Reimplemented `BaseTrajectoryModel._compute_headings_from_trajectory` line for line. |
| `NoObstacles` class | Reimplemented kitti-nav's `CircleField(None)`. |

Also: `predict()` now calls `_validate_cameras`; `_rollout` delegates to
`kitti_nav.shielded_rollout` (which additionally returns the intervention/collision stats
plan step 5 needs); obstacles are constructor-injectable.

**A plugin needs two entry points, not one.** `alpasim.models` alone lets the harness load
the class but gives no way to select it. Added `alpasim.configs` +
`configs/driver/shielded{,_configs}.yaml`, modelled on AlpaSim's `manual` driver (its
closest stock analogue — CPU-only, no checkpoint).

## ⚠ Open decision, blocking everything else

**`ShieldedDriver` should probably become a decorator over another `BaseTrajectoryModel`,
not a standalone one.**

The shield is a *filter*: it takes a proposed `(accel, steer)` and certifies it. It never
proposes one. Today `_rollout` commands `(0.0, 0.0)` — "go straight, hold speed" — which is
coasting, not driving. Nothing here will ever drive itself without a policy upstream.

kitti-nav's PPO policies (`models/ppo_kitti_shielded.zip`, 0 collisions on KITTI transfer)
do drive, but consume **lidar BEV occupancy**, and AlpaSim hands you **camera pixels** — so
they don't transfer without solving problem 1 first.

The alternative: wrap AlpaSim's own camera-based drivers. `from_config` pulls the inner
model from the registry (`alpasim_plugins.models.get("vavam")`), `predict()` becomes
call-inner → invert-to-`(accel, steer)` → shield → re-roll. VaVAM/Transfuser first (light,
AlpaSim's default); Alpamayo only for the part that needs it — it is the only driver that
emits `reasoning_text`, so it is the only one where "when the shield vetoes, what did the
model claim it was doing?" is answerable.

## The two open problems

1. **Where does the obstacle field come from?** `PredictionInput` carries no geometry.
   Options: (a) synthesize a BEV grid from camera frames via monocular depth — gsplat-rt
   already has a depth step worth reusing; (b) check whether AlpaSim exposes privileged
   ground-truth geometry. **Check (b) first** — `src/eval/src/eval/scorers/ground_truth.py`
   and `min_distance_to_obstacle.py` upstream are suggestive, and it is unverified.
2. **Trajectory ⇄ per-step command.** The shield emits `(accel, steer)`; AlpaSim wants
   waypoints. Rolling forward is done; *inverting* an inner model's trajectory into
   commands is not, and is subtle — the shield reasons in a rate-limited kinematic bicycle,
   and a network's waypoints need not be feasible under it.

**The thing that makes this interesting, and the flaw to design around:** a braking
certificate is only as sound as the geometry it is computed against. Putting a learned
depth estimator underneath converts "provably no collisions" into "no collisions if
perception was right" — which is the failure the shield existed to catch. So the sharper
experiment is: run with ground-truth geometry (guarantee intact), then swap in learned
perception and **measure how much the certificate degrades**. That turns the flaw into the
subject and needs no monocular-perception breakthrough to produce a result.

## ▶ Next step

Plan step 2: get AlpaSim running against a bundled sample scene with the inert driver.
Proves entry points, Hydra config, `camera_ids`, and `context_length` before any shield
logic is real. Needs a machine with AlpaSim — cannot be done from the Mac.

Sizing, from AlpaSim's own docs: **A100 40GB is the pick** (Brev, ~$1.10/hr; their H100 at
$1.99 is the upgrade if 40 GB is tight — their A100 80GB at $6.21 is not). Our driver uses
**zero VRAM** (`gpus: null`, `device: cpu`), so the budget is renderer + physics +
trafficsim only. ~200 GB disk. Bring-up is ~$10; realistic cost is hours of setup, not
compute. Note the 48/96 GB figures in `docs/ONBOARDING.md` are for the *optional*
FlashDreams renderer, not the default NuRec path.

**Do before starting the meter:** request access to the gated HF dataset
(`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`) — approval is not instant, and scene
downloads fail with `GatedRepoError` without it. Pick an image with driver ≥ 570.x (the NRE
container is CUDA 12.8; too-old drivers fail with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
Skip `download_vavam_assets.sh` unless wrapping VaVAM — this driver has no checkpoint.

## ▶ Runbook

```bash
python3 -m pytest -q                      # 4 tests, no AlpaSim/GPU
python3 scripts/preview_trajectory.py     # -> docs/preview.png
python3 scripts/preview_trajectory.py --speed 14 --obstacle-x 18 --hz 4

# on a box with AlpaSim:
uv pip install -e path/to/shield-in-alpasim
uv run alpasim-info                       # expect `shielded` under alpasim.models
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded wizard.log_dir=$PWD/out
```

`preview_trajectory.py` injects an obstacle by hand to show the shield braking (stops at
15.8 m, obstacle surface at 20.5 m). Under AlpaSim the field is empty, so a real run today
shows a car driving straight and a shield that never fires — that is expected, not a bug.

Videos come from AlpaSim's eval stage: `eval.video.video_layouts=[DEFAULT]` renders BEV +
camera + metrics per rollout, and output sorts clips into `collision_at_fault/`, `offroad/`.
That directory count is the real scoreboard.
