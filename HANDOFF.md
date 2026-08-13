# shield-in-alpasim — session handoff

Current state, the open decision, and a runbook. The *why* behind the design lives in
[`README.md`](README.md) ("The actual gap"); this file is where to pick up.

---

## ▶ Where things stand

- **`main` = `c5d2bff`**, working tree clean. Branch `fix/real-alpasim-interface` is merged
  and can be deleted.
- **17 tests green**: `python3 -m pytest -q`. Pure Python, no AlpaSim/GPU needed — kitti-nav
  is found as a sibling checkout via `tests/conftest.py`.
- **`obstacles.py` exists and the shield brakes for scene geometry** (`test_obstacles.py`).
  Not yet reachable from `predict()` — see the wiring gap below.
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

Lidar is a "not yet", not a "never": `sensorsim.proto:20` declares `render_lidar` returning
a `point_xyzs_buffer`, but nothing calls it — no Python caller in the tree, and
`runtime/services/sensorsim_service.py:504` reads `TODO(mwatson): Add requests/handling for
lidars`. The protocol slot is reserved; the plumbing is absent. Don't plan around it.

The alternative: wrap AlpaSim's own camera-based drivers. `from_config` pulls the inner
model from the registry (`alpasim_plugins.models.get("vavam")`), `predict()` becomes
call-inner → invert-to-`(accel, steer)` → shield → re-roll. VaVAM/Transfuser first (light,
AlpaSim's default); Alpamayo only for the part that needs it — it is the only driver that
emits `reasoning_text`, so it is the only one where "when the shield vetoes, what did the
model claim it was doing?" is answerable.

## ✔ Settled: where the obstacle field comes from

**Ground-truth geometry is available, but only out-of-band.** Verified by reading upstream
(shallow clone; paths repo-relative to `NVlabs/alpasim`).

The two eval scorers this file used to point at are **post-hoc** — `ground_truth.py` and
`min_distance_to_obstacle.py` both take a `SimulationResult` and read `actor_polygons`,
which the eval accumulator assembles *after* the rollout. Real geometry, wrong side of the
run. That lead is dead.

The in-band channel is genuinely empty, as assumed. `PredictionInput`
(`src/driver/src/alpasim_driver/models/base.py:46`) has no geometry, and neither does the
wire message behind it — `DriveRequest` (`src/grpc/alpasim_grpc/v0/egodriver.proto:96`) is
just `session_uuid`, two timestamps, and a `renderer_data` blob (the renderer's bundled
payload). The servicer is not dropping anything.

**What works instead: load the scene artifact directly.** It is a plain library import from
the package the driver already depends on, so no new dependency and no gRPC client.

| What | Where |
| --- | --- |
| `SceneDataSource.traffic_objects`, and `.map` (a `VectorMap`, also gives offroad) | `alpasim_utils/scene_data_source.py:51` |
| `TrafficObject{track_id, aabb, trajectory, is_static, label_class}` — `AABB` is `size_x/y/z`, `Trajectory` is pose over time | `alpasim_utils/scenario.py:304` |
| Concrete USDZ loader | `alpasim_utils/artifact.py:165` |

Box extent + pose per timestamp is exactly the obstacle field the shield wants — a better
fit than a synthesized BEV grid. The runtime does no more than this itself:
`runtime/services/traffic_service.py:66-95` reads the same `traffic_objs` and packs them
into gRPC.

The driver learns *which* scene from `DriveSessionRequest.debug_info.scene_id`. Read the
comment on that field (`egodriver.proto:53`): *"will not be present in benchmark runs to
avoid any potential data leakage."* **AlpaSim has explicitly marked this as the privileged
channel.** Fine for a ground-truth baseline — that arm is supposed to cheat — but label it
as such in any writeup, and note the GT arm therefore cannot run in benchmark mode.

Two constraints that shape the experiment:

1. **Exact only under replay traffic.** `wizard/configs/base_config.yaml:10` defaults to
   `trafficsim: disabled`, and in that mode physics derives actor poses straight from the
   logged trajectories (`runtime/runtime_context.py:157-165`) — so the on-disk log *is* the
   simulated state, exactly. Under `trafficsim=catk` actors react and diverge after
   `handover_time_us` and the artifact goes stale. The guarantee holds against
   non-reactive traffic, which is arguably the harder test: logged actors never yield.
2. **Even the GT arm inherits ego-pose noise.** Logged trajectories are in the `local` ENU
   frame; the ego pose handed to the driver is deliberately noised (`local -> rig_est`,
   CONTRIBUTING.md "Coordinate Systems"). Transforming GT obstacles into ego frame passes
   through that noise. So the baseline is *perfect geometry, imperfect localization* — a
   sharper story than a clean baseline, and measurable as its own degradation term.

**Consequence:** the guarantee-intact arm needs no monocular-depth work to get running. The
degradation experiment now only needs the learned-perception arm built, and the decorator
decision above is the real blocker again.

### Built: `src/shield_in_alpasim/obstacles.py`

`TrafficObjects` → oriented boxes → disc cover → kitti-nav `CircleField`, in the ego's rig
frame. Same conservative cover kitti-nav uses on the ego (`vehicle.footprint_discs`), but
re-derived rather than called: that one measures offsets from the *rear axle* and takes a
`VehicleConfig`, while an actor's pose sits at its box centre. Two lines of shared algebra —
if the radius formula changes upstream, change it here too.

The pure-numpy core takes plain arrays; only `field_from_traffic_objects` touches AlpaSim
types, and only through `vec3` / `yaw()` / `interpolate_pose` / `get_time_range_tuple`,
because `Pose` and `Trajectory` are **compiled Rust** (`utils_rs`) and cannot be imported on
the Mac. Tests fake exactly that surface and nothing wider.

Decisions worth knowing:
- Actors whose track does not cover *now* are **dropped, not extrapolated** — an ended track
  says nothing about where the actor is, and a phantom obstacle in front of a braking
  certificate is worse than none. Static objects are exempt (constant pose, so clamping is
  exact).
- The `EGO` box is dropped by id; the runtime prepends it, and shielding against your own
  footprint pins the car at zero speed.
- Yaw-only projection to the ground plane. The shield is a BEV kinematic model with no pitch
  or roll, so full SE(3) would be discarded a line later. Costs a little on graded road.
- The cover's **longitudinal** bulge (~0.60 m for a 4.5 m car at 5 discs) dominates its
  lateral excess (~0.12 m) — a disc circumscribing a segment overhangs the flat end. Always
  outward, so gaps read short, never long. Tests assert this analytically.

Sanity check, not vacuous: from 12 m/s at an actor 30 m ahead, an empty field runs the nose
to 75.8 m; this field stops it at 26.30 m, with the actor's face at 27.75 m.

### ⚠ The wiring gap: `predict()` cannot reach this yet

`PredictionInput` has no scene id and no timestamp, and `BaseTrajectoryModel` has **no
session hook** — the servicer keeps `debug_scene_id` on its own `Session`
(`driver/main.py:174,218`) and never passes it to the model. So the driver cannot ask "which
scene am I in?" through the sanctioned interface. Two ways out, unresolved:

- **(a) Config-level scene path.** Add the artifact path to `configs/driver/shielded.yaml`,
  load `traffic_objects` once in `from_config`. No fork. One scene per run — which is all
  the baseline arm needs at first. Timestamp can come off the newest `CameraFrame`, and ego
  pose off the last entry of `ego_pose_history`.
- **(b) Fork `main.py`** to put `scene_id` (and a timestamp) on `PredictionInput`. Honest
  about what it is — the privileged channel, made explicit — and it scales to multi-scene
  runs, but it is a patch against upstream to carry.

Start with (a); it is reversible and unblocks a first result. Neither is verifiable from the
Mac.

## The remaining open problem

**Trajectory ⇄ per-step command.** The shield emits `(accel, steer)`; AlpaSim wants
waypoints. Rolling forward is done; *inverting* an inner model's trajectory into commands
is not, and is subtle — the shield reasons in a rate-limited kinematic bicycle, and a
network's waypoints need not be feasible under it.

**The thing that makes this interesting, and the flaw to design around:** a braking
certificate is only as sound as the geometry it is computed against. Putting a learned
depth estimator underneath converts "provably no collisions" into "no collisions if
perception was right" — which is the failure the shield existed to catch. So the sharper
experiment is: run with ground-truth geometry (guarantee intact), then swap in learned
perception and **measure how much the certificate degrades**. That turns the flaw into the
subject and needs no monocular-perception breakthrough to produce a result — and as of the
section above, the ground-truth arm is confirmed buildable, so this is no longer contingent.

## ▶ Next step

Two tracks, and the first no longer needs a GPU.

**On the Mac:** pick (a) or (b) for the wiring gap above and connect `obstacles.py` to
`predict()`. The geometry itself is done and tested; what is missing is only how the driver
learns which scene it is in. Everything downstream of that call is already exercised.

**Plan step 2, on a rented box:** get AlpaSim running against a bundled sample scene with
the inert driver. Proves entry points, Hydra config, `camera_ids`, and `context_length`
before any shield logic is real. Cannot be done from the Mac.

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

The wizard defaults to `trafficsim: disabled`, which is what the ground-truth arm wants —
leave it alone rather than passing `trafficsim=catk`, or the on-disk geometry stops matching
the sim (see "Settled" above).

`preview_trajectory.py` injects an obstacle by hand to show the shield braking (stops at
15.8 m, obstacle surface at 20.5 m). Until the obstacle adapter lands the field is empty
under AlpaSim, so a real run today shows a car driving straight and a shield that never
fires — that is expected, not a bug.

Videos come from AlpaSim's eval stage: `eval.video.video_layouts=[DEFAULT]` renders BEV +
camera + metrics per rollout, and output sorts clips into `collision_at_fault/`, `offroad/`.
That directory count is the real scoreboard.
