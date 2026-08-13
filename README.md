# shield-in-alpasim

Wrap [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav)'s hard safety shield as a
driver plugin for [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim), NVIDIA's open-source
closed-loop AV validation harness (the companion sim to **Alpamayo**). The question this repo
exists to answer: does a shield that provably holds 0 collisions on real KITTI drives still hold
when it's dropped into a harder, photorealistic closed-loop environment it was never tuned on?

Status: **scaffold.** The plugin is registered and correct against AlpaSim's real driver
interface (verified against the upstream source, not assumed), and it runs kitti-nav's
actual shield — but against an empty obstacle field, so the shield never has anything to
intervene on. Both open problems below are still open.

## Why a separate repo

Same reason [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav) split off from
[gsplat-rt](https://github.com/matthewhamilton3141/gsplat-rt): different dependency stack
(AlpaSim's own microservices, NuRec scene assets, a driver-plugin entry-point system) and a
different story (closed-loop validation of a driving policy) from either the reconstruction
work (gsplat-rt) or the real-sensor nav work (kitti-nav). This repo consumes kitti-nav's shield
as a dependency rather than duplicating it.

## The actual gap (read before writing code here)

AlpaSim's driver interface (`alpasim_driver.models.base.BaseTrajectoryModel`) is
**vision-first**: a driver receives `PredictionInput` (multi-camera RGB frames, speed,
acceleration, a coarse LEFT/STRAIGHT/RIGHT `DriveCommand`, plus `ego_pose_history`, an
optional `route` of rig-frame waypoints, the `previous_plan`, and an `inference_seed`) and
must return a `ModelPrediction` — **6-DoF waypoint poses** in the rig frame
(`candidate_positions` `(K, T, 3)` + `candidate_rotations` `(K, T, 3, 3)`, K sampled
candidates), not a low-level `(accel, steer)` command. Ground-plane planners build one with
the `ModelPrediction.from_planar(trajectory_xy, headings)` classmethod, which is what this
driver does. Nothing in that input carries geometry.

kitti-nav's shield (`kitti_nav.vehicle.safety_shield` /
`kitti_nav.dynamics.dynamic_safety_shield`) is the opposite shape: it filters a per-step
`(accel_cmd, steer_cmd)` against an `ObstacleField` (circle obstacles or a lidar BEV occupancy
grid) and returns a certified-safe `(accel, steer)`. It has no notion of camera pixels, and
AlpaSim's `PredictionInput` has no notion of occupancy or lidar.

So this is not a drop-in adapter. Two real problems have to be solved here, not assumed away:

1. **Where does the shield's obstacle field come from?** AlpaSim gives the driver camera frames,
   not geometry. Options to evaluate: (a) run a monocular/stereo depth or occupancy estimator
   over the camera frames to synthesize a BEV grid the shield can consume (this is exactly the
   kind of depth step gsplat-rt already does — reuse, don't rebuild), or (b) check whether
   AlpaSim's renderer/scene service exposes privileged ground-truth geometry for evaluation
   purposes even though the driver's `predict()` input doesn't carry it.
2. **The shield outputs a per-step command, AlpaSim wants a trajectory.** Need to roll the
   kinematic bicycle forward under the shield's certified `(accel, steer)` for
   `context_length / output_frequency_hz` steps to produce the `(T, 2)` waypoint sequence
   `ModelPrediction` expects, then re-run the shield closed-loop as new frames arrive.

Neither is solved yet. This repo starts with the plugin skeleton and an empty obstacle field so
the AlpaSim harness runs end-to-end with an inert shield before either problem is tackled for
real.

## Plan

1. Scaffold: plugin skeleton registered with AlpaSim's entry-point system (`alpasim.models` for
   the driver, `alpasim.configs` for its Hydra configs), empty `ObstacleField`, no perception
   yet. Interface verified against AlpaSim's upstream source.
2. Get AlpaSim running against its own bundled sample/demo scenes with the inert driver — prove
   the harness plumbing works before any shield logic is real. **Needs a machine with AlpaSim
   installed; can't be done from the Mac dev box**, which is why the tests here stop at the
   AlpaSim boundary.
3. Solve problem 1 (obstacle field from camera frames) with the smallest thing that works.
4. Wire in kitti-nav's actual shield + trajectory rollout (problem 2).
5. Compare: shielded vs. unshielded, and against AlpaSim's stock drivers (VaVAM, Transfuser),
   on collision rate — the same measured-honesty standard as kitti-nav's `RESULTS.md`.

## Layout

```
src/shield_in_alpasim/
  driver.py              # the AlpaSim driver plugin
  configs/driver/        # Hydra configs, discovered via the alpasim.configs entry point
tests/                   # pure-Python tests, no AlpaSim/GPU required to run
```

## Seeing it work

No AlpaSim, no GPU, no scene assets required:

```bash
python3 scripts/preview_trajectory.py     # -> docs/preview.png
```

![shielded trajectory preview](docs/preview.png)

Both panels are the same code path AlpaSim drives — `_rollout`'s `(T, 2)` waypoints, the
ones `ModelPrediction.from_planar` lifts into poses. Left: clear road, waypoints evenly
spaced at `v/output_frequency_hz`. Right: an obstacle in the lane, the shield brakes, and
the waypoints bunch up and stop at 15.8 m — short of the obstacle surface at 20.5 m.

Caveat worth stating plainly: **the right-hand panel injects an obstacle field by hand.**
Under AlpaSim the field is empty (problem 1), so a real sim video today shows a car
driving straight and a shield that never fires. Solving problem 1 is what makes the
in-sim video worth watching.

Inside AlpaSim, the views come from its own eval stage: `eval.video.video_layouts=[DEFAULT]`
renders a BEV map + camera + metrics `mp4` per rollout, and the eval output sorts clips
into per-violation directories (`collision_at_fault`, `offroad`, ...). That last part is
the real scoreboard for this project — the shielded driver's `collision_at_fault/` should
be empty where an unshielded run's is not.

## Running it

The plugin is CPU-only — the shield is numpy, so unlike AlpaSim's stock drivers it needs no
GPU and no checkpoint. From an AlpaSim checkout:

```bash
uv pip install -e path/to/shield-in-alpasim
uv run alpasim-info          # should list `shielded` under alpasim.models
# then reference it in a run as driver=shielded
```

## Dependencies

- [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav) — the shield, vendored as a
  sibling checkout (see `ATTRIBUTION.md`), not copied.
- [AlpaSim](https://github.com/NVlabs/alpasim) — the harness this plugs into. Apache 2.0.

## License

MIT (this repo's code). See `ATTRIBUTION.md` for AlpaSim's and kitti-nav's licenses.
