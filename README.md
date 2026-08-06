# shield-in-alpasim

Wrap [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav)'s hard safety shield as a
driver plugin for [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim), NVIDIA's open-source
closed-loop AV validation harness (the companion sim to **Alpamayo**). The question this repo
exists to answer: does a shield that provably holds 0 collisions on real KITTI drives still hold
when it's dropped into a harder, photorealistic closed-loop environment it was never tuned on?

Status: **scaffold, not yet functional.** No shield logic has been ported here yet.

## Why a separate repo

Same reason [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav) split off from
[gsplat-rt](https://github.com/matthewhamilton3141/gsplat-rt): different dependency stack
(AlpaSim's own microservices, NuRec scene assets, a driver-plugin entry-point system) and a
different story (closed-loop validation of a driving policy) from either the reconstruction
work (gsplat-rt) or the real-sensor nav work (kitti-nav). This repo consumes kitti-nav's shield
as a dependency rather than duplicating it.

## The actual gap (read before writing code here)

AlpaSim's driver interface (`alpasim_driver.models.base.BaseTrajectoryModel`) is **vision-only**:
a driver receives `PredictionInput` (multi-camera RGB frames, speed, acceleration, a coarse
LEFT/STRAIGHT/RIGHT command) and must return a `ModelPrediction` — a **trajectory** of `(x, y)`
waypoints in the rig frame, not a low-level `(accel, steer)` command.

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

Neither is solved yet. This repo starts with the plugin skeleton and a stub obstacle source so
the AlpaSim harness runs end-to-end with a dummy shield before either problem is tackled for
real.

## Plan

1. Scaffold (this commit): plugin skeleton registered with AlpaSim's entry-point system, stub
   `ObstacleField`, no real perception yet.
2. Get AlpaSim running against its own bundled sample/demo scenes with the stub driver — prove
   the harness plumbing works before any shield logic is real.
3. Solve problem 1 (obstacle field from camera frames) with the smallest thing that works.
4. Wire in kitti-nav's actual shield + trajectory rollout (problem 2).
5. Compare: shielded vs. unshielded, and against AlpaSim's stock drivers (VaVAM, Transfuser),
   on collision rate — the same measured-honesty standard as kitti-nav's `RESULTS.md`.

## Layout

```
src/shield_in_alpasim/   # the AlpaSim driver plugin
tests/                   # pure-Python tests, no AlpaSim/GPU required to run
docs/
```

## Dependencies

- [kitti-nav](https://github.com/matthewhamilton3141/kitti-nav) — the shield, vendored as a
  sibling checkout (see `ATTRIBUTION.md`), not copied.
- [AlpaSim](https://github.com/NVlabs/alpasim) — the harness this plugs into. Apache 2.0.

## License

MIT (this repo's code). See `ATTRIBUTION.md` for AlpaSim's and kitti-nav's licenses.
