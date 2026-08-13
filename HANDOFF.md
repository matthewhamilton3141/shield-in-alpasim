# shield-in-alpasim — session handoff

Current state, the open decision, and a runbook. The *why* behind the design lives in
[`README.md`](README.md) ("The actual gap"); this file is where to pick up.

---

## ▶ Session update — first box bring-up (2026-08-13)

A box exists and is **stopped, not deleted** — resume with `brev start shield-a100`
(Crusoe `a100-80gb.1x`, A100 80GB, **stoppable**, ~$1.98/hr). Stopping preserves the disk,
so none of the below needs redoing. Operational lessons are in memory
(`brev-box-operational-notes`); the short version:

- **Transport:** `brev exec` is unreliable (SSH drops, token expiry). Use `brev refresh`
  then `ssh -F ~/.brev/ssh_config shield-a100`, and run long jobs in **tmux**.
- **Driver:** the Crusoe image ships NVIDIA 565; NuRec needs ≥570. Upgraded in place
  (unhold the mass apt-hold → `cuda-drivers-570` → purge 565 → reboot). `nvidia-smi` = 570.211.
- **Billing:** guest `shutdown -h` does NOT stop Brev billing; only `brev stop` does.

**Done on the box (all verified, not assumed):** `uv sync --extra all` (torch 2.8+cu128),
`shielded` registered as Model **and** Config, **32 tests green**, one scene downloaded
(`~/alpasim/data/nre-artifacts/all-usdzs/23dd34ea-…usdz`, 1.7 GB), and
**`check_scene_geometry.py` PASSES** on it (worst clearance 1.02 m — frames, cover, sampling,
ego footprint all agree). `data/drivers` mount dir created. HF token supplied ephemerally,
never persisted to the box.

**Three real bugs found + fixed + pushed (`97b0917` on `origin/main`):**
1. Stale code — the box had cloned `origin/main` (411aec9), 6 commits behind the whole
   ground-truth arm. Now pushed, so a fresh clone is correct.
2. `shielded_configs.yaml` set a stray `wizard.external_services.driver` that conflicted
   with the in-sim driver service and failed wizard validation. Removed.
3. `obstacles.py` clamped actor tracks with an inclusive `[start, end]`, but AlpaSim's
   `interpolate_pose` is half-open `[start, end)` — crashed on a moving actor whose track
   ends exactly at the query time. The test fake had the same inclusive bug, so 32 green
   tests hid it. Both fixed to half-open.

**⚠ The phase-3 blocker (the handoff's long-flagged one, now characterized):** the driver runs
in the **`alpasim-base` container** (built from `Dockerfile` via `uv sync` of the workspace;
in-tree plugins' entry points are baked in at build). Our **out-of-tree** plugin
(`~/shield-in-alpasim`) and `~/kitti-nav` are **not in that image**, and the container mounts
only alpasim's own `src/`+`plugins/` (`base_config.yaml:169-181`). So a render will start the
renderer (GPU) then fail to load `shielded` in the driver container. The host-venv editable
install + `kitti_nav.pth` do **not** reach the container. Two ways in, both CPU/Docker work to
be done **off the meter** before the next render:
  - (a) make our plugin a workspace member under `~/alpasim/plugins/…` (mounted at
    `/repo/plugins`) so `uv run` picks it up — cleanest if `uv run` re-syncs it in-container;
  - (b) add volume mounts for `~/shield-in-alpasim`+`~/kitti-nav` and prefix
    `services.driver.command` with `uv pip install -e … --no-deps` + a `kitti_nav` path.
  Also still to wire: `SHIELD_SCENE_USDZ` **inside** the container via
  `services.driver.environments`, pointing at the container-side scene path.

**Resume recipe (concrete — these specifics cost real time to re-derive):**
```bash
brev start shield-a100 && brev refresh
S="ssh -F ~/.brev/ssh_config shield-a100"     # NOTE: non-interactive ssh does NOT source
                                              # .bashrc, so EVERY remote uv/cargo call must
                                              # first: export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
$S "export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH; cd ~/alpasim; uv run alpasim-info | grep shielded"
```
- Downloaded scene (host path for `SHIELD_SCENE_USDZ` / `check_scene_geometry.py`):
  `~/alpasim/data/nre-artifacts/all-usdzs/23dd34ea-a8d1-410c-aef7-d13f554cc4c9.usdz`
- `kitti_nav` is importable via a `.pth` we added:
  `~/alpasim/.venv/lib/python3.12/site-packages/kitti_nav.pth` → `~/kitti-nav/src`
- Run our code with `uv run --project ~/alpasim …`; unit tests need `--with pytest`.
- HF downloads (token never persisted): pipe it into a tmux env var, e.g.
  `cat ~/.cache/huggingface/token | $S "read -r T; tmux new-session -d -s dl -e HF_TOKEN=$T '<cmd>'"`
- Local `HANDOFF.md` is committed (one commit ahead of `origin`, not pushed; the fixes at
  `97b0917` are pushed). Editing shield code on the Mac then `rsync -az --delete --exclude=.venv
  --exclude=__pycache__ -e "ssh -F ~/.brev/ssh_config" ~/Documents/shield-in-alpasim/
  shield-a100:shield-in-alpasim/` keeps the box in sync; re-run `uv pip install -e
  ~/shield-in-alpasim --no-deps` if entry points change.

---

## ▶ Where things stand

- **Local `main` is one commit ahead of `origin`** (this handoff commit, not pushed).
  `origin/main` = `97b0917` (the three real-scene fixes above). The "session update" section
  at the top is the live frontier; the notes below predate the box and are kept for the *why*.
- **32 tests green**: `python3 -m pytest -q`. Pure Python, no AlpaSim/GPU needed — kitti-nav
  is found as a sibling checkout via `tests/conftest.py`.
- **The ground-truth arm is code-complete.** `predict()` builds a live obstacle field from
  the scene's actors and the shield brakes for them. Needs `$SHIELD_SCENE_USDZ` set.
- **Everything that can be done without a GPU is done.** The next step is a Brev box; see
  [`docs/BOX_SETUP.md`](docs/BOX_SETUP.md).
- **kitti-nav `main` = `7a40d29`** (README leads with `docs/drive_scene.gif` and the
  0-collision result; its shield-in-alpasim section matches reality).
- **⚠ Never run under AlpaSim.** Everything below the AlpaSim boundary is verified by reading
  upstream source, not by executing it. That is the single biggest caveat in this repo, and
  it applies to `scripts/preflight.sh` and `scripts/check_scene_geometry.py` too — both are
  unrun. Preflight is built to fail loudly and early for that reason.

## How we got here

Three rounds, each correcting the one before:

1. **Scaffold**, written against an *assumed* AlpaSim API.
2. **Corrected against upstream** — four fixes, one a guaranteed `TypeError` on first
   inference (`ModelPrediction` takes `candidate_positions`/`candidate_rotations`, not
   `trajectory_xy`/`headings`; real camera IDs look like `camera_front_wide_120fov` and come
   from config; two kitti-nav helpers had been reimplemented instead of imported). Also
   learned that **a plugin needs two entry points**: `alpasim.models` loads the class,
   `alpasim.configs` is what makes `driver=shielded` resolvable.
3. **Ground-truth geometry** — the sections below.

The recurring lesson: every assumption about AlpaSim's interface that was not read from
upstream source turned out to be wrong. `/tmp/alpasim-src` (shallow clone) is worth
re-cloning to check anything before building on it.

## ⚠ Open decision — no longer blocks the first result, still blocks the experiment

**`ShieldedDriver` should probably become a decorator over another `BaseTrajectoryModel`,
not a standalone one.**

The shield is a *filter*: it takes a proposed `(accel, steer)` and certifies it. It never
proposes one. Today `_rollout` commands `(0.0, 0.0)` — "go straight, hold speed" — which is
coasting, not driving. Nothing here will ever drive itself without a policy upstream.

**What this does and does not block.** A first run is now possible without deciding: the car
coasts, the shield brakes for real scene actors, and that already produces a video and an
intervention count. What it blocks is the *interesting* comparison — a policy that actually
drives, and therefore a shield that has something non-trivial to veto.

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

### ✔ Wired: `scene.py`, via an environment variable

`predict()` now builds a live field. `SHIELD_SCENE_USDZ` names the scene artifact; unset
means an empty field and the old inert coasting, because a driver that refuses to start is
worse on a metered box than one that does nothing. A path that is set but wrong raises —
that case means the run was *meant* to have geometry, and silently not having it is
indistinguishable from a shield that never fires.

**Why an env var and not config.** The config route does not work. The driver merges its
YAML onto a structured `DriverConfig` (`driver/main.py:1189`), so OmegaConf is in struct
mode and an unknown key under `model:` raises rather than passing through — adding one means
forking AlpaSim's schema. And the driver cannot discover the scene itself: `PredictionInput`
has no scene id, `BaseTrajectoryModel` has no session hook, and the servicer keeps
`debug_scene_id` to itself (`main.py:174,218`). The wizard already supports an
`environments` list on every service (see `trafficsim/catk.yaml`), so the path rides that.

Ego pose and timestamp come off the last entry of `ego_pose_history` — gRPC `PoseAtTime`,
newest last, kept sorted by the servicer (`main.py:335`).

### ⚠ Caught while wiring: the shield was shielding the wrong car

kitti-nav's `VehicleConfig` is a VW Passat (4.77 x 1.82 m, 0.97 m rear overhang) because
that is what recorded KITTI. **AlpaSim's default ego is a Mercedes S223 — 5.393 x 2.109 m,
1.3 m overhang** (`alpasim_utils/scenario.py:26`), and runtime collides against *that*
(`unbound_rollout.py:294`). Unfixed, the shield would have certified a footprint ~0.6 m
short and ~0.3 m narrow of the real body: optimistic in the exact direction a safety
envelope must never be, and it would have shown up as unexplained collisions in a run that
looked otherwise correct.

`ego_config_from_rig` now takes the geometry from the scene's own rig config. Only geometry
— AlpaSim's `VehicleConfig` has no steering or brake model to copy. **Known remaining gap:**
wheelbase stays kitti-nav's 2.71 m against the S223's ~3.11 m, so the bicycle model turns
slightly tighter than the real car. That affects the swept path, not the straight-line
braking distance the certificate mostly rests on, but it is a real approximation and should
be closed if the shield ever certifies turning manoeuvres.

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

**Everything that can be done off a GPU is done.** The remaining work needs a box.

Full plan, instance sizing, cost model and the phased command sequence now live in
[`docs/BOX_SETUP.md`](docs/BOX_SETUP.md). The short version:

- **One thing to start immediately**, because it has a lead time and blocks everything:
  request the gated HF dataset (`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`). Free.
- **Provider is Brev**, A100 40 GB at ~$1.10/hr. There were $100 of AWS credits available
  and we passed: AWS has no hard spending cap, so the tail risk is an unbounded bill from a
  forgotten instance, and Brev's idle auto-stop addresses exactly that. The whole bring-up
  should cost under $30. Also relevant: AWS has no single-A100 instance, so the sizing would
  have had to change anyway.
- **Order of operations on the box is the whole point:** `scripts/preflight.sh` →
  `wizard.run_method=NONE` (generates configs and fetches artifacts *without simulating*) →
  `scripts/check_scene_geometry.py` → only then a rendered run. Phases 1–2 use no GPU at
  all, and they are where nearly all the hours go.
- **Set the auto-stop, and run `sudo shutdown -h +240` on connect.** A box left running
  overnight costs more than every other inefficiency combined.

`check_scene_geometry.py` is the one to run first and the one most likely to catch a real
bug: it replays the scene's **logged human drive** against the scene's **logged actors** and
asserts the ego never collides. A real drive did not crash, so a collision means our
geometry is wrong — frames, quaternion convention, or ego footprint. Pure CPU, no renderer,
no cost.

### Budget: $24.73 on the Brev account

At ~$1.10/hr that is **~22 hours of A100 time** — the unit worth thinking in, because one
forgotten overnight is ~10 hours, i.e. half the balance for nothing.

Checkpoints to tell whether it is going well:

| By | Should have | Spent |
| --- | --- | --- |
| hour ~4 | preflight green, `check_scene_geometry.py` passing | ~$4.50 |
| hour ~8 | first rendered run done | ~$9 |
| — | **reserve ~8 hours for the actual experiment** | ~$9 |

Past hour 8 with phase 2 still failing means the *approach* is wrong, not the config — stop
the instance and bring it back to the Mac. **The moment you are editing a `.py` file, stop
the instance**; nothing here needs a GPU to edit, and most failure modes are pure Python.

Running out before a result is information, not a disaster: it says setup is harder than the
reading suggested, and *where* it went is what to record before topping up.

## ▶ Runbook

```bash
python3 -m pytest -q                      # 32 tests, no AlpaSim/GPU
python3 scripts/preview_trajectory.py     # -> docs/preview.png
python3 scripts/preview_trajectory.py --speed 14 --obstacle-x 18 --hz 4

# on a box with AlpaSim (see docs/BOX_SETUP.md for the full sequence):
cd ~/alpasim && uv pip install -e ~/shield-in-alpasim --no-deps
uv run alpasim-info                       # expect `shielded` under alpasim.models
./scripts/preflight.sh                    # ordered cheapest-failure-first
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded wizard.log_dir=$PWD/out
```

**`--no-deps` is required.** Our `alpasim_*` dependencies are workspace packages that do not
exist on PyPI; AlpaSim's in-tree plugins resolve them via `[tool.uv.sources]`, but we are an
out-of-tree checkout, so a plain install goes to PyPI and fails. `uv sync --extra all` has
already provided them.

To actually arm the shield, the driver container needs the scene path — inside the
container, not on the host:

```bash
    services.driver.environments='["SHIELD_SCENE_USDZ=/mnt/nre-data/<sceneset>/<scene>.usdz"]'
```

The wizard defaults to `trafficsim: disabled`, which is what the ground-truth arm wants —
leave it alone rather than passing `trafficsim=catk`, or the on-disk geometry stops matching
the sim (see "Settled" above).

`preview_trajectory.py` injects an obstacle by hand to show the shield braking (stops at
15.8 m, obstacle surface at 20.5 m). With `SHIELD_SCENE_USDZ` unset the field under AlpaSim
is empty, so the car drives straight and the shield never fires — the designed fallback, not
a bug. **Check the driver log for `Loaded N scene actors`** before believing any
"no interventions" result: a shield with no geometry looks exactly like a shield with
nothing to avoid.

Videos come from AlpaSim's eval stage: `eval.video.video_layouts=[DEFAULT]` renders BEV +
camera + metrics per rollout, and output sorts clips into `collision_at_fault/`, `offroad/`.
That directory count is the real scoreboard.
