# Next session — surround-camera perception for the shield

Self-contained handoff to give the shield a multi-camera (surround) obstacle field instead of a
single front camera. Pick this up cold. The living session log is `../HANDOFF.md`; this is the
focused plan for one job.

---

## ✔ BOX-VERIFIED + BLIND-WEDGE DIAGNOSIS (2026-08-15) — surround works; the rig has rear-quarter gaps

Ran `driver=shielded_vavam_surround` on `02eadd92` (camera perception + GT debug dump). Results:

- **Steps 1–2 PROVEN.** `cameras delivered: [all 4]` every cycle (policy uses front only);
  `surround field` fuses all 4 cams (~400–520k pts each) into ~900–1500 discs; **no near-field
  phantom ring** (`cam_discs_within_1.5m == 0` every cycle). Nearest perceived obstacle swings
  `y=-7 … +11` across cycles — real lateral/rear coverage the front-only cone never had. BEV
  rendered (`out_smoke_surround_result/bev_*.gif`, keyframes PNG).
- **But surround SCORED WORSE on this one rollout: `at_fault=1.0, progress=0.638, fail`** (vs the
  front-only camera arm's earlier 0.00 at-fault on this scene). One stochastic rollout, but the
  cause is structural, not noise —
- **DIAGNOSIS (free, from the BEV dumps): the 4-camera rig has ~30°-wide BLIND WEDGES at each rear
  quarter (±115°…±145°), and 02eadd92's at-fault actor sits in one.** Coverage math: front (0°,120°)
  → −60..60; cross-left (+55°,120°) → −5..115; cross-right (−55°,120°) → −115..5; **rear is only
  70° (180°) → 145..215**. The 120° side cams reach just ±115°, the narrow rear cam starts at 145°,
  so ±115..145 is uncovered. Confirmed numerically: the nearest *unperceived* GT actors every cycle
  are at bearings −115,−122,−127,−135,−142,+130,+137,+143° — i.e. **in the wedges**. Not a
  calibration bug, not occlusion, not our code — the transfuser rig's camera set.
- **FIX PATH (next session, needs a re-render):** AlpaSim offers `camera_rear_wide_120fov` (one
  120° rear cam → 120..240, closes both wedges to a ~5° sliver) and `camera_rear_right_70fov`
  (the 6-cam presets, e.g. `wizard/configs/exp/presets/alpamayo2_comparison_6cam.yaml`, use
  rear_left+rear_right). **Their calibration is NOT in the transfuser yaml** — only the 4 cams are
  hardcoded there; the extra rear cams' `rig_to_camera`+intrinsics must come from the ego rig config
  (readable from the USDZ, same source `scene.py` uses for the ego rig). Add rear_wide (or
  rear_right) to `SURROUND_RIG_TO_CAMERA` + the surround config's cameras/extra_cameras, re-render,
  confirm the wedge actors go red. THEN the front-only-vs-surround A/B is worth running (n≥3).
- Box was stopped after this (diagnosis is free/off-box from here). Raw: box
  `~/alpasim/out_smoke_surround_clipgt-02eadd92-…/`, Mac `out_smoke_surround_result/`.

---

## ✔ OFF-METER CODE DONE (2026-08-14, Mac) — steps 1–4 built + unit-tested; box does step 5 only

The whole surround plumbing is written and green on the Mac (73 tests, +20 for this arm). No GPU
was touched — what remains is box **verification**, not building. What landed:

- **Multi-camera source** (`obstacle_source.py`): `MultiCameraObstacleSource` fuses N cameras'
  depth into one rig-frame field (per-camera back-project → `camera_to_rig` → union → one
  height-band + occupancy). `CameraCalib` holds per-camera intrinsics/extrinsics; `.surround(...)`
  builds the 4-cam rig from new `SURROUND_*` constants (the transfuser 4-camera calibration, which
  the surround config's `extra_cameras` also uses — one source). Tested: front+rear blobs land
  ahead **and** behind in one field (the win the front-only source can't get).
- **Camera plumbing** (`driver.py`): advertise ALL perception cameras (`camera_ids`), hand the
  inner policy only its subset. `SHIELD_POLICY_CAMERAS` (env) names the policy's cameras;
  `_policy_input` narrows `camera_images` before `inner.predict()` so VaVAM's `_validate_cameras`
  passes; `_build_obstacle_source` picks the surround source when >1 camera is advertised. `predict`
  logs **which cameras were actually delivered** each cycle — the step-1 smoke check, baked in.
- **Depth batching** (`depth.py`): `HFDepthModel(...)` now also accepts a *list* of frames → one
  batched forward pass. Off by default; `SHIELD_DEPTH_BATCH=1` (surround config passes it through)
  enables it — the cost lever for the 4× depth budget.
- **Config**: `driver=shielded_vavam_surround` (+`_configs`) — 4 cameras rendered/advertised, the
  4-cam `extra_cameras` block, `SHIELD_POLICY_CAMERAS=camera_front_wide_120fov`,
  `SHIELD_OBSTACLE_SOURCE` defaulting to `camera`.

**The box session is now just step 5 (verify + re-run), on ~$14:** rsync to the box, run
`driver=shielded_vavam_surround` on `02eadd92` with `SHIELD_DEBUG_DIR` set, pull the npz, and check
the BEV — **does the laterally-adjacent car now appear in red (perceived)?** That single frame is
the deliverable. First confirm the delivered-cameras log shows all four arriving (the real risk,
step 1); then the reduced A/B (§Budget). Re-verify each camera's frame convention on the box,
especially the rear cam. Everything below is the original plan, kept for the box-side specifics.

## Why (the motivating finding — evidence, not a hunch)

The camera-perception arm works but uses **only the front camera**, so the shield's perceived
obstacle field is a **forward cone**. It is blind to anything beside or behind the ego.

Proof, from the BEV debug dump of scene `02eadd92` (39 cycles, `_dump_debug` → `make_bev_video.py`):
across ~90% of cycles the *nearest* ground-truth obstacle within 6 m was to the **RIGHT / LEFT /
BEHIND**, and the camera perceived it in essentially **none** of them — it only ever saw obstacles
that were **AHEAD**. By cycles 27–31 a car was basically on top of the ego (contact) and the front
camera showed nothing. **The visible collision was with a laterally-adjacent car the front camera
cannot see.** So today's "0 at-fault" is partly hollow: the shield only guards the forward cone.

**Goal:** render side (and rear) cameras, fuse their metric depth into one rig-frame BEV occupancy
field, so the shield can brake for lateral/rear threats too. Then re-run the sweep and show the
side car now appearing in red (perceived) and the contact avoided.

## What already exists and works (don't rebuild)

- **Camera arm end to end** (`SHIELD_OBSTACLE_SOURCE=camera`): `depth.py` (`HFDepthModel`, Depth
  Anything V2 Metric Outdoor → metres), `obstacle_source.py` (`CameraObstacleSource`, single front
  cam), driver wiring in `driver.py:_build_obstacle_source`, BEV debug viz (`scripts/make_bev_video.py`
  + `$SHIELD_DEBUG_DIR`).
- **Frame convention VERIFIED on the box** (this is the thing that's easy to get wrong):
  AlpaSim's `rig_to_camera` stores the **camera's pose in the rig** → `p_rig = R @ p_cam + t`
  (`R = quat_xyzw_to_matrix(rotation_xyzw)`, `t = translation_m`). OpenCV camera axes: x right, y
  down, z forward. Output rig frame: x forward, y left, z up. This applies **per camera** — but
  re-verify each new camera on the box via the BEV (side obstacles should land to the sides, rear
  behind).
- **Servicer sends `(timestamp, image)` tuples**, not `CameraFrame` objects (`_frame_image` handles it).
- `transformers`/`torch` are already in the `alpasim-base` image; the metric-depth checkpoint is a
  public HF download into the mounted cache.
- Single-cam calibration is hardcoded (`FRONT_WIDE_*` in `obstacle_source.py`) and rescaled to the
  actual frame resolution (`_intrinsics_for`).

## The work, in order

**1. Deliver multi-camera frames to the driver — solve this first, it's the tricky one.**
AlpaSim only delivers `camera_images` for the cameras the driver *advertises* (`camera_ids` /
`inference.use_cameras`). But the inner policy (VaVAM) calls `_validate_cameras` and expects
**exactly its one** camera. So:
   - Advertise ALL perception cameras (config `inference.use_cameras = [front, cross_left,
     cross_right, rear]`) so AlpaSim renders + delivers them, and set `ShieldedDriver.camera_ids`
     to all of them.
   - Pass the inner policy **only its subset** of `camera_images` before `inner.predict()` (filter
     the dict to VaVAM's front cam), so VaVAM's validation passes. Add a "policy cameras" notion
     (env `SHIELD_POLICY_CAMERAS`, or derive from the inner model) distinct from "perception cameras".
   - `_build_inner_model` must hand the inner model only its own `camera_ids`, not all — otherwise
     VaVAM's `from_config`/validation chokes on the extras.
   - **Write a smoke test first:** a run that just logs which cameras arrive in `camera_images`,
     before touching perception, to confirm the extras are actually delivered.

**2. Config: a surround config** `configs/driver/shielded_vavam_surround.yaml` (+ `_configs`):
list all N cameras in `runtime.simulation_config.cameras` and their calibration in
`runtime.extra_cameras` (copy the 4-camera block below). Keep the existing shield env passthroughs.

**3. Generalize `CameraObstacleSource` → multi-camera.** Take a list of cameras, each
`(camera_id, intrinsics, extrinsics, ref_hw)`. In `field_for`: per camera, grab its frame → depth →
`backproject_depth` with *its* intrinsics → `camera_to_rig` with *its* `(R, t)` → collect points;
concatenate all cameras' rig-frame points; height-band once; `occupancy_to_circles` once on the
union. Add hardcoded calib constants for the 4 cameras (values below). The pure-numpy helpers are
unchanged — only the loop over cameras is new, and it stays unit-testable with synthetic depth.

**4. Depth cost:** N cameras = N forward passes/cycle → slower. Either batch the N frames through
`HFDepthModel` in one call (make it accept a list) or use the Small model and accept it. The
current run was already ~2 s/cycle with one camera.

**5. Verify + re-run.** Diagnostic run with `$SHIELD_DEBUG_DIR` → pull npz → `make_bev_video.py`;
confirm side/rear obstacles now appear in red at the right bearings. Then re-run
`shield_ab.sh SHIELD_OBSTACLE_SOURCE AB_VALUES="gt camera"` (or a 3-way front-only vs surround vs gt)
with `N_ROLLOUTS>=3`. Payoff: the lateral collision that front-only missed should now be
perceived (and, where preventable, avoided). Re-render `02eadd92` BEV to show it.

## The 4-camera calibration (from AlpaSim's transfuser config — paste into the surround config / constants)

`opencv_pinhole` for all four: `focal_length [1545, 1545]`, `principal_point [960, 560]`,
`resolution_hw [1080, 1920]`. `rig_to_camera` per camera (`translation_m`, `rotation_xyzw`):

| logical_id | translation_m | rotation_xyzw |
|---|---|---|
| camera_cross_left_120fov | [1.646354, 0.143369, 1.521469] | [0.679354, -0.207915, 0.215233, -0.670018] |
| camera_front_wide_120fov | [1.670100, -0.025875, 1.522623] | [0.509222, -0.503331, 0.495086, -0.492180] |
| camera_cross_right_120fov | [1.626168, -0.161517, 1.526269] | [0.205424, -0.674057, 0.676355, -0.214458] |
| camera_rear_left_70fov | [-0.486641, -0.000595, 1.486321] | [0.503851, 0.497823, -0.499723, -0.498582] |

(Note the front-wide values here differ slightly from the single-cam `FRONT_WIDE_*` constants
currently in `obstacle_source.py` — reconcile to one source. The rear cam is a 70° FOV; its
translation x is negative, i.e. behind the rig — good for confirming the transform on the box.)
Full block with distortion coeffs is in
`plugins/transfuser_driver/alpasim_transfuser/configs/driver/transfuser_configs.yaml` in
NVlabs/alpasim (re-clone to `/tmp/alpasim-src` if gone).

## Files to touch

- `src/shield_in_alpasim/obstacle_source.py` — multi-camera loop + per-camera calib constants.
- `src/shield_in_alpasim/driver.py` — advertise-all vs policy-subset cameras; multi-cam
  `_build_obstacle_source`; hand inner model only its cameras.
- `src/shield_in_alpasim/depth.py` — optional batched forward pass.
- `src/shield_in_alpasim/configs/driver/shielded_vavam_surround.yaml` (+ `_configs`).
- `scripts/make_bev_video.py` — already handles more discs; no change needed.

## Gotchas / honest caveats

- **Camera-delivery/validation puzzle (step 1) is the real risk** — smoke-test it before building
  perception on top.
- **Not every "collision" is preventable or the shield's fault.** Traffic is non-reactive replay
  (`trafficsim: disabled`); a logged actor that sideswipes the ego without yielding is not an
  at-fault, braking-preventable event. Distinguish "shield was blind" (fixable by surround cams)
  from "unavoidable given non-reactive traffic". The BEV viz is how you tell them apart.
- **Re-verify the frame convention per camera** on the box; the rear cam especially.
- **VaVAM is stochastic** — n≥10 for real numbers; single rollouts are noisy.
- Depth is N× slower with N cameras.

## Budget (~$14 in Brev as of 2026-08-14)

`shield-a100` is `a100-80gb.1x` at **~$1.98/hr**, billed only while *running* (stopped = cheap
storage). ~$14 ≈ **~7 GPU-hours** — and multicam is the pricey arm: 4 cameras = **4× the depth
forward passes per cycle**, so renders run materially slower than the single-cam ones.

**Plan the session to fit $14 — prove it works, defer the big sweep:**
1. Camera-delivery smoke test (which cameras arrive) — minutes, ~$1.
2. Get surround perception building; verify on **1–2 scenes** with `$SHIELD_DEBUG_DIR` → BEV — does
   the laterally-adjacent car on `02eadd92` now show up in **red** (perceived)? ~$2–3. That single
   BEV, side car now visible, is the deliverable that proves the fix.
3. A **reduced** A/B — 3–4 scenes, `N_ROLLOUTS=1`, video off — front-only vs surround, to see if
   the lateral collisions drop. ~$4–6.
   → still leaves a buffer.

**Do NOT start the full 8-scene × 3 surround sweep on $14** — at 4× depth cost it is ~$8–15 on its
own; top up first. Cost levers (already in the scripts): `eval.video.video_layouts=[]` during
sweeps, `docker container/network prune -f` between runs, fewer scenes/rollouts for a first pass,
and **`brev stop` the moment you're editing `.py`** (guest `shutdown` does not stop billing).
Watch the clock — the camera arm is slow, so a "quick" multicam sweep is not quick.

## Box ops

See `../HANDOFF.md` "Resume recipe" and the `brev-box-operational-notes` memory. Short version:
`brev start shield-a100 && brev refresh`; `ssh -F ~/.brev/ssh_config shield-a100`; run long jobs in
tmux; `docker container prune -f && docker network prune -f` between batches (networks leak);
`brev stop` when done (guest `shutdown` does NOT stop billing). Scenes for the 8-scene sweep and the
box-side helper scripts (`scene_sweep.sh`, `shield_ab.sh`, `make_bev_video.py`) are already there.
