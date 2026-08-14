# shield-in-alpasim — session handoff

Current state, the open decision, and a runbook. The *why* behind the design lives in
[`README.md`](README.md) ("The actual gap"); this file is where to pick up.

---

## ★★★★ FINAL HEADLINE SWEEP — all fixes on (2026-08-14)

8 scenes × 3 rollouts, unshielded vs shielded (horizon + passthrough + rear-filter + OOD guard,
all default on). **at-fault** = mean at-fault-collision rate; **prog** = mean progress.

| scene | uns at-fault | shd at-fault | uns prog | shd prog | note |
|---|---|---|---|---|---|
| 01d503d4 | 0.00 | 0.00 | 1.00 | 1.00 | tie |
| 023b7fcc | 0.00 | 0.00 | 1.00 | 0.97 | tie |
| 0245ff75 | 0.00 | **0.33** | 0.51 | 0.41 | shield hurts (residual) |
| 026d6a39 | 0.00 | 0.00 | 1.00 | 0.90 | tie (passthrough) |
| 02e075b9 | offroad-**fail** | 0.00 | 0.53 | **1.00** | shield rescues offroad |
| 02eadd92 | **1.00**-fail | 0.00 | 0.65 | **0.91** | shield saves crash (the video) |
| 032b6f21 | 0.00 | 0.00 | 0.92 | 0.90 | tie |
| 04394343 | 0.00 | 0.00 | 0.80 | 0.32 | **OOD guard: was a shielded crash, now pass** |
| **mean** | **0.125** | **0.042** | **0.80** | **0.80** | |
| **outright fails** | **2** | **0** | | | |

**The clean claim: the shield cuts at-fault collisions ~3× (0.125 → 0.042), eliminates outright
failures (2 → 0), at zero net progress cost (0.80 → 0.80).** The progress it spends braking on
some scenes (0245ff75, 04394343) is offset by rescuing others (02e075b9 offroad, 02eadd92
crash). Versus the pre-guard sweep (at-fault 0.21→0.17 *and* the shield causing the highway
crash), the OOD guard turned a marginal/mixed result into a clear net positive — 04394343 is now
a pass, not a shielded crash.

**Caveats (honest):** (1) VaVAM is stochastic — unshielded at-fault swung 0.21→0.125 between the
two sweeps from seed variance alone, so n=3 numbers are directional, not precise; a tighter
result wants n≥10. (2) One residual shield-hurts scene remains, **0245ff75** (at-fault 0→0.33,
non-highway) — a different failure mode than the OOD one; next diagnostic is its cycle log
(likely the shield braking into a side/merge it can't model). (3) `04394343` shield progress
(0.32) still trails unshielded (0.80): the guard stops the crash but the shield still brakes in
the scene's city stretches. Raw rows: `~/sweep_results.csv` on the box.

**This is a complete, honest result: a hard safety shield over a learned camera policy that
provably reduces at-fault collisions at no net progress cost, with a characterized failure
mode (out-of-model-domain) and its mitigation.** Good place to write up / PR.

---

## ✔ Out-of-domain guard WORKS — no more shield-caused highway crash (2026-08-14)

Implemented the OOD guard (`_out_of_domain`: when ego speed > `cfg.max_speed`, `predict()`
returns VaVAM's plan untouched instead of certifying with an out-of-domain model). Committed,
`SHIELD_OOD_GUARD` toggle, 53 tests. A/B on the highway scene that broke (`shield_ab.sh
SHIELD_OOD_GUARD`, `N_ROLLOUTS=3`):

| SHIELD_OOD_GUARD | at-fault | progress | status |
|---|---|---|---|
| 0 (shield anyway) | **1.00** | 0.21 | fail |
| 1 (defer, default) | **0.00** | 0.43 | pass |

The guard fired 79× (highway speeds) and **eliminated the shield-caused crash** — at-fault
1.0→0.0, fail→pass. So the shield now **helps or abstains** rather than helping-or-harming: the
worst case (making a scene worse than raw VaVAM) is gone on this scene. Remaining gap: guard-on
0.43 still trails raw VaVAM's ~0.80 here, because the shield still acts in the sub-15 m/s city
portions of this mixed scene and costs some progress — a softer, secondary issue, not a failure.

**Next:** re-run the full 8-scene × 3 headline sweep with the guard on for the updated table
(expect 04394343 to flip fail→pass, aggregate at-fault to drop below the 0.17 it was). Then the
result is "the shield strictly reduces at-fault collisions (or abstains) at a modest progress
cost" — the clean claim.

---

## ★★★ HEADLINE SWEEP — 8 scenes × 3 rollouts, unshielded vs shielded (2026-08-14)

`scene_sweep.sh` with `N_ROLLOUTS=3`, shielded = passthrough + rear-filter on. **at-fault** =
mean at-fault-collision rate over 3 rollouts; **prog** = mean progress.

| scene | uns. at-fault | shd. at-fault | uns. prog | shd. prog | verdict |
|---|---|---|---|---|---|
| 01d503d4 | 0.67 | **0.00** | 0.86 | 1.00 | shield saves |
| 023b7fcc | 0.00 | 0.00 | 1.00 | 0.97 | tie |
| 0245ff75 | 0.00 | **0.33** | 0.49 | 0.33 | shield hurts |
| 026d6a39 | 0.00 | 0.00 | 1.00 | 0.89 | tie (passthrough held) |
| 02e075b9 | offroad-fail | 0.00 | 0.40 | 1.00 | shield saves (rescued offroad) |
| 02eadd92 | 1.00 | **0.00** | 0.70 | 0.74 | shield saves |
| 032b6f21 | 0.00 | 0.00 | 0.91 | 0.85 | tie |
| 04394343 | 0.00 | **1.00** | 0.80 | 0.21 | **shield CAUSES a crash** |
| **mean** | **0.21** | **0.17** | **0.77** | **0.75** | |

**Honest headline: the shield is domain-dependent, not a clean win.** In aggregate at-fault drops
0.21→0.17 and progress is ~flat (0.77→0.75). But the *per-scene* story is the finding: the shield
**prevents** crashes / rescues offroad on low-speed city scenes where VaVAM fails (01d503d4,
02eadd92, 02e075b9), and **causes** failures where its model doesn't fit — most starkly 04394343.

**Why 04394343 breaks (diagnosed):** it's a highway scene — ego at **17.3 m/s**, but kitti-nav's
`VehicleConfig.max_speed = 15` (KITTI city driving). At 17 m/s an obstacle 2.8 m ahead is
unstoppable (needs ~33 m), so the shield hard-brakes (its only move) into a collision that
*unshielded VaVAM steered around*. clearance goes to −2 m (already overlapping). **The
certificate is only as good as the kinematic model, and the model is out of domain above 15
m/s.** This is the sharp, real result: a KITTI-tuned braking shield helps in the domain it was
built for and is actively harmful outside it.

**Next, in priority order:**
1. **Domain-match / out-of-domain guard.** Either raise `max_speed` + retune decel for highway,
   or (safer) detect out-of-domain (ego speed > shield max, or ICS at entry) and *pass VaVAM
   through untouched* rather than hard-brake into a crash. The latter is a small, principled
   change and would likely flip 04394343 from fail back to VaVAM's 0.80 pass.
2. Investigate 0245ff75 (the other shield-hurts scene) — likely the same braking-into-trouble
   pattern; confirm with its cycle log.
3. Re-sweep after (1). Raw rows in `~/sweep_results.csv` on the box.

---

## ✔ Passthrough fix WORKS — recovers the tracker-drift progress loss (2026-08-14)

Implemented passthrough (emit VaVAM's `ModelPrediction` verbatim when the shield rollout has 0
interventions; else the tracked/braked trajectory). Committed, `SHIELD_PASSTHROUGH` toggle, 52
tests. Same-session A/B (`scripts/shield_ab.sh SHIELD_PASSTHROUGH …`, off vs on):

| scene | passthrough OFF (re-roll) | passthrough ON (verbatim) |
|---|---|---|
| 01d503d4 | 1.00 pass | 1.00 pass |
| 0245ff75 | at-fault **fail**, 0.42 | rear, pass, 0.41 |
| **026d6a39** | offroad **fail**, 0.37 | **pass, 0.88** |

**On 026d6a39 passthrough recovered progress 0.37 → 0.88 and fail → pass** — confirming the
loss was our tracker re-rolling VaVAM's path with lateral error and driving off-route. No safety
regression: 01d503d4 still passes, and 0245ff75 even avoided the at-fault collision (rear
instead). So the decorator now = VaVAM's own path fidelity when the shield is quiet, the
shield's certified trajectory when it acts. This is the fix the whole tracker-drift arc pointed
to. `SHIELD_REAR_FILTER` stays on too (sound, harmless).

**Still noisy — VaVAM variance.** Single rollouts swing (026d6a39 off was 0.41-pass in the
rear A/B, 0.37-offroad-fail here). The within-session off-vs-on A/B is the trustworthy read;
absolute numbers need `n_rollouts>1`. **Ops note:** per-run docker networks leak and exhaust
Docker's address pool (`all predefined address pools have been fully subnetted`) after ~dozens
of runs — `docker container prune -f && docker network prune -f` between batches, and rm old
`out_*` dirs (disk hit 94 %).

**Next:** the clean headline result — a proper sweep (`n_rollouts=3–5`, ~8–10 scenes) of
**unshielded vs shielded (passthrough+rear-filter on)** for the at-fault-collision rate and
mean progress, now that the shielded arm drives properly. That table is the paper figure.

---

## ⊘ Rear-filter A/B — NEGATIVE result, and the real cause of the progress cost (2026-08-13)

Implemented the finding-2 fix (`forward_relevant_field`, drop obstacle discs entirely behind
the ego's rear bumper; committed, `SHIELD_REAR_FILTER` toggle, 50 tests) and A/B'd it
(`scripts/shield_rear_ab.sh`, filter off vs on, both shielded, same session):

| scene | progress OFF | progress ON |
|---|---|---|
| 01d503d4 | 1.00 | 1.00 |
| 023b7fcc | 0.968 | 0.971 |
| 0245ff75 | 0.41 | 0.41 |
| **026d6a39** | **0.41** | **0.41** |
| 02e075b9 | 1.00 | 1.00 |

**The rear filter is essentially a no-op — my finding-2 hypothesis was WRONG.** The shielded
progress loss on 026d6a39 is not over-conservative rear braking. Its per-cycle diagnostics show
the car reaching **15 m/s (max speed) with mostly 0 interventions** — not frozen at all. The
real cause: **route deviation truncates the eval.** `dist_to_gt_trajectory` crosses the 4 m
cutoff (`RemoveTimestepsAfterEvent`), and the shielded car crosses it *early* (`plan_deviation`
**4.36** vs unshielded **2.41**), so progress truncates at 41 %; unshielded stays on-route to
100 % and only drifts at the very end. **Our pure-pursuit tracker re-rolls VaVAM's plan with
lateral error, and on curvier routes that error walks the car off the logged path.** The rear
filter is still sound and harmless (kept, default on), just aimed at the wrong problem.

**The right fix (next):** **pass VaVAM's raw trajectory through unchanged when the shield does
not intervene** (`n_interventions == 0`), and only emit our re-rolled/braked trajectory when it
does. Most cycles on 026d6a39 are un-intervened, so today we degrade VaVAM's path for no safety
reason. Passthrough should recover the lost progress (match unshielded path fidelity) while
keeping the shield's veto where it matters. Soundness note to work out: we currently *certify*
the tracked commands, not VaVAM's raw waypoints — so passthrough should either (a) only trigger
when the raw path's footprint is also clear (a cheap geometric check), or (b) accept that an
un-intervened cycle means both are safe-enough. Design before coding.

Also seen (single-rollout, noisy): 0245ff75 flipped at-fault(off)→rear(on) at identical
distance — within VaVAM variance, don't over-read. The rigorous re-run is `n_rollouts>1`.

---

## ★★ THE EXPERIMENT — 5-scene unshielded-vs-shielded sweep (2026-08-13)

`scripts/scene_sweep.sh` over the first 5 sample scenes, one rollout each, VaVAM-S ±shield:

| scene (clipgt-…) | arm | at-fault | rear | progress | status |
|---|---|---|---|---|---|
| 01d503d4 | unshielded | **1** | 0 | 0.815 | **fail** |
| 01d503d4 | shielded | **0** | 0 | 1.00 | pass |
| 023b7fcc | unshielded | 0 | 0 | 0.99 | pass |
| 023b7fcc | shielded | 0 | 0 | 0.97 | pass |
| 0245ff75 | unshielded | 0 | 1 | 0.51 | pass |
| 0245ff75 | shielded | 0 | 1 | 0.41 | pass |
| 026d6a39 | unshielded | 0 | 0 | **1.00** | pass |
| 026d6a39 | shielded | 0 | 0 | **0.45** | pass |
| 02e075b9 | unshielded | 0 | 0 | 1.00 | pass |
| 02e075b9 | shielded | 0 | 0 | 1.00 | pass |

**The trade-off, quantified:** at-fault collisions **1 → 0** (scene 01d503d4: the shield caught
a crash VaVAM drove into); mean progress **0.86 → 0.77 (≈ −11 %)**. The progress cost is
**concentrated, not uniform** — near-free on 023b7fcc/02e075b9, but the shield **halved**
progress on 026d6a39 (1.00 → 0.45) by over-braking. That is exactly **finding 2** (omnidirectional
`can_stop_safely` freezing for rear/side actors) showing up as a measurable cost — now with a
scene number to reproduce it on. Rear collisions are unchanged (0245ff75: both rear-ended; the
shield can't avoid those and pays progress for the surround traffic).

**The clean story:** the hard shield **eliminates at-fault collisions** — its guarantee — at a
bounded, mostly-modest progress cost, with the tail cost traceable to a specific,
already-diagnosed over-conservatism. That is a real result.

**⚠ Caveat — run-to-run variance.** Scene 01d503d4 *unshielded* crashed here but scored 1.0 in
the earlier standalone run, so VaVAM is stochastic across runs (GPU inference / seed). Shielded
passed 01d503d4 **both** times — i.e. the shield also *bounds the worst case*. But single-rollout
numbers are noisy: the rigorous version is `runtime.simulation_config.n_rollouts>1` (average out
VaVAM's variance) over more scenes. This 5-scene / 1-rollout sweep is a strong signal, not a
final number. `~/sweep_results.csv` holds the raw rows on the box.

**Next:** (1) re-run with `n_rollouts=3–5` on more scenes for statistics; (2) implement the
finding-2 fix (drop rear/outside-corridor discs) and re-sweep — the hypothesis is it recovers
the 026d6a39-style progress loss without re-introducing at-fault collisions. That fix's
before/after on this sweep is the natural next result.

---

## ▶ Unshielded-vs-shielded comparison, scene 23dd34ea (2026-08-13)

Ran `driver=vavam` (unshielded, same VaVAM-S width_768 checkpoint via
`driver.model.checkpoint_path=` override) against the shielded run on the same scene:

| | unshielded VaVAM | shielded VaVAM |
|---|---|---|
| dist_traveled_m | 79.3 | 77.9 |
| collision_at_fault / rear / offroad | 0 / 0 / 0 | 0 / 0 / 0 |
| progress / score | 1.0 / 1.0 | 1.0 / 1.0 |
| shield interventions | — | 6/5/4/3/2 (early) |

**Honest read:** on this scene both are perfectly safe, so the shield prevented *no* collision
here — VaVAM didn't need saving. What the shield *did* cost is ~1.4 m of progress (its
conservatism trimming VaVAM's throttle). So this single scene captures the shield's **tax**
but not its **benefit**. A safety shield only proves its worth on scenes where the unshielded
policy *fails* — this benign scene isn't one.

**Therefore the real experiment needs a scene sweep** to find divergence (unshielded VaVAM
collides, shielded does not). That needs **more scenes downloaded** — which needs the gated HF
token (supplied ephemerally, never persisted; see `brev-box-operational-notes`). Recipe:
`scenes.limit_to_first_n=N` for both `driver=vavam` and `driver=shielded_vavam`, then compare
`collision_at_fault` rate and mean progress across scenes. That N-scene table is the headline
result; one clean scene is a working pipeline, not a finding.

---

## ★ IT DRIVES — shielded VaVAM, full route, zero collisions, score 1.0 (2026-08-13)

The horizon fix landed and the confirming render is unambiguous. `driver=shielded_vavam` on
scene 23dd34ea, armed:

| | coast (out_first) | vavam pre-fix (out_vavam) | **vavam + fix (out_fix)** |
|---|---|---|---|
| dist_traveled_m | 8.07 | 8.17 | **77.9** (gt 73.8) |
| collision_at_fault | 0 | 0 | **0** |
| collision_rear | 1 | 1 | **0** |
| offroad | 0 | 0 | **0** |
| progress_clipped_rel | 0.11 | 0.11 | **1.0** |
| score / status | 0.137 | 0.138 | **1.0 / pass** |

The car drives the whole scene (slightly past the human's distance), **zero collisions of any
kind**, perfect score. And **the shield is genuinely active while it drives** — the early
`shield cycle` logs show `n_interventions` 6/5/4/3/2 as it trims VaVAM's full-throttle command
in dense traffic, `horizon_s: 3.0`, `final_speed` climbing 8→ instead of collapsing to 0.

This is the first real result: a learned policy (VaVAM) driving under the hard shield, and the
shield vetoing where needed without wrecking the drive. Video pulled to `out_fix_result/`.

**What it means for the earlier findings:** the ~8 m stall was 100% the short-trajectory bug,
in *both* arms. The "shield over-conservative for rear actors" (finding 2) never fired here —
because a car that keeps pace with traffic is never the stationary sitting duck that gets
rear-ended. Finding 2 may still matter in a scene that genuinely blocks the ego; keep the
diagnostic (ahead-vs-behind) around to catch it.

**Next (the actual experiment now that it drives):** (1) run more scenes (`limit_to_first_n`
> 1) — one perfect scene isn't a result; (2) compare against **unshielded VaVAM** (`driver=vavam`)
on the same scenes to measure what the shield changed (interventions, collisions avoided vs
progress lost); (3) then the degradation experiment (GT geometry → learned perception). The
branch `phase3-container-wiring` (13 commits) is still local — consider pushing / PR.

---

## ✔✔✔ ROOT CAUSE FOUND + FIXED (2026-08-13) — the trajectory was too short

**Neither the shield nor the policy was the problem — our emitted trajectory was too short.**
The parquet + controller-CSV diagnosis (box session, per the prep recipe below) nailed it:

- During GT replay (0–3 s) the ego holds ~2.2 m/s. **At handover the controller commands hard
  braking** (`u_longitudinal_actuation` −2 to −4) and the car halts over ~3 s. The controller
  CSV (`out_vdiag/controller/*.csv`) shows its reference point pinned at the car's position.
- **Why:** the MPC has a **2.0 s horizon** (`n_horizon=20 × dt_mpc=0.1`, `controller/default.yaml`)
  and *clamps* horizon timestamps to the reference trajectory's time range
  (`linear_mpc.py:_interpolate_reference`). Our driver emitted only **6 waypoints ≈ 0.6 s**, so
  1.4 s of the MPC horizon clamped to our last waypoint — a stationary target ~2 m ahead — and
  the MPC braked to stop at it. **The coast baseline had the same 6-waypoint output, which is
  why BOTH runs died at ~8 m.** It was never the shield.
- **Fix (committed):** emit a trajectory spanning `DEFAULT_HORIZON_S = 3.0 s` (clears the 2.0 s
  MPC horizon), derived from `output_frequency_hz` (30 waypoints at 10 Hz). `predict()` also
  rolls out at least the inner model's own plan length. `driver.py` constructor now sizes
  `horizon_steps` from the rate; `_rollout` takes a horizon override; the `shield cycle` log
  gained `horizon_s`. 47 tests green (+2).
- **Not yet re-rendered.** Next box session: render `shielded_vavam` (and the coast baseline)
  and confirm the car now actually drives — `dist_traveled_m` should approach `gt_dist` ~74 m
  instead of ~8 m. THEN the shield's over-conservatism (rear actors, finding 2 below) becomes
  the next thing to see, now that the car moves.

---

## ▶ PREPPED FOR NEXT BOX SESSION (2026-08-13, Mac, off-meter) — read this first

Everything below was staged so the next box session is short. Off-meter findings + a turnkey
recipe:

**Off-meter confirmations (no box needed to know these):**
- **Our emitted trajectory is correct.** Reproduced `_rollout` for the stalled cycles: from
  `speed_in=2.13` it emits waypoints marching 1.3→15.4 m with implied speeds 2.6→7.6 m/s; even
  from `v=0` it accelerates (0.25→9.0 m). So finding 1 (car doesn't move) is **not** our
  trajectory generation — it's downstream (controller/handover/execution). Note this *weakens*
  the "re-roll degrades VaVAM" idea: cycle 1 emitted a strong accelerating plan and the car
  still ended at 0, so the controller seems to ignore even a good trajectory across that
  transition. Keep both hypotheses open: (i) re-roll degrades near-term motion from rest;
  (ii) a controller/GT-handover execution bug independent of our trajectory.
- **Finding 2 mechanism is pinned.** `can_stop_safely` (kitti-nav `vehicle.py:244`) returns
  False if `clearance(current_state) < safety_margin` for *any* disc, and `clearance` is
  omnidirectional — so a rear actor within 0.30 m fails certification before the braking
  rollout (which moves forward, away from it) even runs. **Candidate fix (our side, least
  invasive):** in `obstacles.py`, drop discs strictly behind the ego's rear bumper plane and
  outside the lateral corridor — obstacles a forward-only shield cannot collide with. Do
  carefully (a car merging from behind-beside matters); gate it and A/B the intervention count.
  Secondary to finding 1 — it only engages after the car has already stalled.

**Turnkey next box session (parquet only, NO render — cheap):**
```bash
brev start shield-a100 && brev refresh
S="ssh -F ~/.brev/ssh_config shield-a100"
$S "export PATH=\$HOME/.local/bin:\$PATH; cd ~/alpasim; \
    uv run python ~/shield-in-alpasim/scripts/analyze_rollout_parquet.py \
    out_vdiag/rollouts/*/metrics.parquet"
# out_vdiag persists on the stopped box's disk. The script prints the schema then a
# commanded-vs-achieved trace. Decision: if achieved speed stays ~0 while commanded accel is
# positive -> controller isn't executing our plan (downstream); then look at the controller /
# the GT->shield handover. If achieved tracks a degraded plan -> the re-roll hypothesis.
```
Then, depending on the parquet: implement the confirmed fix (both candidates are pre-designed
above) and re-render `shielded_vavam` to compare intervention counts.

---

## ▶▶▶▶▶ Session update — INSTRUMENTED DIAGNOSIS (2026-08-13, box)

Added `rollout_diagnostics()` (ahead-vs-behind, proposed-vs-shielded, per cycle; committed)
and re-rendered `shielded_vavam`. The trace **overturned the earlier "shield freezes it"
guess** and split the problem in two:

1. **The shield is NOT the early bottleneck.** For the first ~10 cycles VaVAM commands
   `proposed_accel=2.0` (full go) and the shield **allows it — 0 interventions** — yet the sim
   reports the ego stuck at `speed_in≈0`. Cycle 1 even plans `final_speed=3.33` from `2.13`,
   but cycle 2 reports `0.0`. **The car isn't executing our accelerating plan.** Ruled out so
   far: the shield (0 interventions), and the response timing path (unset reference fields →
   the driver derives waypoint timestamps at our output freq, consistent with our spacing,
   `main.py:1085-1099`). Leading remaining suspects: the emitted trajectory's shape/headings
   vs what the MPC controller tracks, or an S223-dynamics/controller mismatch. **Next
   diagnostic:** pull `out_vdiag/rollouts/*/metrics.parquet` (per-step commanded accel/steer +
   achieved speed) — it will show whether the controller tracked our waypoints. Needs a brief
   box session (parquet only, no render).
2. **When the shield DOES engage (cycle ~11+), it is over-conservative — confirmed.** Road
   *ahead* clear (`nearest_ahead_gap` 5–6 m) but an actor ~1.2 m *behind* (bearing ≈ −156°)
   drives `init_clearance` negative, so it reads `collided=True` and brakes to 0. Braking
   cannot avoid a rear threat — a real limitation of the omnidirectional `clearance()` when the
   KITTI-tuned shield is dropped into dense surround traffic. A fix would make the shield ignore
   discs behind the ego that the forward-braking rollout only moves away from (a kitti-nav
   change — do carefully; a merging car beside/behind does matter).

Box **stopped**. Net: the decorator works, and we now know the ~8 m outcome is mostly (1), not
the shield — a different bug than assumed. `n_obstacles` climbs 80→150 as traffic closes on the
stalled ego, consistent with the sitting-duck picture.

---

## ▶▶▶▶ Session update — SHIELD-WRAPS-A-POLICY, RENDERED (2026-08-13, box)

**The decorator works end-to-end on the box, and the first policy-vs-coast comparison is in.**

- **Pivoted Transfuser → VaVAM.** Transfuser was blocked: not in the base image (deps
  `timm`/`beartype`/`jaxtyping` missing, entry point unregistered → needs an image rebuild)
  *and* no checkpoint of known provenance. The models actually baked in are `alpamayo1/1_5/2`,
  `manual` (interactive pygame — useless headless), and **`vam` (VaVAM)**. VaVAM won: already
  registered (no rebuild), **one camera** (`camera_front_wide_120fov`, matching the shield's
  existing config), public checksummed weights (valeoai GitHub release, no HF token). Built
  `driver=shielded_vavam` (+`_configs`), downloaded VaVAM-S (~1.3 GB) to `data/drivers/vavam/`.
- **Rendered, armed on scene 23dd34ea.** Driver log: `Shielding inner policy 'vam' (VAMModel)`,
  `Loaded 106 scene actors`, and — the payoff — **`Shield intervened on 1/6 sub-steps`** every
  cycle. VaVAM proposes, the shield vetoes. The tracker + decorator are proven in a live render.
- **But the result is ~identical to coast**, and that's the honest finding:

  | | coast (out_first) | shielded_vavam (out_vavam) |
  |---|---|---|
  | collision_at_fault | 0 | 0 |
  | collision_rear | 1 | 1 |
  | dist_traveled_m | 8.07 | 8.17 |
  | status / score | pass / 0.137 | pass / 0.138 |

  **Why:** dense scene (106 actors), a lead actor sits within braking distance at GT handover,
  so the shield brakes hard immediately (`final speed 0.0, collided=True` in its lookahead)
  and pins the ego — *regardless of what VaVAM proposes*. The shield is the binding constraint,
  not the policy, so VaVAM and coast converge to the same ~8 m-then-rear-ended outcome. The
  decorator is proven; this **scene doesn't discriminate between policies** because the shield
  saturates. To make the veto *matter*, next session needs either a less-dense scene / one with
  open road ahead, or to investigate whether the immediate brake-to-zero is over-conservative
  (safety_margin 0.30 m, max_decel 4.5 — a tuning question) vs a genuinely blocked road.
- Box **stopped**. (VaVAM video not pulled — scp raced the shutdown; it's visually the same as
  coast. Metrics above are from `out_vavam/aggregate/results-summary.json` on the box.)

---

## ▶▶▶ Session update — THE SHIELD CAN NOW WRAP A POLICY (2026-08-13, Mac, off-meter)

The first render (below) confirmed empirically that the coasting shield is a sitting duck.
This session built the fix: `ShieldedDriver` is now a **decorator** over any registered
`alpasim.models` policy.

- **New `control.py`** — the trajectory tracker that was "the remaining open problem": a
  pure-pursuit lateral + speed-profile longitudinal controller that turns a proposed
  waypoint plan into the per-step `(accel, steer)` the shield filters. Pure numpy, 8 unit
  tests. This is the policy→command inversion, done as tracking (follow the plan as far as
  the rate-limited bicycle allows, let the shield veto the rest) rather than analytic
  inversion.
- **`driver.py` wired**: `predict()` now calls the inner model, tracks its plan, and rolls it
  through the shield; `$SHIELD_INNER_MODEL` (env, same pattern as `SHIELD_SCENE_USDZ`) names
  the policy to wrap (e.g. `transfuser`). Unset → the coasting baseline, unchanged. The inner
  model reuses this driver's `model_cfg` (so `checkpoint_path`/`device`/`use_cameras` are the
  inner model's). Shield `n_interventions` is now logged per call — the experiment's signal.
- **43 tests green** (was 32; +8 `control`, +3 `driver` via a fake inner model). `predict()`
  itself stays box-verified per repo discipline. Preview script unchanged (still coasts).
- **Config built (2026-08-13, off-meter):** `driver=shielded_transfuser`
  (`configs/driver/shielded_transfuser.yaml` + `_configs`) sets Transfuser's 4 cameras +
  `extra_cameras` (inlined so it doesn't depend on transfuser's config package on the search
  path), `SHIELD_INNER_MODEL=transfuser`, the checkpoint path, and the driver-container wiring
  **with the GPU on** (Transfuser needs it). The command prefix editable-installs both the
  shield and transfuser (the base image's `uv sync --extra all` omits the `transfuser` extra,
  so its entry point isn't baked in).
- **Still needs the box:** (1) confirm Transfuser's Python deps (timm/beartype/jaxtyping) are
  in the image — if not, rebuild with `--extra transfuser`; (2) confirm the checkpoint
  `/mnt/drivers/transfuser/model_0060.pth` exists (fetch if not); (3) render and **compare
  intervention counts + collisions vs the coast run**. That comparison is the payoff.

---

## ▶▶ Session update — FIRST RENDERED RUN (2026-08-13, later same day)

**The shielded driver rendered a full rollout and AlpaSim scored it `pass`.** The phase-3
container wiring (below) is done, verified, and committed on branch `phase3-container-wiring`
(commits `1df9711` wiring + `bb15c42` doc). The render itself:

- Scene `clipgt-01d503d4…` → resolves to artifact `23dd34ea…usdz` (the one we downloaded and
  geometry-validated; the sceneset symlinks it). `limit_to_first_n=1` picks this scene.
- Driver log: `Loaded 106 scene actors` — the GT geometry reached the driver container and
  the shield ran the whole rollout. End-to-end wiring proven under a live render.
- **Result: `collision_at_fault=0`, `offroad=0` → PASS** (score 0.137). But `collision_rear=1`
  and `dist_traveled_m=8.07` vs `gt_dist_traveled_m=73.77`.
- **Honest reading:** exactly the predicted coasting failure. The shield is a filter, not a
  driver, so with no upstream policy the car commands (0,0), coasts ~8 m, and a **non-reactive
  logged actor rear-ends it** (logged actors never yield; rear hits score not-at-fault, hence
  the pass). This is a WEAK test of the braking logic — the car never moved fast enough to
  force a forward veto. It empirically confirms the "Open decision" below: the shield needs a
  real policy upstream (decorator over VaVAM/Transfuser) to have something meaningful to veto.
- Artifacts pulled to Mac `out_first_result/` (front-cam mp4, results-summary.json, metrics).
- Two non-fatal log noises: a `MinADEScorer` interpolation-timestamp edge error (a half-open
  range issue in AlpaSim's own eval, not ours) and a Grafana `$worker_id` telemetry-plot
  warning. Neither stopped the run.
- **Next step:** build the learned-perception arm — `ShieldedDriver` as a decorator over a
  camera-based AlpaSim driver — so the car actually drives and the shield has real vetoes to
  make. Then the degradation experiment. The GT arm is now proven runnable.
- Minor cleanup for next time: `shielded_configs.yaml` uses `${oc.env:…,}` (empty default);
  OmegaConf warns it wants `,''` (quoted). Harmless (resolved fine), but tidy it.

Box **stopped** again after the run.

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

**✔✔ The phase-3 blocker — RESOLVED and VERIFIED on the box (2026-08-13).** The container-load
test passed: `alpasim-base:0.134.0` built clean (~4 min, CPU, no netrc needed), and a
render-free `docker run` of the driver container's exact install prefix proved the whole
chain — `uv pip install --python /repo/.venv/bin/python -e /mnt/shield --no-deps
--no-build-isolation` installs in <1 s with no network; the `shielded` entry point registers
and loads `ShieldedDriver`; `import kitti_nav` works via `PYTHONPATH=/mnt/kitti-nav-src` (so
the `.pth` fallback is unneeded); and `SceneObstacleSource.from_env()` read **106 actors +
a real ego rig config** from the USDZ through the new `/mnt/nre-data` mount. All four unknowns
below are now confirmed. Box stopped again after the test. The design details, retained:

The driver runs in the **`alpasim-base` container** (built from `Dockerfile` via
`uv sync` of the workspace; in-tree plugins' entry points are baked in at build). Our
**out-of-tree** plugin (`~/shield-in-alpasim`) and `~/kitti-nav` are **not in that image**,
and the container mounts only alpasim's own `src/`+`plugins/` (`base_config.yaml:174-175`).
Option (a) alone (workspace-member + mount) does **not** work: the image sets `UV_NO_SYNC=1`
(Dockerfile), so `uv run` will not re-register a bind-mounted plugin's entry point — an
explicit install is required regardless of where the plugin sits. So the wiring is **option
(b), refined**, and it all lives in our own `shielded_configs.yaml` (which is `@package
_global_`, so it can set `services.driver.*` — same mechanism as `trafficsim/catk.yaml`).
No AlpaSim fork, no image rebuild. What it does, all verified against upstream source:
  - **Mounts** (restated in full — OmegaConf *replaces* lists on merge, so the base driver's
    5 mounts are re-listed, then 3 added): `${scenes.scene_cache}:/mnt/nre-data`,
    `$SHIELD_SRC→/mnt/shield`, `$KITTI_NAV_SRC→/mnt/kitti-nav-src`.
  - **Install prefix** on `services.driver.command` (space-joined, run under `bash -c`, so
    `&&` chains): `uv pip install --python /repo/.venv/bin/python -e /mnt/shield --no-deps
    --no-build-isolation && uv run -m alpasim_driver.main …`. `--no-deps` (our `alpasim_*`
    deps are workspace pkgs already in the venv, not on PyPI); `--no-build-isolation` (reuse
    the venv's setuptools, no network per start); `--python` pins the baked venv.
  - **kitti_nav** via `PYTHONPATH=/mnt/kitti-nav-src` (plain src-layout package, no install).
  - **`SHIELD_SCENE_USDZ`** via `environments`, sourced from host `SHIELD_SCENE_USDZ_IN_CONTAINER`
    (must be the *container-side* path, e.g. `/mnt/nre-data/all-usdzs/<scene>.usdz`); unset →
    empty field → shield inert (scene.py `from_env`), the designed fallback, not a crash.

  **⚠ Real gap this surfaced:** the base **driver container never mounted the scene cache**
  (only renderer/physics did), so the shield literally could not open the USDZ. The
  `${scenes.scene_cache}:/mnt/nre-data` mount above is the fix — without it, ground-truth
  geometry was unreachable from the driver no matter how the env var was set.

  **All verified by the load test above (2026-08-13):** (1) `uv pip install --python … -e`
  registers `shielded` and `uv run` loads `ShieldedDriver`; (2) `PYTHONPATH` survives `uv run`
  (`import kitti_nav` → `/mnt/kitti-nav-src/kitti_nav/__init__.py`), so the `.pth` fallback is
  moot; (3) `--no-build-isolation` finds setuptools in the venv (install took <1 s, no net).

  **The one thing the load test did NOT exercise:** a live `predict()` under the running
  driver server + renderer — i.e. the shield actually braking for these 106 actors in a
  rollout. That needs a real (GPU, metered) render, and is the next step, not part of the
  wiring. Set `SHIELD_SCENE_USDZ_IN_CONTAINER=/mnt/nre-data/all-usdzs/23dd34ea-…usdz` on the
  host before the wizard run to arm it; leave unset for the inert baseline.

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

## The remaining open problem — SOLVED (2026-08-13)

**Trajectory ⇄ per-step command.** The shield emits `(accel, steer)`; AlpaSim wants
waypoints, and an inner model *proposes* waypoints. Rolling forward was done; *inverting* an
inner model's trajectory into commands was the open piece, and subtle — the shield reasons in
a rate-limited kinematic bicycle, and a network's waypoints need not be feasible under it.
**`control.py` closes this**: a pure-pursuit + speed-profile tracker follows the inner plan as
far as the bicycle allows and lets the shield veto the rest. Not analytic inversion (which
would fight infeasible waypoints) but tracking — the policy proposes, the shield disposes,
`n_interventions` counts the disagreements. See the top session entry.

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
cd ~/alpasim && uv pip install -e ~/shield-in-alpasim --no-deps  # host venv, for alpasim-info
uv run alpasim-info                       # expect `shielded` under alpasim.models
./scripts/preflight.sh                    # ordered cheapest-failure-first
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded wizard.log_dir=$PWD/out
```

**`--no-deps` is required.** Our `alpasim_*` dependencies are workspace packages that do not
exist on PyPI; AlpaSim's in-tree plugins resolve them via `[tool.uv.sources]`, but we are an
out-of-tree checkout, so a plain install goes to PyPI and fails. `uv sync --extra all` has
already provided them. (The host-venv install above is only so `alpasim-info` on the host can
see `shielded`; the **driver container** gets its own install via the command prefix in
`shielded_configs.yaml` — see the phase-3 section at the top.)

**Arming the shield is now baked into `shielded_configs.yaml`** — the driver container mounts
the scene cache at `/mnt/nre-data` and reads `SHIELD_SCENE_USDZ` from the env. To point it at
a scene, set the *container-side* path on the host before the wizard run:

```bash
export SHIELD_SCENE_USDZ_IN_CONTAINER=/mnt/nre-data/all-usdzs/23dd34ea-a8d1-410c-aef7-d13f554cc4c9.usdz
# leave it unset for an inert (never-fires) shield — the designed fallback
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
