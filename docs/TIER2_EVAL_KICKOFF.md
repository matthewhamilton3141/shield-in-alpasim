# Next session kickoff — evaluate the shield-trained policy in AlpaSim

Paste-ready prompt for a fresh session (post-`/clear`). The Tier 2 *surrogate* work is done; this
is the one remaining, box-dependent step — and it needs an off-box integration build first, so
**do not burn GPU until that is written and unit-tested on CPU.**

> Continue shield-in-alpasim. Tier 0/1 (perception degradation) and the whole **Tier 2 safe-RL
> surrogate arc are DONE, merged, pushed** — do not redo them. This session: **evaluate the
> shield-trained RL policy inside AlpaSim's photoreal closed loop**, to answer two things: (1) does a
> policy raised under the shield in the kitti_nav surrogate *transfer* to AlpaSim, and (2) does the
> "crutch" finding hold in the real sim — i.e. is it safe only *with* the shield, and does the
> intervention-penalty "teacher" deploy more safely?
>
> **READ FIRST (do not act until you have context):**
> - `HANDOFF.md` — top banner is current state (Tier 2 arc complete; open user actions).
> - `docs/TIER2_PROBE.md` — the full Tier 2 write-up: feasibility (AlpaSim has no RL interface),
>   probe→scaled PPO→teacher-or-crutch (crutch: shield-off collision 0.94)→intervention-penalty
>   frontier (penalty 0.6 → 0.12). Figures in `results/rl_*.{png,csv}`, video `rl_tier2_preview.mp4`.
> - `src/shield_in_alpasim/rl_env.py` — `ShieldNavEnv`: the obs (ego v/steer/dist-to-goal +
>   nearest-K obstacle discs in the rig frame), the 15 discrete `(accel,steer)` actions, the
>   `shield=` / `intervention_penalty=` / `layouts=` knobs.
> - `src/shield_in_alpasim/driver.py` — `ShieldedDriver`, the AlpaSim decorator; `_obstacles_for`
>   already builds the live `CircleField` (GT or learned camera) each cycle.
>
> **CRITICAL — the eval is an INTEGRATION build, not just "run renders". Do it OFF-BOX, on CPU, first:**
> The shield-trained policy is a low-dim torch MLP over the *obstacle-field observation*, NOT a camera
> policy. To run it in AlpaSim you must, inside the driver:
>   1. **Checkpoint save/load.** `scripts/rl_scaled.py` / `rl_transfer.py` train and discard the net.
>      Add `torch.save` of a chosen policy (state_dict + obs_dim/arch), and a loader. Pick one
>      shield-trained (crutch) and one teacher (penalty 0.6) checkpoint.
>   2. **Build the `ShieldNavEnv` observation from the live field** each `predict()` cycle — reuse the
>      exact obs construction in `rl_env._observe` / `_obstacle_discs_rig` against the driver's
>      `_obstacles_for(...)` `CircleField` and the ego state. Factor that obs code so the env and the
>      driver share ONE implementation (don't fork it — a mismatch silently breaks transfer).
>   3. **Run the net → discrete action → track to waypoints.** Map the chosen `(accel,steer)` through
>      the existing `control.py` / `_rollout` path so it emits the waypoint `ModelPrediction` AlpaSim
>      expects. This becomes a new inner-policy mode (e.g. `$SHIELD_INNER_MODEL=rl_ckpt` +
>      `$SHIELD_RL_CKPT=<path>`), still wrapped by the shield exactly as today.
>   4. **Unit-test the whole chain on the Mac** (fake `PredictionInput`, a saved tiny checkpoint):
>      obs matches the env's, action is finite, waypoints are the right shape. `python3 -m pytest -q`
>      (111 pass today) must stay green.
>
> **THEN, and only then, provision a box and run the eval** (~$10–30):
>   - `HF_TOKEN=<inline, never persist> DATA_FS=$HOME/shield-data bash scripts/setup_box.sh`.
>   - Eval each checkpoint on the 10 curated NuRec scenes, obstacle field = **GT** and **learned
>     camera** (the Tier 1 seam, `SHIELD_OBSTACLE_SOURCE=camera`), shield **on** and **off**
>     (`SHIELD_*` toggles / a small ablation) → at-fault collision + progress. The 2×2 (shield on/off ×
>     GT/camera field) is the result: does the surrogate-trained policy drive in the photoreal sim,
>     and is it a crutch there too?
>   - Optional stronger training: use `ShieldNavEnv(layouts=...)` to train on obstacle fields
>     **sampled from the real scenes** (via `SceneObstacleSource` at a set of ego poses) so the
>     training distribution matches the eval, then re-eval.
>
> **Report a clear result + honest caveats before any big spend.** Terminate the box the moment you
> step away (Lambda has no auto-stop; ~$30–45/day idle).
>
> **Open user actions (from the prior session, may still be pending):** terminate the old idle A100
> if it's still up; re-merge `phase3-container-wiring` → `main` to bring the post-PR Tier 2 commits in.

## Why the off-box gate matters (the honest risk)

The GPU cost is small; the risk is spending it before the integration works. The policy transfers
only if the observation the driver feeds it is *identical* to the one it trained on — same
nearest-K ordering, same rig-frame convention, same feature scaling. That is pure-CPU work to get
right and unit-test, and it is where a silent bug would waste every render. Build and test the obs
bridge on the Mac; the box is only for the photoreal rollouts themselves.
