# Tier 2 feasibility probe — safe RL via shielding: **GO**

Scoped 2026-08-16. The question the kickoff gated the whole tier behind: *can a policy learn
inside this system under the shield, cheaply enough to be worth the credits?* Answer: **yes, and
the mechanism is already demonstrable at toy scale.** No GPU was spent reaching this verdict.

## The feasibility finding (why the probe looks the way it does)

**AlpaSim exposes no RL training interface.** It is a gRPC closed-loop *eval* harness — services
(`egodriver / controller / physics / sensorsim / traffic`) orchestrated per rollout, scorers
(`collision`, `offroad`, `progress`, `safety`) computed *post-hoc*. There is no reward, no reset,
no per-step action injection (`grep` for `reward|gym|reset|ppo|train` across `~/alpasim/src` =
zero hits). The only action seam is `ShieldedDriver.predict()` returning a trajectory each cycle,
and each env step there is a NuRec render (~seconds) with a docker/service bring-up per episode
(~6–10 min). Training RL directly against the photoreal loop — millions of steps — is **infeasible
at any budget.**

**But the shield already ships its own fast training world.** `kitti_nav.vehicle` is a
self-contained pure-numpy kinematic sim with exactly the primitives AlpaSim lacks:

| RL primitive | kitti-nav |
| --- | --- |
| transition (step) | `step_state(state, accel, steer, cfg)` |
| the veto (safe exploration) | `safety_shield(...)` — least-restrictive certified action |
| reward / done / obstacles | `clearance`, `can_stop_safely`, `CircleField` |

It runs at ~10⁴ steps/s on one CPU core. So the feasible Tier 2, and the standard shielding-RL
setup: **train a low-dim policy in the kitti-nav surrogate *under the shield*, then evaluate the
learned policy in AlpaSim's photoreal loop** (AlpaSim's actual strength — and the eval ties back
to the Tier 1 finding that perception degrades the shield). Honest caveat to carry into any
write-up: *training is in the shield's kinematic surrogate, not the photoreal sim.*

## The probe

`ShieldNavEnv` (`src/shield_in_alpasim/rl_env.py`): a corridor-navigation task over the shield's
own model — thread 4 obstacle discs to a goal line without leaving the corridor; obs is low-dim
(ego speed/steer/dist-to-goal + nearest-4 discs in the rig frame); 15 discrete `(accel, steer)`
actions. One boolean is the whole experiment: `shield=True` routes every action through
`safety_shield` before the world sees it. A random-feature softmax policy trained with REINFORCE
(`scripts/rl_probe.py`, pure numpy, no torch, ~4 min/arm on a laptop). Two arms, identical except
the shield flag; held-out greedy + stochastic eval.

Load-bearing unit test (`tests/test_rl_env.py`): with the shield on and a certified start, the
agent **never collides across 40 random layouts even under a full-throttle-into-obstacles policy**;
the same reckless stream without the shield does collide. The safe-exploration guarantee is a test,
not a hope.

## Result (150 iters × 40 episodes/arm, ~700k env steps each)

![probe](../results/rl_probe.png)

| arm | return (initial → converged) | **collisions during training** | greedy goal-rate | eval collisions |
| --- | --- | --- | --- | --- |
| **shielded** | **7.4 → 17.7** (peak 22.9) | **0** | **0.30** | 0.00 |
| unshielded | 0.9 → 0.9 | **654** | 0.00 | 0.00 |

Three things, all pointing the same way:

1. **Something learns under the shield** — return rises ~3× to a policy that completes 30% of
   held-out layouts, crash-free (greedy and stochastic eval agree at ~0.30, so it's a real learned
   policy, not eval-mode luck).
2. **Shielded exploration is safe** — **0 collisions in 6,000 training episodes** vs the unshielded
   arm's **654**. Exactly the guarantee: from a certified start the shield's induction holds, so the
   agent provably never crashes *while learning*.
3. **Shielding did not cost learning — it enabled it.** The unshielded arm *collapsed*: around
   350k steps its return drops to ~0 and its collision count plateaus, because crash penalties
   terminate episodes and drown the forward-progress signal, so the policy flees to a do-nothing
   local optimum (goal-rate 0, but also crash-rate 0 — it learned to sit still). The shield removes
   that pathology by keeping every episode alive and on the road.

## Honest caveats

- **Surrogate, not photoreal.** This trains in the shield's kinematic world; the point is
  feasibility of the *learning*, not a finished driving policy. AlpaSim is the eval, not the trainer.
- **The unshielded baseline is deliberately vanilla.** Its collapse is partly the reward shape
  (a large collision penalty on episode termination); a tuned unshielded baseline (reward
  reshaping, penalty annealing) would likely learn *something*. The robust, defensible claim is the
  one the guarantee makes for free: **shielded exploration is crash-free during training and learns
  a competent policy at a sane budget** — not "unshielded RL can never learn."
- Tiny scale (700k steps, one seed/arm, a random-feature policy). This is a go/no-go probe, not the
  result. The scaled run below is where the numbers get error bars.

## Verdict: **GO** — recommended scaled run

The mechanism works and is cheap. Proposed flagship, still well inside the Tier 2 budget:

1. **Scale the surrogate run** — a real policy net (small MLP in torch), PPO, obstacle fields
   *sampled from the 10 curated NuRec scenes* (GT actor geometry) so training distribution matches
   the eval, ≥5 seeds/arm, shielded vs unshielded learning curves with error bars. Still CPU, still
   ~$0 — this is the paper's learning-curve figure.
2. **Evaluate the shield-trained policy in AlpaSim** closed-loop (a few dozen renders, ~$10–30) —
   does a policy raised under the shield transfer to the photoreal sim, and how does it fare when
   the shield's obstacle field comes from *learned camera perception* (the Tier 1 seam)? That
   closes the loop between the two tiers.

Reproduce: `python3 scripts/rl_probe.py` (writes `results/rl_probe.{csv,png}` + the verdict).
Tests: `python3 -m pytest -q` (110 pass, no GPU).
