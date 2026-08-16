# Tier 2 kickoff — safe RL via shielding

Paste-ready prompt for a fresh session (post-`/clear`). Goal: **train a driving policy *inside*
AlpaSim under the hard shield, so the shield guarantees no catastrophic crash during exploration.**
Research question: does shielded exploration learn faster / to a safer policy than unshielded?

> Start Tier 2 of shield-in-alpasim: **safe RL via shielding** — train a driving policy *inside*
> AlpaSim under the hard shield, so the shield vetoes unsafe actions during exploration. Research
> question: does shielded exploration learn faster / to a safer policy than unshielded?
>
> **READ FIRST (don't act until you have context):**
> - `HANDOFF.md` — top banner is current state. Tier 0 + Tier 1 are DONE at n=10, committed + pushed
>   on branch `phase3-container-wiring`. Headline: learned camera perception degrades the shield's
>   at-fault rate ~0.02→0.23 (order of magnitude) at flat progress; it fails only on the
>   collision-relevant obstacle. The box is **kept running** — connect to it, don't re-bring-up.
> - `docs/COMPUTE_PLAN.md` — Tier 2 plan + the feasibility gate.
> - `docs/RESULTS.md` — how the shield decorates VaVAM; obstacle field = GT or learned camera.
>
> **DO NOT BURN GPU UNTIL FEASIBILITY IS SCOPED.** The critical unknown: AlpaSim is an eval/
> validation harness — it may have **no RL training interface** (step / reward / reset), only the
> wizard closed-loop eval rollout. First task, mostly **OFF-BOX** (read upstream AlpaSim source in
> `~/alpasim/src` or a fresh clone):
>   1. Can you get an RL env out of it — per-step action injection, a reward signal, episode reset?
>      Is there any training loop, or only eval rollouts?
>   2. What is the **minimal trainable policy**? Training a camera policy from photoreal renders via
>      RL is millions of steps × seconds/render = likely infeasible. Prefer a low-dim policy over the
>      shield's obstacle-field / BEV state, or a tiny action head, so the probe is cheap.
>   3. How does the shield plug into training? It already filters actions in `ShieldedDriver`
>      (`src/shield_in_alpasim/driver.py`) — during RL it vetoes unsafe exploration → the "safe
>      exploration" guarantee. That's the whole point.
>
> **THEN design a tiny FEASIBILITY PROBE** (~$100–300 per COMPUTE_PLAN): confirm *something learns at
> all* in a sane step budget before committing thousands. Report a clear go/no-go before any large run.
>
> BOX: connect to the existing Lambda A100 (see HANDOFF banner); if it's gone,
> `HF_TOKEN=<inline, never persist> DATA_FS=$HOME/shield-data bash scripts/setup_box.sh`. Tests:
> `python3 -m pytest -q` (102 pass, no GPU).

## Why the feasibility gate matters (the honest risk)

AlpaSim was built to *validate* policies (closed-loop eval), not to *train* them. If it exposes no
step/reward/reset, Tier 2 needs either (a) a custom RL loop wrapping the runtime's gRPC services, or
(b) training a policy against a cheaper surrogate (e.g. the shield's BEV obstacle state) and only
*evaluating* in AlpaSim. Either is a real build. The whole tier is high-reward, high-risk — that's
why COMPUTE_PLAN gates it behind a small probe. Don't bet the credits before the probe says "learns."
