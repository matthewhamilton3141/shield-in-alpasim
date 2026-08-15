# Compute plan — spending the Lambda $7,500

We spent the whole bring-up on ~$14 of Brev, rationing every render (n=1 rollouts, stop the box
between edits). $7,500 removes that entirely. This is how to turn it into results without wasting it.

## Lambda economics (read first — it differs from Brev)

- **A100 on-demand ≈ $1.10–1.30/hr.** $7,500 ≈ **~5,000 GPU-hours**. Our driver uses ~zero VRAM
  (numpy shield); the budget goes to the NuRec renderer + physics, so **A100 40 GB is the pick**,
  not 80 GB (see BOX_SETUP.md).
- **No stop-and-preserve.** Unlike Brev, Lambda instances only *run* or *terminate* (which wipes the
  disk). Two consequences: (1) no idle auto-stop safety net — an idle A100 burns **~$30/day**, so
  terminate the moment you step away; (2) put `data/nre-artifacts` + `data/drivers` on a **Lambda
  persistent filesystem** so re-provisioning doesn't re-download the gated scenes + VaVAM weights.
- **Money is not the constraint — wall-clock is.** A render is ~6–10 min (startup + rollout).
  1,500 renders ≈ ~200 GPU-hours ≈ ~8 days on one GPU, or ~1 day across 8 instances. **Parallelize
  across instances**; the credits easily cover it.
- **Check if the credits expire.** Promo credits usually do. If so, **front-load Tier 1** (a
  guaranteed result) before Tier 2 (speculative).

Rough unit cost: **~$0.15–0.20 per shielded render** (surround+depth), less for GT / front-only.

## The tiers

### Tier 0 — stand up + smoke ( < $50 )
`setup_box.sh` → phase-3 first render → re-validate the **semantic filter on real fisheye frames**
(the one thing the dead Brev GPU blocked — SegFormer on NuRec fisheye is unproven). If semantic
mislabels on the distorted frames, fall back to the corridor gate (already validated) and note it.

### Tier 1 — the degradation curve (the backbone, ~$500–1,000)
The project's reason for existing, done rigorously. Sweep the **perception ladder** and plot how the
shield's at-fault rate / progress degrade as perception gets more realistic:

| rung | obstacle field |
|---|---|
| GT geometry | scene actors (privileged) |
| GT + localization noise | actors through the noised ego pose |
| front camera | single-cam depth |
| surround (gated) | 5-cam ftheta 360° + corridor gate |
| surround (semantic) | + vehicle/pedestrian-only |

**5 rungs × ~30 scenes × n≥10 rollouts ≈ 1,500 renders ≈ ~$250–350.** Add adversarial conditions
(day / night / rain / glare / highway-lead) → ~3× → **~$700–1,000 with re-runs**. This is the paper
figure, with error bars instead of the n=1 noise we lived with. Front-load this.

### Tier 2 — shielded RL (the flagship, $1,000–4,000, gated)
Train a policy **inside AlpaSim under the shield** — the shield guarantees no catastrophic crash
during exploration, so "safe RL via shielding" on a photoreal camera sim. Research question: does
shielded exploration learn faster / to a safer policy than unshielded? This is the genuinely
compute-hungry, novel direction (RL needs millions of env steps, each a render).

**Gate it behind a feasibility probe (~$100–300):** a tiny-scale run to confirm it learns *at all*
in a sane step budget before committing thousands. High reward, real risk — don't bet the whole pot
on it before Tier 1 banks a result.

### Reserve (~15–20%, ~$1,000)
Infra toil is real — this session alone hit a GPU that wouldn't initialize, docker network-pool
exhaustion, and off-trajectory NuRec smearing. Keep a buffer for re-runs and dead ends.

## Suggested order
1. Tier 0 (stand up, validate semantic) — hours.
2. Tier 1 (degradation curve) — bank the rigorous result first. ~$1k.
3. Tier 2 feasibility probe — decide if the flagship is worth the big spend.
4. Tier 2 full run if the probe is promising.

Even Tier 1 + a Tier 2 probe is < $1.5k of the $7.5k — there's ample room to be ambitious once a
result is in hand.
