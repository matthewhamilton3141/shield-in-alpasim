# Box setup — getting a first AlpaSim run for the least money

The expensive resource is **wall-clock time on a GPU instance**, not compute. Every step
below is ordered so that anything that can fail cheaply fails before the meter starts. The
realistic cost of a first run is a few dollars of GPU time and several hours of setup, so
the goal of this document is to make the setup hours happen while nothing is billing.

Read `HANDOFF.md` first for *what* we are running and why. This is only *how*.

---

## ▶ Do these now, off the meter

These have the longest lead times and none of them need a machine. Nothing else matters if
these are not started.

1. **Request access to the gated dataset.**
   [nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
   — approval is not instant. Without it every scene download fails with `GatedRepoError`.
   Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

2. **Request an AWS GPU quota increase.** New accounts have a quota of **0** vCPUs for GPU
   instances, and the request can take from hours to a couple of days. Service Quotas →
   EC2 → *Running On-Demand G and VT instances* → request **8 vCPUs** (a `g6e.xlarge` or
   `g5.xlarge` is 4; asking for 8 leaves room to resize once). Do this before anything else
   — it is the single most likely thing to stall the whole plan.

3. **Confirm the credits apply to EC2 GPU instances** in your billing console, and note the
   expiry. Promotional credits sometimes exclude specific services.

---

## Which instance

AWS has no single-A100 instance — `p4d.24xlarge` is 8×A100 at roughly $32/hr and is not
worth considering. The realistic single-GPU choices:

| Instance | GPU | VRAM | Rough $/hr | Hours from $100 |
| --- | --- | --- | --- | --- |
| **`g6e.xlarge`** | L40S | **48 GB** | ~$1.86 | ~53 |
| `g5.xlarge` | A10G | 24 GB | ~$1.01 | ~99 |
| `g6.xlarge` | L4 | 24 GB | ~$0.80 | ~125 |

*Prices are on-demand, vary by region, and move — check the console rather than trusting
this table.*

**Take `g6e.xlarge`.** Our driver uses zero VRAM (`gpus: null`, `device: cpu`), so the whole
budget goes to the NuRec renderer, and we have never measured what that actually needs.
`HANDOFF.md` reasons that 40 GB is the safe pick and that the 48/96 GB figures in AlpaSim's
`docs/ONBOARDING.md` are for the *optional* FlashDreams renderer. 48 GB removes the question
entirely, and ~53 hours is far more than the handful this bring-up should take. Drop to
`g5.xlarge` only if the g6e quota is refused or unavailable in your region — 24 GB will
probably work, it is just an unknown we do not need to buy.

**Spot instances** are roughly 60–70% cheaper and are a good fit here: our runs are short and
restartable, and an interruption costs a re-run, not a corrupted result. Consider spot once
the first successful run is in hand — not during bring-up, where a mid-setup interruption
wastes more than it saves.

### Storage, the cost that does not stop

Budget **~200 GB** of `gp3` EBS. Two things to internalise:

- **EBS bills while the instance is stopped.** Roughly $16/month for 200 GB, so about
  $0.50/day whether or not you are using it. Stopping the instance saves the GPU cost, which
  is the overwhelming majority — do that every time you walk away.
- **Terminating an instance does not always delete its volume.** Check "delete on
  termination", or delete the volume by hand when finished, or it quietly eats credits for
  months.

### AMI

Use a **Deep Learning AMI (Ubuntu 22.04 or 24.04)** — it ships Docker, the NVIDIA Container
Toolkit and a recent driver, which is most of AlpaSim's prerequisite list already done.
**Verify the driver is ≥ 570.x** before anything else: the NRE container is CUDA 12.8 and an
older driver fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. `scripts/preflight.sh` checks
this first and tells you to reimage rather than debug it, which is the right call —
in-place driver upgrades on a DLAMI are a reliable way to lose an hour.

---

## On the box

### Phase 1 — setup (CPU only, ~30–60 min, mostly downloads)

```bash
export HF_TOKEN=<token>

git clone https://github.com/NVlabs/alpasim.git ~/alpasim
git clone <your-fork>/kitti-nav.git       ~/kitti-nav
git clone <your-fork>/shield-in-alpasim.git ~/shield-in-alpasim

cd ~/alpasim
./setup_local_env.sh          # installs the Rust toolchain if missing
uv sync --extra all
```

Then install our plugin. **`--no-deps` is required, not optional:**

```bash
cd ~/alpasim
uv pip install -e ~/shield-in-alpasim --no-deps
```

Our `pyproject.toml` declares `alpasim_plugins` and `alpasim_driver` as dependencies, but
those are *workspace* packages and do not exist on PyPI. AlpaSim's own in-tree plugins get
away with a plain install because they sit inside the workspace and resolve via
`[tool.uv.sources] … { workspace = true }`; we are an out-of-tree checkout, so resolution
would go to PyPI and fail. `--no-deps` is correct here because `uv sync --extra all` has
already installed everything we actually need.

*(Alternative, if you would rather not remember the flag: symlink the repo into
`~/alpasim/plugins/` so the workspace's `plugins/*` glob picks it up, and add a
`[tool.uv.sources]` block. Equivalent outcome, more moving parts.)*

### Phase 2 — validate everything except rendering (CPU only, minutes)

```bash
cd ~/shield-in-alpasim
./scripts/preflight.sh
```

Checks, in order of how cheap they are to fail: HF token → GPU driver version → disk →
docker/uv → checkouts → **is `shielded` registered** → our unit tests under the box's
Python → scene geometry.

Then download one scene and run the check that matters most:

```bash
cd ~/alpasim
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
    wizard.run_method=NONE wizard.log_dir=$PWD/out_dryrun \
    scenes.limit_to_first_n=1
```

`wizard.run_method=NONE` generates the configs and fetches artifacts **without running the
simulation**. This validates the entry points, the Hydra config, the plugin discovery and
the scene download — the four things most likely to be wrong — with the GPU idle.

Now point the geometry checker at the downloaded scene:

```bash
cd ~/shield-in-alpasim
export SHIELD_SCENE_USDZ=$(find ~/alpasim -name '*.usdz' | head -1)
uv run --project ~/alpasim python scripts/check_scene_geometry.py
```

This is the highest-value check in the whole document and it costs nothing. It replays the
scene's **logged human drive** against the scene's **logged actors** and asserts the ego
never collides. A real human drive did not crash, so if our geometry says it did, the bug is
ours — a frame confusion, a quaternion convention, the wrong ego footprint. It also prints
the scene's real camera logical IDs, the actor count and the ego dimensions, all of which
feed the config. **Do not start a rendered run until this passes.**

### Phase 3 — the first metered run (GPU)

```bash
cd ~/alpasim
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
    wizard.log_dir=$PWD/out_first \
    scenes.limit_to_first_n=1 \
    runtime.simulation_config.n_rollouts=1
```

One scene, one rollout. Expect the first run to be dominated by pulling container images.

Leave `trafficsim` alone — it defaults to `disabled`, which is exactly what the
ground-truth arm needs. Passing `trafficsim=catk` makes actors reactive, and our on-disk
geometry silently stops matching the simulation (see `HANDOFF.md`, "Settled").

To actually enable the shield, the driver container needs the scene path:

```bash
    services.driver.environments='["SHIELD_SCENE_USDZ=/mnt/nre-data/<sceneset>/<scene>.usdz"]'
```

That path is **inside the container**, under the wizard's standard `/mnt/nre-data` mount —
not the host path. With the variable unset the driver still runs, but with an empty obstacle
field: the car coasts and the shield never fires. That is the designed fallback, and it is
also exactly what a misconfigured path looks like, so check the driver log for
`Loaded N scene actors` before believing a "no interventions" result.

### Phase 4 — the actual experiment

Only once phase 3 produces video. Videos come from the eval stage —
`eval.video.video_layouts=[DEFAULT]` renders BEV + camera + metrics per rollout, and the
output sorts clips into `collision_at_fault/` and `offroad/`. That directory count is the
scoreboard.

---

## Habits that save the most

1. **Stop the instance whenever you step away.** GPU billing stops; EBS does not. This is
   where most of a credit budget silently goes.
2. **Never debug Python on the meter.** Every failure that is not GPU-specific belongs in
   phase 2, or on the Mac. If you find yourself editing a `.py` file with the GPU idle, stop
   the instance first.
3. **Snapshot once it works.** After the first successful run, make an AMI. Re-creating a
   working environment is the second-largest time sink after driver problems.
4. **`limit_to_first_n=1` until the pipeline is proven.** Scene count multiplies everything.
5. **Check the driver log says `Loaded N scene actors`.** A shield that never fires looks
   identical to a shield that was never given any geometry.
