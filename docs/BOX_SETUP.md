# Box setup — getting a first AlpaSim run for the least money

The expensive resource is **wall-clock time on a GPU instance**, not compute. Every step
below is ordered so that anything that can fail cheaply fails before the meter starts. The
realistic cost of a first run is a few dollars of GPU time and several hours of setup, so
the goal of this document is to make the setup hours happen while as little as possible is
billing.

Read `HANDOFF.md` first for *what* we are running and why. This is only *how*.

**Provider: Brev.** We considered AWS (there were $100 of credits available) and chose Brev
anyway. The reasoning is worth keeping: AWS has **no hard spending cap**, budget alerts only
email you hours late, and billing continues silently once credits run out — so the tail risk
is a forgotten instance, unbounded. Brev's idle auto-stop addresses exactly that failure
mode, and the whole bring-up is expected to cost less than $30. Paying a little to remove an
unbounded tail was judged the better trade. The AWS credits remain available for something
that is a better fit.

---

## ▶ Do this now, off the meter

**Request access to the gated dataset.**
[nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
— approval is not instant, and every scene download fails with `GatedRepoError` without it.
Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

This is now the only long-lead blocker, and it is free. Do it before you provision anything,
or you will pay for an idle box while waiting on an approval email.

---

## Which instance

| GPU | VRAM | Rough $/hr | Verdict |
| --- | --- | --- | --- |
| **A100 40 GB** | 40 GB | ~$1.10 | **The pick** |
| H100 | 80 GB | ~$1.99 | Upgrade only if 40 GB turns out to be tight |
| A100 80 GB | 80 GB | ~$6.21 | No — you are paying for VRAM nothing here uses |

*Check current prices in the console; these were noted earlier and move.*

Our driver uses **zero VRAM** (`gpus: null`, `device: cpu`) — the shield is numpy. The whole
budget goes to the NuRec renderer plus physics. Note that the 48/96 GB figures in AlpaSim's
`docs/ONBOARDING.md` are for the *optional* FlashDreams renderer, not the default NuRec path
we are on.

**Requirements to check when picking an image:**

- **NVIDIA driver ≥ 570.x.** The NRE container is CUDA 12.8; an older driver fails with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. `scripts/preflight.sh` checks this first and tells
  you to reimage rather than debug it — that is the right call, since in-place driver
  upgrades are a reliable way to lose an hour.
- **~200 GB disk** for container images and scene artifacts.
- Docker and the NVIDIA Container Toolkit. Most Brev GPU images have these; if not, see
  AlpaSim's `docs/ONBOARDING.md`.

Skip `download_vavam_assets.sh` — that is for the VaVAM driver, and ours has no checkpoint.

---

## Not paying for an idle box

The failure mode that costs real money is a running instance you forgot about, not the work
itself. Three habits, in descending order of how much they actually protect you:

1. **Set the idle auto-stop when you create the instance.** This is the main reason we chose
   Brev over AWS — use it, do not leave it at "never". An hour of idle timeout is plenty.
2. **A dead-man's switch on the box**, as the first command after you connect:

   ```bash
   sudo shutdown -h +240      # hard stop in 4 hours, no matter what
   ```

   Self-enforcing, needs no provider configuration, and survives you closing the laptop and
   forgetting. Re-issue it when you want more time. Belt and braces with the auto-stop,
   because auto-stop only triggers on *idle* — a hung job looks busy.
3. **Stop the instance whenever you step away.** Not just when you finish.

Also: **storage usually keeps billing while an instance is stopped.** Check what a stopped
instance costs per day, and delete the instance (not just stop it) once you have a result
worth keeping elsewhere.

---

## The phase split that saves the most

**Most of the hours are setup, and setup needs no GPU.** Phases 1 and 2 below touch the GPU
only to read a driver version. If your provider offers a cheap CPU-only instance, do them
there and snapshot; otherwise just be aware that the GPU is idle through all of it, so this
is the part to be efficient about — and never the part to debug slowly.

The rendered run in phase 3 is the only genuinely GPU-bound step.

### Phase 1 — setup (CPU only, ~30–60 min, mostly downloads)

```bash
export HF_TOKEN=<token>

git clone https://github.com/NVlabs/alpasim.git ~/alpasim
git clone <your-fork>/kitti-nav.git           ~/kitti-nav
git clone <your-fork>/shield-in-alpasim.git   ~/shield-in-alpasim

cd ~/alpasim
./setup_local_env.sh          # installs the Rust toolchain if missing (utils_rs needs it)
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
would go to PyPI and fail. `--no-deps` is correct because `uv sync --extra all` has already
installed everything we actually need.

### Phase 2 — validate everything except rendering (CPU only, minutes)

```bash
cd ~/shield-in-alpasim
./scripts/preflight.sh
```

Checks, ordered by how cheap they are to fail: HF token → GPU driver version → disk →
docker/uv → checkouts → **is `shielded` registered** → our unit tests under the box's
Python → scene geometry.

Then fetch one scene and generate configs *without simulating*:

```bash
cd ~/alpasim
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
    wizard.run_method=NONE wizard.log_dir=$PWD/out_dryrun \
    scenes.limit_to_first_n=1
```

`wizard.run_method=NONE` validates entry points, Hydra config, plugin discovery and the
scene download — the four things most likely to be wrong — with the GPU idle.

Now the check that matters most:

```bash
cd ~/shield-in-alpasim
export SHIELD_SCENE_USDZ=$(find ~/alpasim -name '*.usdz' | head -1)
uv run --project ~/alpasim python scripts/check_scene_geometry.py
```

It replays the scene's **logged human drive** against the scene's **logged actors** and
asserts the ego never collides. A real human drive did not crash, so if our geometry says it
did, the bug is ours — a frame confusion, a quaternion convention, the wrong ego footprint.
It also prints the scene's real camera logical IDs, the actor count and the ego dimensions,
all of which feed the config. Pure CPU, no renderer, no cost. **Do not start a rendered run
until this passes.**

### Phase 3 — the first metered run (GPU)

```bash
cd ~/alpasim
uv run alpasim_wizard deploy=local topology=1gpu driver=shielded \
    wizard.log_dir=$PWD/out_first \
    scenes.limit_to_first_n=1 \
    runtime.simulation_config.n_rollouts=1
```

One scene, one rollout. Expect the first run to be dominated by pulling container images.

Leave `trafficsim` alone — it defaults to `disabled`, which is exactly what the ground-truth
arm needs. Passing `trafficsim=catk` makes actors reactive, and our on-disk geometry
silently stops matching the simulation (see `HANDOFF.md`, "Settled").

To actually arm the shield, set the **container-side** scene path on the host before the
wizard run. The driver container's mounts, editable install and env are baked into
`shielded_configs.yaml` (verified 2026-08-13; see HANDOFF.md phase-3), including
`SHIELD_SCENE_USDZ=${oc.env:SHIELD_SCENE_USDZ_IN_CONTAINER,}`, so all you set is:

```bash
export SHIELD_SCENE_USDZ_IN_CONTAINER=/mnt/nre-data/all-usdzs/<scene>.usdz
```

That path is **inside the container**, under the wizard's standard `/mnt/nre-data` mount —
not the host path. Export it in the shell that runs the wizard, because `${oc.env:...}`
resolves in the host wizard process. With it unset the driver still runs, but with an empty
obstacle field: the car coasts and the shield never fires. That is the designed fallback, and
it is also exactly what a misconfigured path looks like, so check the driver log for
`Loaded N scene actors` before believing a "no interventions" result.

### Phase 4 — the actual experiment

Only once phase 3 produces video. Videos come from the eval stage —
`eval.video.video_layouts=[DEFAULT]` renders BEV + camera + metrics per rollout, and the
output sorts clips into `collision_at_fault/` and `offroad/`. That directory count is the
scoreboard.

---

## Habits that save the most

1. **Set the auto-stop, and run `shutdown -h +240` on connect.** Everything else is rounding
   error next to a box left running overnight.
2. **Never debug Python on the meter.** Any failure that is not GPU-specific belongs in
   phase 2, or on the Mac. If you are editing a `.py` file while the GPU sits idle, stop the
   instance first.
3. **Snapshot once it works.** Re-creating a working environment is the second-largest time
   sink after driver problems.
4. **`limit_to_first_n=1` until the pipeline is proven.** Scene count multiplies everything.
5. **Check the driver log says `Loaded N scene actors`.** A shield with no geometry looks
   identical to a shield with nothing to avoid.
