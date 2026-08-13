# Attribution

Everything this project builds on, with its license. Anything adapted from an upstream
source also carries a provenance header in the file that uses it.

## Repos this depends on

### AlpaSim
- **Source:** <https://github.com/NVlabs/alpasim>
- **Author:** NVIDIA Corporation.
- **License:** Apache 2.0.
- **Used for:** the closed-loop AV driver-plugin harness (renderer + physics + traffic
  microservices) this repo's plugin registers against. Not vendored — installed as a
  dependency per its own setup instructions.
- **Adapted from:** `src/shield_in_alpasim/configs/driver/*.yaml` follow the structure of
  AlpaSim's own `src/wizard/configs/driver/manual{,_configs}.yaml` — the closest stock
  driver to this one (CPU-only, no checkpoint). Field names and layout are AlpaSim's; the
  values are this repo's.

### kitti-nav
- **Source:** <https://github.com/matthewhamilton3141/kitti-nav>
- **Author:** Matthew Hamilton (this repo's author).
- **License:** MIT.
- **Used for:** the kinematic-bicycle vehicle model and the hard safety shield
  (`kitti_nav.vehicle.safety_shield`, `kitti_nav.dynamics.dynamic_safety_shield`) this
  plugin wraps. Imported as a sibling checkout, not copied — see `requirements.txt` /
  `conftest.py` for how it's put on the path.

### Alpamayo (reference only, not a runtime dependency yet)
- **Source:** <https://developer.nvidia.com/alpamayo>
- **Author:** NVIDIA Corporation.
- **License:** models distributed under OpenMDW-1.1 (see NVIDIA's model cards on Hugging
  Face for the exact terms per checkpoint).
- **Used for:** the stock reasoning-VLA driver this repo's shielded driver is eventually
  meant to be compared against inside AlpaSim. Not required to run this repo's plugin.
