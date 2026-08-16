#!/usr/bin/env python3
"""Dump every camera's calibration (rig_to_camera + intrinsics) from a scene's rig config.

The transfuser driver config only hardcodes 4 cameras; the surround arm's rear-quarter blind
wedges (docs/MULTICAM_HANDOFF.md) want the extra rear cameras (rear_wide_120fov / rear_right_70fov)
whose calibration is NOT in that yaml. The canonical source is the ego sensor rig baked into the
scene `.usdz` (same `Artifact.rig` scene.py reads the ego vehicle_config from). This prints all of
them so we can paste the ones we need into SURROUND_RIG_TO_CAMERA + the surround config.

Box-only (needs alpasim_utils). Usage:
    uv run --project ~/alpasim python ~/shield-in-alpasim/scripts/dump_rig_cameras.py <scene.usdz>

The rig's camera API isn't documented here, so this introspects: it walks likely containers
(cameras / sensors / camera_configs / sensor_rig ...) and prints, per camera, the fields that look
like a pose (translation/rotation) and pinhole intrinsics (focal/principal/resolution). If nothing
matches, it dumps `dir(rig)` and the repr so the next run knows where to look.
"""

from __future__ import annotations

import json
import sys


def _attrs(obj):
    return [a for a in dir(obj) if not a.startswith("__")]


def _get(obj, *names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) else v
    return None


def _pose_intrinsics(cam) -> dict:
    """Best-effort pull of a camera's pose + pinhole intrinsics, tolerant of the exact API."""
    out: dict = {}
    r2c = _get(cam, "rig_to_camera", "extrinsics", "pose", "transform")
    if r2c is not None:
        out["rig_to_camera_raw"] = repr(r2c)
        t = _get(r2c, "translation_m", "translation", "t")
        q = _get(r2c, "rotation_xyzw", "quaternion", "rotation", "q")
        if t is not None:
            out["translation_m"] = [float(x) for x in list(t)]
        if q is not None:
            try:
                out["rotation_xyzw"] = [float(x) for x in list(q)]
            except TypeError:
                out["rotation_repr"] = repr(q)
    intr = _get(cam, "intrinsics", "opencv_pinhole", "pinhole", "camera_model")
    if intr is not None:
        out["intrinsics_raw"] = repr(intr)
        fl = _get(intr, "focal_length", "focal", "f")
        pp = _get(intr, "principal_point", "principal", "c")
        res = _get(intr, "resolution_hw", "resolution", "size_hw")
        if fl is not None:
            try:
                out["focal_length"] = [float(x) for x in list(fl)]
            except TypeError:
                out["focal_length"] = float(fl)
        if pp is not None:
            out["principal_point"] = [float(x) for x in list(pp)]
        if res is not None:
            out["resolution_hw"] = [int(x) for x in list(res)]
    return out


def _cameras(rig):
    """Yield (logical_id, camera_obj) from whatever container the rig uses."""
    for holder in ("cameras", "camera_configs", "sensors", "sensor_rig", "camera_rig", "rig"):
        c = _get(rig, holder)
        if c is None:
            continue
        # dict-like {logical_id: cam} or list of cams with a .logical_id
        if hasattr(c, "items"):
            for k, v in c.items():
                yield str(k), v
            return
        try:
            for cam in c:
                lid = _get(cam, "logical_id", "id", "name") or "?"
                yield str(lid), cam
            return
        except TypeError:
            continue


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    from alpasim_utils.artifact import Artifact

    artifact = Artifact(source=argv[0])
    rig = artifact.rig
    print("=== rig attrs ===")
    print(_attrs(rig))

    found = list(_cameras(rig))
    if not found:
        print("\n!! no camera container matched. rig repr:")
        print(repr(rig)[:2000])
        return 1

    result = {}
    for lid, cam in found:
        result[lid] = _pose_intrinsics(cam)
    print("\n=== cameras ===")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
