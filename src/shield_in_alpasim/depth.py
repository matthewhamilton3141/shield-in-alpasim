"""Metric monocular depth for the camera obstacle source — the one box/GPU-only piece.

Wraps a HuggingFace depth-estimation pipeline (default: Depth Anything V2 **Metric Outdoor**,
which outputs metres, which is what a braking-distance shield needs — a relative-depth model
would give unit-less values and the certificate would be meaningless). Kept out of
`obstacle_source.py` so that module stays pure-numpy and testable off the box; this one imports
`torch`/`transformers` lazily and only runs where they exist (the driver container).

Callable: `HWC image -> (H, W) depth in metres` at the input resolution.

Sanity checks to run on the box the first time (logged by `CameraObstacleSource.field_for`):
a road scene's median depth should be tens of metres, not ~1 or ~1000 — if it is, the model is
not the metric variant, or its output needs a scale/units fix here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class HFDepthModel:
    """Metric depth from a HuggingFace `depth-estimation` pipeline. Loads the model once."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        import torch  # noqa: F401 — imported to resolve the device and for interpolate
        from transformers import pipeline

        self._torch = __import__("torch")
        dev = 0 if str(device).startswith("cuda") and self._torch.cuda.is_available() else -1
        logger.info("loading depth model %s on %s", model_name, "cuda" if dev == 0 else "cpu")
        self._pipe = pipeline("depth-estimation", model=model_name, device=dev)

    def __call__(self, image) -> np.ndarray:
        from PIL import Image

        arr = np.asarray(image)
        pil = Image.fromarray(arr.astype("uint8")) if arr.ndim == 3 else image
        out = self._pipe(pil)
        # The pipeline returns raw model output in `predicted_depth` (metric for the metric
        # variants); `depth` is a normalised PIL image for visualisation, not metric — use the
        # tensor. Resize to the input resolution (the model runs at its own internal size).
        pred = out["predicted_depth"]
        t = self._torch
        d = pred if hasattr(pred, "dim") else t.as_tensor(np.asarray(pred))
        d = d.detach().float().cpu()
        while d.dim() < 4:
            d = d.unsqueeze(0)
        h, w = arr.shape[:2]
        d = t.nn.functional.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return d.squeeze().numpy().astype(np.float32)
