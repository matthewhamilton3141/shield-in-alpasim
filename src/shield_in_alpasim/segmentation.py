"""Semantic segmentation for the camera obstacle source — the GPU/box-only piece of the *better*
perception filter.

Wraps a HuggingFace semantic-segmentation model (default: SegFormer finetuned on Cityscapes, whose
classes include car / truck / bus / person / rider / bicycle / motorcycle as well as road / building
/ vegetation / sky). Returns a per-pixel label map; `SemanticDepthMask` (obstacle_source.py) turns
that into "keep only the dynamic-actor pixels" so the camera obstacle field becomes comparable to
the GT *actor* field instead of reconstructing the whole street.

Kept out of `obstacle_source.py` so that module stays pure-numpy and testable off the box; this one
imports `torch`/`transformers` lazily and only runs where they exist (the driver container).

Callable: `HWC image -> (H, W) int label ids` (at the model's output resolution; the caller resizes
to the depth resolution).

Box sanity check the first time: on a street frame the label histogram should be dominated by road /
building / vegetation, with car/person pixels where the vehicles and pedestrians are — if it's all
one class, the model or preprocessing is wrong.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


class HFSegmenter:
    """Semantic label map from a HuggingFace SegFormer (or compatible) model. Loads once."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self._torch = torch
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self._device = ("cuda" if str(device).startswith("cuda") and torch.cuda.is_available()
                        else "cpu")
        self._model.to(self._device).eval()
        logger.info("loading segmentation model %s on %s", model_name, self._device)

    def __call__(self, image) -> np.ndarray:
        from PIL import Image

        arr = np.asarray(image)
        pil = Image.fromarray(arr.astype("uint8")) if arr.ndim == 3 else image
        inputs = self._proc(images=pil, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits  # (1, C, h, w) at the model's output resolution
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
