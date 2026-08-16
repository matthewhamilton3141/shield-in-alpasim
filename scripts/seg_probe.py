"""Tier 0 sanity probe: run the shield's SegFormer on the dumped NuRec fisheye frames and report
whether it actually labels vehicles/pedestrians on the distorted images (the one thing the dead Brev
GPU never got to). For each frame: a class histogram, the actor-pixel fraction, and a side-by-side
(original | colorized segmentation | original with actor pixels tinted magenta). If actors land on
the real cars and the histogram is dominated by road/building/vegetation, SegFormer works on fisheye.
Run in the alpasim venv (has torch/transformers): uv run --project ~/alpasim python seg_probe.py <jpgs...>
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
           "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
           "truck", "bus", "train", "motorcycle", "bicycle"]
ACTORS = set(range(11, 19))  # person..bicycle == SemanticDepthMask.CITYSCAPES_ACTORS
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)

proc = AutoImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to("cuda").eval()

outdir = os.path.expanduser("~/seg_probe_out")
os.makedirs(outdir, exist_ok=True)
for fp in sorted(sys.argv[1:]):
    img = Image.open(fp).convert("RGB")
    W, H = img.size
    inp = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inp).logits
    up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()

    ids, counts = np.unique(seg, return_counts=True)
    tot = float(seg.size)
    actor_frac = sum(int(c) for d, c in zip(ids, counts) if d in ACTORS) / tot
    print("\n== %s == actor%%=%.2f" % (os.path.basename(fp), actor_frac * 100))
    for d, c in sorted(zip(ids, counts), key=lambda x: -x[1])[:8]:
        tag = " <ACTOR" if d in ACTORS else ""
        print("   %-14s %5.1f%%%s" % (CLASSES[d], c / tot * 100, tag))

    color = PALETTE[seg]
    over = np.array(img).copy()
    m = np.isin(seg, list(ACTORS))
    over[m] = (0.35 * over[m] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    side = np.concatenate([np.array(img), color, over], axis=1)
    Image.fromarray(side).save(os.path.join(outdir, os.path.basename(fp).replace(".jpg", "_seg.png")))
print("\nDONE ->", outdir)
