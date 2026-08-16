"""Tier 0 hero video: SegFormer's kept-actor mask across the fisheye surround, cycle by cycle.

For every debug cycle it segments the dumped front / cross-left / rear-left NuRec fisheye frames,
tints the kept dynamic-actor pixels (the shield's SemanticDepthMask classes) magenta, tiles the
three views into one row with a caption (camera + actor%), and stitches the sequence to an mp4.
This is the moving version of the Tier 0 check: the pink is exactly what the semantic filter keeps
as obstacle evidence, so you can watch it stay locked on the real vehicles through the drive.

Run in the alpasim venv (torch/transformers/PIL) with ffmpeg on PATH:
  uv run --project ~/alpasim python seg_video.py <debug_dir> <out.mp4>
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
ACTORS = list(range(11, 19))  # person..bicycle == SemanticDepthMask.CITYSCAPES_ACTORS
CAMS = [("camera_front_wide_120fov", "FRONT"),
        ("camera_cross_left_120fov", "CROSS-LEFT"),
        ("camera_rear_left_70fov", "REAR-LEFT")]
H = 360  # common tile height
FPS = 4

debug_dir, out_mp4 = sys.argv[1], sys.argv[2]
proc = AutoImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to("cuda").eval()


def overlay(fp):
    """Load a frame, segment, return an (H, W, 3) array with actor pixels tinted magenta + actor%."""
    img = Image.open(fp).convert("RGB")
    W0, H0 = img.size
    inp = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inp).logits
    up = torch.nn.functional.interpolate(logits, size=(H0, W0), mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()
    arr = np.array(img)
    m = np.isin(seg, ACTORS)
    arr[m] = (0.35 * arr[m] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    return arr, float(m.mean() * 100)


cycles = sorted(int(re.search(r"cyc_(\d+)_", os.path.basename(p)).group(1))
                for p in glob.glob(os.path.join(debug_dir, "cyc_*_%s.jpg" % CAMS[0][0])))
frame_dir = os.path.expanduser("~/seg_video_frames")
os.makedirs(frame_dir, exist_ok=True)
for f in glob.glob(os.path.join(frame_dir, "*.png")):
    os.remove(f)

for i in cycles:
    tiles = []
    for cam, label in CAMS:
        fp = os.path.join(debug_dir, "cyc_%04d_%s.jpg" % (i, cam))
        if not os.path.exists(fp):
            tiles.append(np.zeros((H, int(H * 16 / 9), 3), np.uint8))
            continue
        arr, pct = overlay(fp)
        im = Image.fromarray(arr)
        im = im.resize((int(im.width * H / im.height), H))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 210, 22], fill=(20, 20, 20))
        d.text((5, 5), "%s  actors %.1f%%" % (label, pct), fill=(255, 255, 255))
        tiles.append(np.array(im))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    row = np.concatenate(tiles, axis=1)
    banner = np.full((26, row.shape[1], 3), 245, np.uint8)
    bim = Image.fromarray(banner)
    ImageDraw.Draw(bim).text((6, 6), "shield semantic filter (SegFormer) on NuRec fisheye surround "
                             "- scene 02eadd92 - cycle %02d/%d" % (i, cycles[-1]), fill=(0, 0, 0))
    frame = np.concatenate([np.array(bim), row], axis=0)
    Image.fromarray(frame).save(os.path.join(frame_dir, "f_%04d.png" % i))

subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-pattern_type", "glob",
                "-i", os.path.join(frame_dir, "f_*.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4], check=True)
print("WROTE", out_mp4)
