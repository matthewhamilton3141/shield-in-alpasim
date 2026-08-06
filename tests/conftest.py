import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# kitti-nav is a sibling checkout: ../kitti-nav relative to this repo's root.
_kitti_nav_src = Path(__file__).resolve().parents[2] / "kitti-nav" / "src"
if _kitti_nav_src.is_dir():
    sys.path.insert(0, str(_kitti_nav_src))
