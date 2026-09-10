"""Test-only adapter: exercise the real CLI with smaller synthetic images.

The production loader and model are unchanged. A generated-data marker is
mandatory, so this adapter cannot silently resize the user's actual OCT data.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, required=True)
parser.add_argument("script")
parser.add_argument("args", nargs=argparse.REMAINDER)
opts = parser.parse_args()
if not 16 <= opts.image_size <= 128:
    raise ValueError("Synthetic image_size must be between 16 and 128")
if "--data_dir" in opts.args:
    folder = Path(opts.args[opts.args.index("--data_dir") + 1])
    marker = folder / "SYNTHETIC_TEST_DATA.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("synthetic") is not True:
        raise RuntimeError("Small-image adapter requires generated synthetic data")
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src"))
import data
original_loader = data._load_oct2017
data._load_oct2017 = lambda cfg: original_loader(cfg, img_size=opts.image_size)
sys.argv = [opts.script, *opts.args]
runpy.run_path(opts.script, run_name="__main__")
