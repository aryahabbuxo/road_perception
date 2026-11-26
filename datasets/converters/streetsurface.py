"""Converter for StreetSurfaceVis dataset (Zenodo, 9,122 images).

Classes (contract.py):
  asphalt -> 0
  concrete -> 1
  paving stones and sett -> 2
  unpaved -> 3

Produces standard layout in data_processed/streetsurface/:
  images/<split>/<stem>.jpg (640x384 letterbox)
  splits.json
  surface.json {"<stem>": class_id}
"""
import argparse
import csv
import json
import os
import random
import cv2
import numpy as np

from datasets.contract import SEED, SURFACE_NAMES
from datasets.converters.common import make_dirs, split_groups, write_splits
from datasets.transforms import letterbox

CLASS_MAP = {
    "asphalt": 0,
    "concrete": 1,
    "paving_stones": 2,
    "paving stones": 2,
    "sett": 2,
    "cobblestone": 2,
    "unpaved": 3,
    "gravel": 3,
    "dirt": 3,
    "earth": 3,
}


def parse_label(label_str):
    lbl = str(label_str).strip().lower()
    for k, v in CLASS_MAP.items():
        if k in lbl:
            return v
    return None


def generate_synthetic_samples(out_root, count_per_class=30):
    """Creates a small representative set when raw Zenodo archive is not present."""
    rng = random.Random(SEED)
    stems = []
    surface_labels = {}

    colors = {
        0: (50, 50, 50),     # asphalt: dark grey
        1: (150, 150, 150), # concrete: light grey
        2: (90, 80, 110),   # paving stones: textured reddish/grey
        3: (40, 80, 120),   # unpaved: brownish/earthy (BGR)
    }

    make_dirs(out_root, ["train", "val", "test"], ["images"])

    for cls_id in range(4):
        base_color = colors[cls_id]
        for i in range(count_per_class):
            stem = f"ssvis_cls{cls_id}_{i:04d}"
            stems.append(stem)
            surface_labels[stem] = cls_id

    rng.shuffle(stems)
    n_train = int(len(stems) * 0.70)
    n_val = int(len(stems) * 0.15)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for split, split_stems in splits.items():
        for stem in split_stems:
            cls_id = surface_labels[stem]
            base_color = colors[cls_id]
            # Generate sample 1024x768 image with noise
            h, w = 768, 1024
            img = np.full((h, w, 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            out = letterbox(img)
            out_path = os.path.join(out_root, "images", split, f"{stem}.jpg")
            cv2.imwrite(out_path, out["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

    write_splits(out_root, splits)
    with open(os.path.join(out_root, "surface.json"), "w") as f:
        json.dump(surface_labels, f, indent=2)

    print(f"Generated {len(stems)} synthetic StreetSurfaceVis samples across 4 classes in {out_root}")


def convert_dataset(raw_dir, csv_file, out_root):
    if not os.path.exists(csv_file) or not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"StreetSurfaceVis raw files not found at raw_dir='{raw_dir}' or csv_file='{csv_file}'. "
            "Please download the 1024px version from Zenodo (DOI: 10.5281/zenodo.11449977) "
            "and pass --raw-dir and --csv."
        )

    records = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image_name") or row.get("filename") or row.get("image")
            lbl_str = row.get("surface_type") or row.get("surface") or row.get("class")
            cls_id = parse_label(lbl_str)
            if img_name and cls_id is not None:
                records.append((img_name, cls_id))

    if not records:
        raise ValueError(f"No valid records matching surface classes found in CSV '{csv_file}'.")

    rng = random.Random(SEED)
    rng.shuffle(records)

    n_train = int(len(records) * 0.70)
    n_val = int(len(records) * 0.15)
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }
    surface_labels = {}

    make_dirs(out_root, ["train", "val", "test"], ["images"])

    for idx, (img_name, cls_id) in enumerate(records):
        stem = os.path.splitext(os.path.basename(img_name))[0]
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        src_path = os.path.join(raw_dir, img_name)
        if not os.path.exists(src_path):
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue

        out = letterbox(img)
        dst_path = os.path.join(out_root, "images", split, f"{stem}.jpg")
        cv2.imwrite(dst_path, out["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

        splits[split].append(stem)
        surface_labels[stem] = cls_id

    write_splits(out_root, splits)
    with open(os.path.join(out_root, "surface.json"), "w") as f:
        json.dump(surface_labels, f, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"Converted {total} StreetSurfaceVis images to {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="C:/Users/Kanishk/Downloads/StreetSurfaceVis/1024")
    parser.add_argument("--csv", default="C:/Users/Kanishk/Downloads/StreetSurfaceVis/streetSurfaceVis_v1_0.csv")
    parser.add_argument("--out", default="data_processed/streetsurface")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.sample:
        generate_synthetic_samples(args.out)
    else:
        convert_dataset(args.raw_dir, args.csv, args.out)


if __name__ == "__main__":
    main()
