"""Converter for RTK dataset (Mendeley Data / Road Traversing Knowledge).

Classes (contract.py):
  asphalt -> 0
  paved -> 2
  unpaved -> 3

Rules:
- Keep every 10th frame (video frames).
- Split by video sequence (groups).
- Letterbox to 640x384.
- Folder: data_processed/rtk with images/, splits.json, surface.json.
"""
import argparse
import json
import os
import random
import cv2
import numpy as np

from datasets.contract import SEED
from datasets.converters.common import make_dirs, split_groups, write_splits
from datasets.transforms import letterbox

CLASS_MAP = {
    "asphalt": 0,
    "paved": 2,
    "cobble": 2,
    "unpaved": 3,
    "dirt": 3,
}


def parse_rtk_class(folder_or_name):
    s = str(folder_or_name).lower()
    for k, v in CLASS_MAP.items():
        if k in s:
            return v
    return 0


def generate_synthetic_rtk(out_root, num_seqs=12, frames_per_seq=20):
    """Generates synthetic video sequence frames if raw Mendeley archive is not present."""
    rng = random.Random(SEED)
    make_dirs(out_root, ["train", "val", "test"], ["images"])

    seq_classes = {}
    classes = [0, 2, 3]  # asphalt, paved, unpaved
    colors = {
        0: (45, 45, 45),    # asphalt
        2: (80, 85, 110),   # paved
        3: (45, 75, 115),   # unpaved (BGR)
    }

    groups = [f"seq_{i:02d}" for i in range(num_seqs)]
    for g in groups:
        seq_classes[g] = rng.choice(classes)

    splits_groups = split_groups(groups, rng, fracs=(0.70, 0.15, 0.15))
    splits = {"train": [], "val": [], "test": []}
    surface_labels = {}

    for split, seq_list in splits_groups.items():
        for seq in seq_list:
            cls_id = seq_classes[seq]
            base_col = colors[cls_id]
            for frame_idx in range(0, frames_per_seq * 10, 10):
                stem = f"{seq}_frame_{frame_idx:04d}"
                splits[split].append(stem)
                surface_labels[stem] = cls_id

                h, w = 288, 352
                img = np.full((h, w, 3), base_col, dtype=np.uint8)
                noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                out = letterbox(img)
                dst = os.path.join(out_root, "images", split, f"{stem}.jpg")
                cv2.imwrite(dst, out["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

    write_splits(out_root, splits)
    with open(os.path.join(out_root, "surface.json"), "w") as f:
        json.dump(surface_labels, f, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"Generated {total} synthetic RTK video frames across {num_seqs} sequences in {out_root}")


def convert_rtk(raw_dir, out_root):
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"RTK raw directory not found at '{raw_dir}'. "
            "Please download the RTK dataset from Mendeley Data and pass --raw-dir."
        )

    # Find sequences and frames
    seq_frames = {}
    seq_classes = {}

    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(root, f)
            seq_name = os.path.basename(root)
            cls_id = parse_rtk_class(root)
            seq_classes[seq_name] = cls_id
            seq_frames.setdefault(seq_name, []).append((path, f))

    if not seq_frames:
        raise ValueError(f"No image frames found in RTK raw directory '{raw_dir}'.")

    rng = random.Random(SEED)
    all_seqs = sorted(seq_frames.keys())
    splits_groups = split_groups(all_seqs, rng, fracs=(0.70, 0.15, 0.15))

    make_dirs(out_root, ["train", "val", "test"], ["images"])
    splits = {"train": [], "val": [], "test": []}
    surface_labels = {}

    for split, seq_list in splits_groups.items():
        for seq in seq_list:
            cls_id = seq_classes[seq]
            frames = sorted(seq_frames[seq], key=lambda x: x[1])
            # Keep every 10th frame
            sampled = frames[::10]
            for path, fname in sampled:
                stem = f"{seq}_{os.path.splitext(fname)[0]}"
                img = cv2.imread(path)
                if img is None:
                    continue

                out = letterbox(img)
                dst = os.path.join(out_root, "images", split, f"{stem}.jpg")
                cv2.imwrite(dst, out["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

                splits[split].append(stem)
                surface_labels[stem] = cls_id

    write_splits(out_root, splits)
    with open(os.path.join(out_root, "surface.json"), "w") as f:
        json.dump(surface_labels, f, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"Converted {total} RTK frames to {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="C:/Users/Kanishk/Downloads/RTK")
    parser.add_argument("--out", default="data_processed/rtk")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.sample:
        generate_synthetic_rtk(args.out)
    else:
        convert_rtk(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
