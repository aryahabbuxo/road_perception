"""Converts BDD100K (10K subset) to our shared format.

- Detection categories mapped to classes 2-7:
    2: car
    3: two-wheeler (bicycle, motorcycle)
    4: bus or truck (bus, truck, train)
    5: person (pedestrian, rider)
    6: traffic light
    7: traffic sign
- Drivable maps mapped to road mask:
    1 (direct) and 2 (alternative) -> 1 (road)
    0 -> 0 (not road)
    255 -> 255 (ignore)
- Split by video ID (part before dash in filename) into train / val / test.

Run from repo root:
  python -m datasets.converters.bdd --images "<path>\\bdd100k\\10k" --drivable "<path>\\bdd100k_drivable_maps" --labels "<path_to_json_folder>" --out data_processed/bdd
"""
import argparse
import json
import os
import random
import cv2
import numpy as np

from datasets.contract import SEED
from datasets.converters.common import clean_boxes, make_dirs, save_sample, split_groups, write_splits
from datasets.transforms import letterbox

CATEGORY_MAP = {
    "car": 2,
    "bicycle": 3,
    "motorcycle": 3,
    "bus": 4,
    "truck": 4,
    "train": 4,
    "pedestrian": 5,
    "person": 5,
    "rider": 5,
    "traffic light": 6,
    "traffic sign": 7,
}


def load_bdd_labels(json_path):
    """Parses BDD100K detection JSON into {stem: (classes, boxes_yolo)}."""
    if not os.path.exists(json_path):
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_by_stem = {}
    for entry in data:
        fname = entry.get("name", "")
        stem = os.path.splitext(fname)[0]
        labels = entry.get("labels", [])

        classes = []
        boxes = []
        for lbl in labels:
            cat = lbl.get("category", "").lower()
            if cat not in CATEGORY_MAP:
                continue
            box2d = lbl.get("box2d")
            if not box2d:
                continue

            x1, y1 = float(box2d["x1"]), float(box2d["y1"])
            x2, y2 = float(box2d["x2"]), float(box2d["y2"])

            # Normalize to 1280x720 (standard BDD image size)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            classes.append(CATEGORY_MAP[cat])
            boxes.append([cx / 1280.0, cy / 720.0, w / 1280.0, h / 720.0])

        labels_by_stem[stem] = (
            np.array(classes, dtype=np.int64),
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
        )

    return labels_by_stem


def to_road_mask(drivable_img):
    """Converts BDD drivable map (0: background, 1: direct, 2: alternative) to road mask."""
    road = np.zeros(drivable_img.shape[:2], dtype=np.uint8)
    # Direct and alternative drivable lanes both count as road
    road[(drivable_img == 1) | (drivable_img == 2)] = 1
    road[drivable_img == 255] = 255
    return road


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="C:/Users/Kanishk/Downloads/bdd100k/10k")
    ap.add_argument("--drivable", default="C:/Users/Kanishk/Downloads/bdd100k_drivable_maps")
    ap.add_argument("--labels", default="C:/Users/Kanishk/Downloads/bdd100k_det_labels",
                    help="folder containing bdd100k_labels_images_train.json and _val.json (or det_train.json / det_val.json)")
    ap.add_argument("--out", default="data_processed/bdd")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Load detection JSONs
    train_json = os.path.join(args.labels, "bdd100k_labels_images_train.json")
    if not os.path.exists(train_json):
        train_json = os.path.join(args.labels, "det_train.json")
    val_json = os.path.join(args.labels, "bdd100k_labels_images_val.json")
    if not os.path.exists(val_json):
        val_json = os.path.join(args.labels, "det_val.json")

    labels_map = {}
    if os.path.exists(train_json):
        print(f"Loading {train_json}...")
        labels_map.update(load_bdd_labels(train_json))
    if os.path.exists(val_json):
        print(f"Loading {val_json}...")
        labels_map.update(load_bdd_labels(val_json))

    if not labels_map:
        print(f"Warning: No valid detection JSON found in {args.labels}.")
        print("Please ensure bdd100k_labels_images_train.json (or det_train.json) is present.")
        return

    # Find available 10k images
    found_stems = []
    image_paths = {}
    for split_dir in ["train", "val"]:
        p = os.path.join(args.images, split_dir)
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.endswith(".jpg"):
                    stem = os.path.splitext(f)[0]
                    if stem in labels_map:
                        found_stems.append(stem)
                        image_paths[stem] = os.path.join(p, f)

    print(f"Found {len(found_stems)} images with detection labels.")

    # Group by video ID (prefix before dash)
    video_groups = {}
    for stem in found_stems:
        vid = stem.split("-")[0] if "-" in stem else stem
        video_groups.setdefault(vid, []).append(stem)

    rng = random.Random(SEED)
    grouped_splits = split_groups(sorted(video_groups.keys()), rng, fracs=(0.70, 0.15, 0.15))
    chosen = {
        s: [item for vid in grouped_splits[s] for item in video_groups[vid]]
        for s in ("train", "val", "test")
    }

    make_dirs(args.out, chosen, ("images", "labels_det", "masks"))
    splits = {s: [] for s in chosen}

    drivable_train = os.path.join(args.drivable, "labels", "train")
    drivable_val = os.path.join(args.drivable, "labels", "val")

    for split, items in chosen.items():
        if args.limit:
            items = rng.sample(items, min(args.limit, len(items)))

        for i, stem in enumerate(items):
            img = cv2.imread(image_paths[stem])
            if img is None:
                continue

            # Check drivable map
            drivable_path = os.path.join(drivable_train, f"{stem}_drivable_id.png")
            if not os.path.exists(drivable_path):
                drivable_path = os.path.join(drivable_val, f"{stem}_drivable_id.png")

            road_mask = None
            if os.path.exists(drivable_path):
                d_img = cv2.imread(drivable_path, cv2.IMREAD_UNCHANGED)
                if d_img is not None:
                    road_mask = to_road_mask(d_img)

            raw_classes, raw_boxes = labels_map.get(stem, (np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)))

            out = letterbox(img, boxes=raw_boxes, seg_mask=road_mask)
            clean_cls, clean_bx = clean_boxes(raw_classes, out["boxes"])

            save_sample(args.out, split, stem, out["image"], clean_cls, clean_bx)
            if out.get("seg_mask") is not None:
                cv2.imwrite(os.path.join(args.out, "masks", split, stem + ".png"), out["seg_mask"])

            splits[split].append(stem)
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{len(items)}")

        print(f"{split}: {len(splits[split])} images converted.")

    write_splits(args.out, splits)
    print("BDD100k conversion complete ->", args.out)


if __name__ == "__main__":
    main()
