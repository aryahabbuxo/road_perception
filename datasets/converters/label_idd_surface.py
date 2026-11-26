"""Labels ~200 IDD val and test images with road surface types.
Classes (datasets/contract.py):
  0: asphalt
  1: concrete
  2: paving_stones
  3: unpaved

Only val and test stems are labeled; train stems are never included.
Saves to data_processed/idd/surface.json as {"<stem>": class_id}.
"""
import json
import os
import random
import cv2
import numpy as np

from datasets.contract import SEED, SURFACE_NAMES

IDD_DIR = "data_processed/idd"


def classify_road(img_bgr, mask):
    """Classifies road patch based on color, brightness and texture."""
    road = (mask == 1)
    if road.sum() < 200:
        return 0  # default asphalt

    pixels = img_bgr[road]
    b, g, r = pixels.mean(axis=0)
    std = float(pixels.std())
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    # Color difference for earthy/dirt tones (red/yellow >> blue)
    brown_ratio = (r + 0.1) / (b + 0.1)

    # Gradient / texture for paver blocks
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    edge_mag = float(np.abs(sobel[road]).mean()) if road.sum() > 0 else 0.0

    if brown_ratio > 1.35 and r > 65 and b < 85:
        return 3  # unpaved (dirt/mud/gravel shoulder)
    elif edge_mag > 3.8 and std > 28.0:
        return 2  # paving_stones
    elif brightness > 110.0 and abs(r - b) < 20 and abs(g - b) < 20:
        return 1  # concrete (light grey/whitish flat)
    else:
        return 0  # asphalt (dark/medium grey)


def main():
    splits_file = os.path.join(IDD_DIR, "splits.json")
    if not os.path.exists(splits_file):
        print(f"Error: {splits_file} not found.")
        return

    with open(splits_file) as f:
        splits = json.load(f)

    val_stems = splits.get("val", [])
    test_stems = splits.get("test", [])

    rng = random.Random(SEED)
    # Pick 100 from val and 100 from test
    selected_val = rng.sample(val_stems, min(100, len(val_stems)))
    selected_test = rng.sample(test_stems, min(100, len(test_stems)))
    selected_stems = selected_val + selected_test

    surface_labels = {}
    class_counts = {i: 0 for i in range(len(SURFACE_NAMES))}

    for stem in selected_stems:
        # Determine split
        split = "val" if stem in val_stems else "test"
        img_path = os.path.join(IDD_DIR, "images", split, f"{stem}.jpg")
        mask_path = os.path.join(IDD_DIR, "masks", split, f"{stem}.png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) if os.path.exists(mask_path) else None

        if img is not None and mask is not None:
            cls = classify_road(img, mask)
        else:
            cls = 0

        surface_labels[stem] = cls
        class_counts[cls] += 1

    out_path = os.path.join(IDD_DIR, "surface.json")
    with open(out_path, "w") as f:
        json.dump(surface_labels, f, indent=2)

    print(f"Wrote {len(surface_labels)} surface labels to {out_path}:")
    for i, name in enumerate(SURFACE_NAMES):
        print(f"  {name} ({i}): {class_counts[i]}")


if __name__ == "__main__":
    main()
