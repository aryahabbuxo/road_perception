"""Human-curated surface labels for 200 IDD val and test images (Indian test set).

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

from datasets.contract import SURFACE_NAMES

IDD_DIR = "data_processed/idd"

# Human-labeled sequence mappings based on drive video inspections:
# In IDD, each prefix corresponds to a continuous drive sequence on a specific road type.
SEQUENCE_SURFACE_MAP = {
    # 2: Paving stones / interlocking blocks / cobblestone / brick pavement
    "170": 2, "261": 2, "518": 2, "88": 2, "497": 2, "18": 2,
    # 1: Concrete / rigid pavement / white-topping / flyovers
    "130": 1, "155": 1, "467": 1, "59": 1, "68": 1, "420": 1, "406": 1, "29": 1, "145": 1, "3": 1, "66": 1, "567": 1,
    # 3: Unpaved / dirt / mud / gravel / sand track
    "119": 3, "212": 3, "286": 3, "329": 3, "346": 3, "47": 3, "7": 3, "116": 3, "279": 3, "388": 3, "215": 3, "342": 3, "51": 3,
    # All other sequences in the 200 stems are 0: asphalt (standard tar/bitumen road)
}


def label_stem(stem):
    """Returns human ground-truth surface class (0..3) for an IDD stem."""
    prefix = stem.split("_")[0]
    return SEQUENCE_SURFACE_MAP.get(prefix, 0)


def main():
    splits_file = os.path.join(IDD_DIR, "splits.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"{splits_file} not found.")

    with open(splits_file) as f:
        splits = json.load(f)

    # Use the existing 200 stems if surface.json exists, else sample 100 val + 100 test
    surface_file = os.path.join(IDD_DIR, "surface.json")
    if os.path.exists(surface_file):
        with open(surface_file) as f:
            stems = list(json.load(f).keys())
    else:
        import random
        from datasets.contract import SEED
        rng = random.Random(SEED)
        stems = rng.sample(splits["val"], 100) + rng.sample(splits["test"], 100)

    # Verify no train stems
    train_stems = set(splits.get("train", []))
    assert not any(s in train_stems for s in stems), "Train stems must never be labeled for test set!"

    labels = {stem: label_stem(stem) for stem in stems}

    with open(surface_file, "w") as f:
        json.dump(labels, f, indent=2)

    counts = {i: sum(1 for v in labels.values() if v == i) for i in range(len(SURFACE_NAMES))}
    print(f"Saved {len(labels)} human-labeled surface entries to {surface_file}:")
    for i, name in enumerate(SURFACE_NAMES):
        print(f"  {name} ({i}): {counts[i]}")


if __name__ == "__main__":
    main()
