"""Surface classification evaluation and confusion matrix (mentor update, Nov 2026).

Computes surface accuracy, per-class recall, and confusion matrix over:
  0: asphalt
  1: concrete
  2: paving_stones
  3: unpaved

Evaluated separately on:
  - data_processed/streetsurface (test split)
  - data_processed/rtk (test split)
  - data_processed/idd (Indian test set: val/test stems with surface.json)
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.contract import NUM_SURFACE_CLASSES, SURFACE_NAMES, collate_fn
from datasets.multitask_dataset import MultiTaskDataset


def evaluate_surface(model, data_root="data_processed/idd", split="test", batch_size=4):
    """Evaluates surface classification accuracy and confusion matrix."""
    ds = MultiTaskDataset(data_root, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    y_true = []
    y_pred = []

    for batch in loader:
        imgs = batch["img"]
        surfaces = batch["surface"].numpy()  # [B] -1 = unknown

        # Only evaluate images that have ground truth surface labels
        valid_indices = [idx for idx, s in enumerate(surfaces) if 0 <= s < NUM_SURFACE_CLASSES]
        if not valid_indices:
            continue

        res = model.predict(imgs)
        pred_surface = res.get("surface")
        if pred_surface is None:
            continue

        if isinstance(pred_surface, torch.Tensor):
            pred_surface = pred_surface.detach().cpu().numpy()
        elif isinstance(pred_surface, (int, np.integer)):
            pred_surface = np.full(len(imgs), pred_surface)

        for idx in valid_indices:
            y_true.append(int(surfaces[idx]))
            y_pred.append(int(pred_surface[idx]))

    if not y_true:
        return {"surface_acc": 0.0, "confusion_matrix": np.zeros((NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES), dtype=int)}

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(np.mean(y_true == y_pred))

    # Build 4x4 confusion matrix
    cm = np.zeros((NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < NUM_SURFACE_CLASSES and 0 <= p < NUM_SURFACE_CLASSES:
            cm[t, p] += 1

    per_class_acc = {}
    for i, name in enumerate(SURFACE_NAMES):
        row_sum = cm[i].sum()
        per_class_acc[f"surface_acc_{name}"] = float(cm[i, i] / row_sum) if row_sum > 0 else float("nan")

    metrics = {
        "surface_acc": acc,
        **{k: v for k, v in per_class_acc.items() if not np.isnan(v)},
        "confusion_matrix": cm,
    }
    return metrics


def print_confusion_matrix(cm, title="Surface Confusion Matrix"):
    print(f"\n=== {title} ===")
    label = "True \\ Pred"
    header = f"{label:<15}" + "".join(f"{name:>12}" for name in SURFACE_NAMES)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(SURFACE_NAMES):
        row_str = f"{name:<15}" + "".join(f"{cm[i, j]:>12d}" for j in range(NUM_SURFACE_CLASSES))
        print(row_str)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_processed/idd")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    class ConstantPredictor:
        def predict(self, imgs):
            b = imgs.shape[0]
            return {"boxes": None, "seg": None, "depth": None, "surface": np.zeros(b, dtype=int)}

    model = ConstantPredictor()
    metrics = evaluate_surface(model, data_root=args.data, split=args.split)
    print(f"Surface accuracy on {args.data} ({args.split}): {metrics['surface_acc']:.4f}")
    print_confusion_matrix(metrics["confusion_matrix"], f"{args.data} ({args.split})")
