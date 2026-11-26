"""Curvature agreement evaluation (mentor update, Nov 2026).

Measures how well the model's curvature prediction matches the ground-truth
road mask curvature on the val split of IDD and BDD.

For each image with has_seg=True we compute road_curvature() on:
  - the ground-truth road mask → gt_code
  - the model's predicted road mask → pred_code

Then we report:
  curvature_agreement_pct : fraction of images where pred_code == gt_code
  per_code_agreement       : per CURVATURE_NAMES agreement rate

Run standalone:
  python -m eval.eval_curvature --dataset data_processed/idd --split val
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.contract import CURVATURE_NAMES, SEG_IGNORE, collate_fn
from datasets.multitask_dataset import MultiTaskDataset
from models.postprocess import road_curvature


def evaluate_curvature(model, data_root="data_processed/idd", split="val", batch_size=4):
    """Evaluates curvature agreement between model predictions and GT road masks.

    Args:
        model: any wrapper with predict(imgs) -> dict containing optional "seg" key
               (B, H, W) int64, 0=non-road, 1=road.  If the model returns no "seg",
               this evaluator returns {}.
        data_root: path to dataset folder.
        split: 'train', 'val', or 'test'.
        batch_size: loader batch size.

    Returns:
        dict with 'curvature_agreement_pct' and per-code 'curvature_agree_<name>'.
    """
    ds = MultiTaskDataset(data_root, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    gt_codes = []
    pred_codes = []

    for batch in loader:
        has_seg = batch["has_seg"]   # bool tensor [B]
        gt_masks = batch["seg_mask"]  # int64 [B, H, W]; SEG_IGNORE where unlabelled

        # Only process images that have segmentation GT
        seg_indices = [i for i in range(len(has_seg)) if bool(has_seg[i])]
        if not seg_indices:
            continue

        imgs = batch["img"]
        res = model.predict(imgs)

        pred_seg = res.get("seg")
        if pred_seg is None:
            continue  # model doesn't do segmentation → skip

        if isinstance(pred_seg, torch.Tensor):
            pred_seg = pred_seg.cpu()
        else:
            pred_seg = torch.tensor(pred_seg)

        # Optional lean correction from model output
        lean_preds = res.get("lean")  # float [B] or None

        for i in seg_indices:
            gt_mask = gt_masks[i]  # [H, W], values 0/1/255
            # Build binary road mask from GT (ignore SEG_IGNORE pixels → 0)
            gt_road = (gt_mask == 1).long()

            pred_road = (pred_seg[i] == 1).long()

            lean_deg = 0.0
            if lean_preds is not None:
                lean_deg = float(lean_preds[i])

            gt_code, _ = road_curvature(gt_road, lean_deg=0.0)
            pred_code, _ = road_curvature(pred_road, lean_deg=lean_deg)

            gt_codes.append(gt_code)
            pred_codes.append(pred_code)

    if not gt_codes:
        return {}

    gt_arr = np.array(gt_codes, dtype=int)
    pred_arr = np.array(pred_codes, dtype=int)

    agreement = float(np.mean(gt_arr == pred_arr))
    metrics = {"curvature_agreement_pct": agreement}

    # Per-code breakdown
    num_codes = len(CURVATURE_NAMES)
    for code, name in enumerate(CURVATURE_NAMES):
        mask = gt_arr == code
        if mask.sum() > 0:
            metrics[f"curvature_agree_{name}"] = float(np.mean(pred_arr[mask] == code))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data_processed/idd",
                        help="Path to dataset root (default: data_processed/idd)")
    parser.add_argument("--split", default="val",
                        help="Split to evaluate (default: val)")
    args = parser.parse_args()

    # Dummy predictor: always predicts all-road mask (best-case seg, worst-case curvature)
    class DummyRoadPredictor:
        def predict(self, imgs):
            b, _, h, w = imgs.shape
            return {
                "boxes": None,
                "seg": torch.ones((b, h, w), dtype=torch.int64),
                "depth": None,
                "lean": None,
            }

    model = DummyRoadPredictor()
    metrics = evaluate_curvature(model, data_root=args.dataset, split=args.split)

    if not metrics:
        print("No segmentation labels found in this split.")
    else:
        print(f"Curvature evaluation on {args.dataset} ({args.split}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
