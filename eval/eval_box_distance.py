"""Per-object distance evaluation on KITTI (mentor update, Nov 2026).

Compares model predicted box_distance against the median real distance in the same
bounding box region using the ground-truth LiDAR depth map:
  models.postprocess.box_distances(gt_depth, pred_boxes)

Reports MAE and AbsRel per distance band:
  - box_dist_mae_0_10, box_dist_mae_10_25, box_dist_mae_25_50
  - box_dist_absrel
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.contract import collate_fn
from datasets.multitask_dataset import MultiTaskDataset
from models.postprocess import box_distances


def evaluate_box_distance(model, data_root="data_processed/kitti", split="test", batch_size=4):
    """Evaluates per-object distance estimation on KITTI ground-truth depth."""
    ds = MultiTaskDataset(data_root, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    diffs_by_band = {
        "0_10": [],
        "10_25": [],
        "25_50": [],
    }
    all_absrel = []

    for batch in loader:
        imgs = batch["img"]
        gt_depths = batch["depth"]  # [B, H, W] in metres
        B = imgs.shape[0]

        res = model.predict(imgs)
        pred_boxes_batch = res.get("boxes", [])
        pred_dist_batch = res.get("box_distance")

        # If model outputs depth map instead of box_distance directly, compute it
        pred_depth_batch = res.get("depth")

        for i in range(B):
            gt_d = gt_depths[i]  # [H, W] float
            if pred_boxes_batch is not None and len(pred_boxes_batch) > i:
                boxes = pred_boxes_batch[i]
            else:
                continue

            if boxes is None or len(boxes) == 0:
                continue

            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes)

            # Ground truth distance per box
            gt_box_d = box_distances(gt_d, boxes)

            # Predicted distance per box
            if pred_dist_batch is not None and len(pred_dist_batch) > i:
                p_dist = pred_dist_batch[i]
                if isinstance(p_dist, np.ndarray):
                    p_dist = torch.from_numpy(p_dist)
            elif pred_depth_batch is not None:
                p_depth = pred_depth_batch[i]
                if isinstance(p_depth, np.ndarray):
                    p_depth = torch.from_numpy(p_depth)
                p_dist = box_distances(p_depth, boxes)
            else:
                continue

            for p, g in zip(p_dist.tolist(), gt_box_d.tolist()):
                if g <= 0.0 or p <= 0.0:
                    continue
                err = abs(p - g)
                rel = err / g
                all_absrel.append(rel)

                if 0.0 <= g < 10.0:
                    diffs_by_band["0_10"].append(err)
                elif 10.0 <= g < 25.0:
                    diffs_by_band["10_25"].append(err)
                elif 25.0 <= g <= 50.0:
                    diffs_by_band["25_50"].append(err)

    metrics = {
        "box_dist_mae_0_10": float(np.mean(diffs_by_band["0_10"])) if diffs_by_band["0_10"] else 0.0,
        "box_dist_mae_10_25": float(np.mean(diffs_by_band["10_25"])) if diffs_by_band["10_25"] else 0.0,
        "box_dist_mae_25_50": float(np.mean(diffs_by_band["25_50"])) if diffs_by_band["25_50"] else 0.0,
        "box_dist_absrel": float(np.mean(all_absrel)) if all_absrel else 0.0,
    }
    return metrics


if __name__ == "__main__":
    from eval.depth_wrappers import StereoSGBMWrapper

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_processed/kitti")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    model = StereoSGBMWrapper(dataset_root=args.data, split=args.split)
    # Give StereoSGBMWrapper dummy box generator for test
    orig_pred = model.predict
    def pred_with_boxes(imgs, stems=None):
        out = orig_pred(imgs, stems=stems)
        b = len(imgs)
        # 1 test box in center of each image
        dummy_boxes = torch.tensor([[200.0, 150.0, 440.0, 300.0, 0.9, 2.0]])
        out["boxes"] = [dummy_boxes for _ in range(b)]
        return out
    model.predict = pred_with_boxes

    metrics = evaluate_box_distance(model, data_root=args.data, split=args.split)
    print("Per-object distance metrics on KITTI:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
