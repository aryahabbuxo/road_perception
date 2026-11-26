"""Unified evaluation runner (owned by Person B).

Runs detection, segmentation, depth, surface, curvature, box-distance and FPS
benchmarking for a given model, and appends all results to results/metrics.csv
following Rule 6:
  date, run_name, model, dataset, split, metric_name, value

Supported --model choices:
  stereosgbm       StereoSGBM depth baseline (depth_wrappers)
  depthanything    Depth Anything V2 baseline (depth_wrappers)
  yolo             YOLO detection baseline   (models/wrappers.YoloBaseline)
  seg              SegFormer road baseline   (models/wrappers.SegBaseline)
  multitask        Combined MultiTaskNet     (models/wrappers.MultiTaskWrapper)

Rule 6 metric names (non-exhaustive):
  mAP50, mAP50_cls0 … mAP50_cls5
  mIoU, IoU_road
  depth_rmse_0_10, depth_rmse_10_25, depth_rmse_25_50, depth_absrel
  box_distance_mae_0_10, box_distance_mae_10_25, box_distance_absrel
  surface_acc, surface_acc_asphalt, …
  curvature_agreement_pct
  fps, latency_ms, params_M
"""
import argparse
import csv
from datetime import datetime
import os

METRICS_CSV = "results/metrics.csv"
CSV_COLUMNS = ["date", "run_name", "model", "dataset", "split", "metric_name", "value"]


def record_metric(date_str, run_name, model_name, dataset, split, metric_name, value):
    """Appends a single metric record to results/metrics.csv."""
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    file_exists = os.path.exists(METRICS_CSV) and os.path.getsize(METRICS_CSV) > 0

    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([date_str, run_name, model_name, dataset, split, metric_name, f"{value:.4f}"])


def _record_dict(metrics, date_str, run_name, model_name, dataset, split):
    """Write a whole metrics dict, skipping non-numeric values (e.g. confusion_matrix)."""
    import numpy as np
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            record_metric(date_str, run_name, model_name, dataset, split, k, float(v))


def _ds_name(path):
    return os.path.basename(os.path.normpath(path))


def evaluate_all(
    model,
    run_name="baseline",
    model_name="model",
    det_datasets=("data_processed/speedbump", "data_processed/rdd"),
    seg_datasets=("data_processed/idd", "data_processed/bdd"),
    depth_dataset="data_processed/kitti",
    surface_datasets=("data_processed/streetsurface", "data_processed/rtk",
                      "data_processed/idd"),
    curvature_datasets=("data_processed/idd", "data_processed/bdd"),
    split="test",
    benchmark=True,
):
    """Run all evaluations and return a merged dict of all metrics."""
    from eval.benchmark_fps import benchmark_speed
    from eval.eval_box_distance import evaluate_box_distance
    from eval.eval_curvature import evaluate_curvature
    from eval.eval_depth import evaluate_depth
    from eval.eval_det import evaluate_detection
    from eval.eval_seg import evaluate_seg
    from eval.eval_surface import evaluate_surface

    date_str = datetime.now().strftime("%Y-%m-%d")
    results = {}

    # ── 1. Detection ─────────────────────────────────────────────────────────
    for det_ds in det_datasets:
        if os.path.isdir(os.path.join(det_ds, "images", split)):
            print(f"[det] {det_ds} ({split}) …")
            try:
                m = evaluate_detection(model, data_root=det_ds, split=split)
                results.update(m)
                _record_dict(m, date_str, run_name, model_name, _ds_name(det_ds), split)
            except Exception as e:
                print(f"  skipped: {e}")

    # ── 2. Box-distance ───────────────────────────────────────────────────────
    for det_ds in det_datasets:
        if os.path.isdir(os.path.join(det_ds, "images", split)):
            print(f"[box_dist] {det_ds} ({split}) …")
            try:
                m = evaluate_box_distance(model, data_root=det_ds, split=split)
                results.update(m)
                _record_dict(m, date_str, run_name, model_name, _ds_name(det_ds), split)
            except Exception as e:
                print(f"  skipped: {e}")

    # ── 3. Segmentation ───────────────────────────────────────────────────────
    for seg_ds in seg_datasets:
        seg_split = "val" if split == "test" else split   # BDD/IDD seg val is richer
        if os.path.isdir(os.path.join(seg_ds, "images", seg_split)):
            print(f"[seg] {seg_ds} ({seg_split}) …")
            try:
                m = evaluate_seg(model, data_root=seg_ds, split=seg_split)
                results.update(m)
                _record_dict(m, date_str, run_name, model_name, _ds_name(seg_ds), seg_split)
            except Exception as e:
                print(f"  skipped: {e}")

    # ── 4. Depth ──────────────────────────────────────────────────────────────
    if os.path.isdir(os.path.join(depth_dataset, "images", split)):
        print(f"[depth] {depth_dataset} ({split}) …")
        try:
            m = evaluate_depth(model, data_root=depth_dataset, split=split)
            results.update(m)
            _record_dict(m, date_str, run_name, model_name, _ds_name(depth_dataset), split)
        except Exception as e:
            print(f"  skipped: {e}")

    # ── 5. Surface ────────────────────────────────────────────────────────────
    for surf_ds in surface_datasets:
        surf_split = "test"
        if surf_ds.endswith("idd"):
            surf_split = "val"           # IDD Indian test-set labels are on val/test stems
        if os.path.isdir(os.path.join(surf_ds, "images", surf_split)):
            print(f"[surface] {surf_ds} ({surf_split}) …")
            try:
                m = evaluate_surface(model, data_root=surf_ds, split=surf_split)
                results.update(m)
                _record_dict(m, date_str, run_name, model_name, _ds_name(surf_ds), surf_split)
            except Exception as e:
                print(f"  skipped: {e}")

    # ── 6. Curvature ──────────────────────────────────────────────────────────
    for curv_ds in curvature_datasets:
        curv_split = "val"
        if os.path.isdir(os.path.join(curv_ds, "images", curv_split)):
            print(f"[curvature] {curv_ds} ({curv_split}) …")
            try:
                m = evaluate_curvature(model, data_root=curv_ds, split=curv_split)
                results.update(m)
                _record_dict(m, date_str, run_name, model_name, _ds_name(curv_ds), curv_split)
            except Exception as e:
                print(f"  skipped: {e}")

    # ── 7. FPS benchmark ──────────────────────────────────────────────────────
    if benchmark:
        print("[fps] benchmarking …")
        try:
            m = benchmark_speed(model, n_warmup=10, n_runs=50)
            results.update(m)
            _record_dict(m, date_str, run_name, model_name, "hardware", split)
        except Exception as e:
            print(f"  skipped: {e}")

    return results


def _load_model(args):
    """Instantiate the right model from --model (and optional --weights / --checkpoint)."""
    name = args.model
    if name in ("stereosgbm", "depthanything"):
        from eval.depth_wrappers import get_depth_model
        return get_depth_model(name, dataset_root=args.depth_dataset, split=args.split)

    if name == "yolo":
        from models.wrappers import YoloBaseline
        return YoloBaseline(weights=args.weights or "yolov8n.pt")

    if name == "seg":
        from models.wrappers import SegBaseline
        return SegBaseline(checkpoint=args.checkpoint)

    if name == "multitask":
        from models.wrappers import MultiTaskWrapper
        return MultiTaskWrapper(checkpoint=args.checkpoint)

    raise ValueError(f"Unknown model: {name!r}. "
                     "Choose from stereosgbm, depthanything, yolo, seg, multitask.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluations for a given model.")
    parser.add_argument("--run-name", default="baseline", help="Friendly label for this run.")
    parser.add_argument(
        "--model", default="stereosgbm",
        choices=["stereosgbm", "depthanything", "yolo", "seg", "multitask"],
        help="Which model/baseline to evaluate.",
    )
    parser.add_argument("--weights", default=None,
                        help="YOLO .pt weights path (for --model yolo).")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (for --model seg / multitask).")
    parser.add_argument("--split", default="test", help="Evaluation split.")
    parser.add_argument("--depth-dataset", default="data_processed/kitti",
                        help="Depth evaluation dataset root.")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip FPS benchmarking.")
    args = parser.parse_args()

    model = _load_model(args)

    res = evaluate_all(
        model,
        run_name=args.run_name,
        model_name=args.model,
        split=args.split,
        depth_dataset=args.depth_dataset,
        benchmark=not args.no_benchmark,
    )

    print("\n── All evaluation results ──")
    import numpy as np
    for k, v in res.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"  {k}: {v:.4f}")
    print(f"\nSaved metrics to {METRICS_CSV}")
