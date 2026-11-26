"""Run Saarth on a video and save everything the demo page needs.

  # with the trained baselines you have now
  python -m app.process_video --video my_ride.mp4 --name ride1 --engine baselines \
      --yolo checkpoints/20261031_yolo11s_all8/best.pt \
      --seg checkpoints/20261031_segformer_b0_road/best.pt --every 3

  # with the final combined model, once it is trained
  python -m app.process_video --video my_ride.mp4 --name ride1 --engine multitask \
      --checkpoint checkpoints/20261103_multitask_main/best.pt --every 3

Before processing, a few frames are sampled to find the rider's own bike in the picture
(app/scene.py), the bike body is cropped off before the models run, and the horizon is found
from where the road disappears. Results are smoothed over time so the overlays stay steady.

Writes app/web/runs/<name>/video.mp4 (browser-playable copy), frames.json and ego.png (the bike
area that was ignored, for checking), and adds the run to app/web/runs/index.json.
A laptop CPU cannot run the models in real time, so the work is done here once and the page
then plays the video at full speed with the saved results in sync.
"""
import argparse
import datetime
import json
import os
import subprocess
import time

import cv2
import numpy as np
import torch

from app.can_frames import encode_frame
from app.engines import DEFAULT_CAMERA_HEIGHT_M, make_engine
from app.scene import (BoxSmoother, CropMap, CurveSmoother, MaskSmoother, auto_keep_top,
                       choose_keep_top, drop_ego_boxes, edges_mask, estimate_ego, focal_model, ground_distance,
                       horizon_from_edges, road_edges, sample_frames)
from datasets.contract import IMG_H, IMG_W
from models.hazards import DEFAULT_HFOV_DEG, HazardTracker, pothole_severity, rider_alerts
from models.postprocess import road_curvature

WEB_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "runs")


def browser_copy(src, dst, max_seconds=None):
    """H.264 copy the page can play in any browser (phone videos are often HEVC)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    cmd = [exe, "-y", "-loglevel", "error", "-i", src]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-vf", "scale=-2:'min(720,ih)'", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"WARNING: could not make a browser copy of the video ({e}). "
              f"Install it with: pip install imageio-ffmpeg")
        return False


class Scene:
    """Runs the engine on one frame and returns results in full-frame model space."""

    def __init__(self, engine, crop, ego, horizon_row, focal, camera_height):
        self.engine, self.crop, self.ego = engine, crop, ego
        self.horizon_row, self.focal, self.camera_height = horizon_row, focal, camera_height

    def __call__(self, rgb, crop=None):
        crop = crop or self.crop
        img = torch.from_numpy(crop.model_input(rgb)).permute(2, 0, 1).float().div(255)[None]
        with torch.no_grad():
            out = self.engine(img)
        boxes = crop.boxes(out["boxes"].cpu() if torch.is_tensor(out["boxes"]) else out["boxes"])
        keep = drop_ego_boxes(boxes, self.ego, crop)
        boxes = boxes[keep]
        if self.engine.name == "multitask":           # distance from the depth head
            dist = np.asarray(out["box_distance"].cpu() if torch.is_tensor(out["box_distance"])
                              else out["box_distance"], dtype=np.float64).reshape(-1)[keep]
        else:                                         # flat-road geometry from where it meets the road
            dist = np.array([ground_distance(b[3], self.horizon_row, self.focal, self.camera_height)
                             for b in boxes])
        road = None
        if out["seg"] is not None:
            road = crop.mask(out["seg"].cpu() if torch.is_tensor(out["seg"]) else out["seg"])
            road[self.ego > 0] = 0
        return boxes, dist, road, out["surface"], out["lean"]


def process(args):
    engine = make_engine(args.engine, checkpoint=args.checkpoint, yolo=args.yolo, seg=args.seg,
                         device=args.device, conf=args.conf)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_frame = total if not args.max_seconds else min(total or 10**9, int(args.max_seconds * fps))
    out_dir = os.path.join(WEB_RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. look at the whole clip once: the rider's bike, the crop, the horizon ----
    print("Checking the camera view ...")
    samples = sample_frames(args.video, n=48, max_seconds=args.max_seconds)
    ego = estimate_ego(samples) if args.ego else np.zeros((IMG_H, IMG_W), np.uint8)
    focal = focal_model(src_w, src_h, args.hfov)
    full = CropMap(src_w, src_h, 1.0)
    pt, nh = full.pf[1], full.nf[1]
    scene = Scene(engine, full, ego, pt + 0.45 * nh, focal, args.camera_height)
    has_road = bool(args.seg) or args.engine == "multitask"
    tops = []
    if args.keep_top != "auto":
        keep_top = float(args.keep_top)
    elif ego.any():                                   # bike found: crop just above it
        keep_top = auto_keep_top(ego, src_w, src_h)
    elif has_road:                                    # try a few crops; keep the one where the road shows best
        keep_top, tops = choose_keep_top(lambda c, rgb: scene(rgb, c)[2], samples[::4], src_w, src_h)
    else:
        keep_top = 1.0
    scene.crop = CropMap(src_w, src_h, keep_top)
    if args.horizon == "auto":
        if has_road and not tops:
            for rgb in samples[::4]:
                e = road_edges(scene(rgb)[2])
                if e:
                    tops.append(e[-1][0])
        scene.horizon_row = horizon_from_edges(tops, src_w, src_h)
    else:
        scene.horizon_row = pt + float(args.horizon) * nh
    horizon_row, crop = scene.horizon_row, scene.crop
    cv2.imwrite(os.path.join(out_dir, "ego.png"), ego * 255)
    print(f"  bike in view: {ego.mean() * 100:.0f}% of the picture | models see the top "
          f"{keep_top * 100:.0f}% of the frame | horizon at {(horizon_row - pt) / nh * 100:.0f}% "
          f"of the height")

    # ---- 2. every frame ----
    tracker = HazardTracker(hfov_deg=args.hfov)
    road_s, box_s, curve_s = MaskSmoother(), BoxSmoother(), CurveSmoother()
    frames, counter, t0, idx, prev_t = [], 0, time.time(), 0, None
    while True:
        ok, bgr = cap.read()
        if not ok or (last_frame and idx >= last_frame):
            break
        if idx % args.every:
            idx += 1
            continue
        t = idx / fps
        dt = 1.0 / fps if prev_t is None else t - prev_t
        prev_t = t
        boxes, dist, road, surface, lean = scene(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        edges, road = [], (road_s.update(road, dt) if road is not None else None)
        if road is not None:
            edges = road_edges(road)
            road = edges_mask(edges)                 # just the road ahead, without stray bits
        tracks = tracker.update(boxes, dist, road)
        smooth = box_s.update(tracks, dt)
        alert_ids = {a["id"] for a in rider_alerts(tracks)}
        if edges:
            code, bend = road_curvature(torch.from_numpy(road).long(), lean_deg=lean or 0.0)
        else:
            code, bend = 0, 0.0
        code, bend = curve_s.update(code, bend, dt)

        objs = []
        for tr in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in smooth[tr["id"]]]
            width_m = 0.0
            if tr["cls"] == 0 and tr["distance"] > 0:
                width_m = pothole_severity(tr["box"], tr["distance"], road, args.hfov)["width_m"] or 0.0
            lat = ((x1 + x2) / 2 - IMG_W / 2) / focal * tr["distance"] if tr["distance"] > 0 else 0.0
            objs.append([tr["id"], tr["cls"], x1, y1, x2, y2, round(tr["distance"], 1), tr["severity"] or 0,
                         round(tr["conf"], 2), int(tr["id"] in alert_ids), round(width_m, 2), round(lat, 1)])

        proc_fps = (len(frames) + 1) / max(1e-6, time.time() - t0)
        can = encode_frame(tracks, code, bend, surface, lean, proc_fps, counter,
                           degraded=road is not None and not edges)
        counter += 1
        frames.append({
            "t": round(t, 3),
            "objs": objs,
            "edges": edges,
            "curv": [code, round(bend, 1)],
            "surf": None if surface is None else [surface[0], round(surface[1], 2)],
            "lean": None if lean is None else round(lean, 1),
            "can": can,
        })
        if len(frames) % 25 == 0:
            print(f"  {len(frames)} frames processed ({proc_fps:.1f} per second)")
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit("No frames were read from the video.")

    meta = {"name": args.name, "engine": engine.name, "fps": fps, "src_size": [src_w, src_h],
            "letterbox": {"scale": crop.sf, "pad": list(crop.pf), "size": [IMG_W, IMG_H]},
            "hfov": args.hfov, "focal": round(focal, 2), "horizon_row": round(horizon_row, 1),
            "camera_height": args.camera_height, "keep_top": round(keep_top, 3),
            "ego_share": round(float(ego.mean()), 3), "every": args.every,
            "processed_fps": round(len(frames) / max(1e-6, time.time() - t0), 2),
            "duration": frames[-1]["t"], "frames": len(frames),
            "has": {"surface": frames[0]["surf"] is not None, "lean": frames[0]["lean"] is not None,
                    "road": any(f["edges"] for f in frames)},
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump({"meta": meta, "frames": frames}, f, separators=(",", ":"))

    browser_copy(args.video, os.path.join(out_dir, "video.mp4"), max_seconds=args.max_seconds)

    index_path = os.path.join(WEB_RUNS, "index.json")
    runs = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            runs = [r for r in json.load(f) if r["name"] != args.name]
    runs.insert(0, {"name": args.name, "engine": engine.name, "duration": meta["duration"],
                    "created": meta["created"]})
    with open(index_path, "w") as f:
        json.dump(runs, f, indent=1)
    print(f"Done: {len(frames)} frames, {meta['processed_fps']} per second. Saved to {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--name", required=True, help="run name shown in the page (letters, digits, - and _)")
    p.add_argument("--engine", choices=["baselines", "multitask"], default="baselines")
    p.add_argument("--checkpoint", help="combined model best.pt (engine multitask)")
    p.add_argument("--yolo", help="YOLO baseline best.pt (engine baselines)")
    p.add_argument("--seg", help="SegFormer baseline best.pt (engine baselines, optional)")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--every", type=int, default=1, help="run the models on every Nth frame (3 = 3x faster)")
    p.add_argument("--max-seconds", type=float, default=None, help="only use the first N seconds")
    p.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG, help="camera field of view (deg)")
    p.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    p.add_argument("--horizon", default="auto",
                   help="'auto', or the horizon row as a share of the frame height (0.45 = slightly above middle)")
    p.add_argument("--keep-top", default="auto",
                   help="'auto', or the share of the frame height (from the top) the models see; 1 = whole frame")
    p.add_argument("--no-ego", dest="ego", action="store_false",
                   help="do not look for the rider's own bike in the picture")
    args = p.parse_args()
    if not all(ch.isalnum() or ch in "-_" for ch in args.name):
        raise SystemExit("--name may only use letters, digits, - and _")
    if args.engine == "multitask" and not args.checkpoint:
        raise SystemExit("--engine multitask needs --checkpoint")
    if args.engine == "baselines" and not args.yolo:
        raise SystemExit("--engine baselines needs --yolo (and optionally --seg)")
    process(args)


if __name__ == "__main__":
    main()
