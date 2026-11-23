"""Pothole severity (L1-L5), hazard tracking across video frames, and rider alerts.
Works on the model's outputs only (boxes, per-object distance, road mask): no training.

  severity = pothole_severity(box, distance_m, road_mask)       # one frame, one pothole
  tracker  = HazardTracker()
  tracks   = tracker.update(boxes, distances, road_mask)        # call once per frame
  alerts   = rider_alerts(tracks)                               # what to warn about now

Severity is judged from the pothole's real width (from its pixel width and distance) and
from how much of the road it blocks. A single camera cannot measure how deep a pothole
is, so depth is not used.
"""
import math
from collections import deque

import numpy as np
import torch

from datasets.contract import IMG_H, IMG_W

POTHOLE, HUMP = 0, 1
CLASS_NAMES = ("pothole", "speed_breaker", "car", "two_wheeler", "bus_truck", "person")

# Pothole width in metres: up to 0.3 = L1, 0.6 = L2, 1.0 = L3, 1.5 = L4, wider = L5
SEVERITY_WIDTH_M = (0.3, 0.6, 1.0, 1.5)
# A pothole covering at least this share of the road width is at least L4 (hard to avoid)
BLOCKING_COVERAGE = 0.5
# Camera horizontal field of view. Typical dashcam / phone value; set it for your camera.
DEFAULT_HFOV_DEG = 70.0


def focal_px(width=IMG_W, hfov_deg=DEFAULT_HFOV_DEG):
    """Focal length in pixels for an image `width` wide with the given field of view."""
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def road_width_px(road_mask, y, x):
    """Width (pixels) of the road segment in row y that contains x, or is nearest to it."""
    m = road_mask.cpu().numpy() if torch.is_tensor(road_mask) else np.asarray(road_mask)
    y = int(min(max(y, 0), m.shape[0] - 1))
    cols = np.flatnonzero(m[y] == 1)
    if cols.size == 0:
        return 0
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    run = min(runs, key=lambda r: 0 if r[0] <= x <= r[-1] else min(abs(r[0] - x), abs(r[-1] - x)))
    return int(run[-1] - run[0] + 1)


def pothole_severity(box, distance_m, road_mask=None, hfov_deg=DEFAULT_HFOV_DEG):
    """box: x1, y1, x2, y2 (pixels). distance_m: distance to the pothole (0 = unknown).
    Returns {"level": 1-5 or 0 if unknown, "width_m", "coverage"}."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    width_px = max(0.0, x2 - x1)
    width_m = width_px * distance_m / focal_px(IMG_W, hfov_deg) if distance_m > 0 else None

    coverage = None
    if road_mask is not None:
        road_w = road_width_px(road_mask, y2 - 1, (x1 + x2) / 2)
        coverage = min(1.0, width_px / road_w) if road_w > 0 else None

    if width_m is None and coverage is None:
        return {"level": 0, "width_m": None, "coverage": None}
    level = 1
    if width_m is not None:
        level = 1 + sum(width_m > t for t in SEVERITY_WIDTH_M)
    if coverage is not None and coverage >= BLOCKING_COVERAGE:
        level = max(level, 4)
    return {"level": int(level), "width_m": width_m, "coverage": coverage}


def box_iou(a, b):
    """IoU between boxes a [N, 4] and b [M, 4] (x1, y1, x2, y2)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


class Track:
    def __init__(self, track_id, cls, box, conf, distance, history):
        self.id, self.cls, self.box, self.conf = track_id, cls, box, conf
        self.distance = distance          # smoothed metres (0 = unknown)
        self.hits, self.misses = 1, 0
        self.levels = deque(maxlen=history)

    def as_dict(self):
        level = int(np.median(self.levels)) if self.levels else None
        return {"id": self.id, "cls": self.cls, "name": CLASS_NAMES[self.cls] if self.cls < len(CLASS_NAMES) else str(self.cls),
                "box": self.box, "conf": self.conf, "distance": self.distance, "severity": level}


class HazardTracker:
    """Keeps an ID on each object across frames (matched by box overlap, same class only).
    A track is reported once seen in `min_hits` frames, which stops one-frame false alarms,
    and dropped after `max_misses` frames without a match. Distance is smoothed, and pothole
    severity is the median of the last `history` frames, so it does not flicker."""

    def __init__(self, iou_thr=0.3, min_hits=3, max_misses=5, smooth=0.5, history=5,
                 hfov_deg=DEFAULT_HFOV_DEG):
        self.iou_thr, self.min_hits, self.max_misses = iou_thr, min_hits, max_misses
        self.smooth, self.history, self.hfov_deg = smooth, history, hfov_deg
        self.tracks, self.next_id = [], 1

    def update(self, boxes, distances=None, road_mask=None):
        """boxes: [K, 6] x1, y1, x2, y2, conf, cls (MultiTaskWrapper.predict()["boxes"][i]).
        distances: [K] metres (predict()["box_distance"][i]). road_mask: [H, W] (predict()["seg"][i]).
        Returns the confirmed tracks as dicts, nearest first."""
        boxes = np.asarray(boxes.cpu() if torch.is_tensor(boxes) else boxes, dtype=np.float64).reshape(-1, 6)
        if distances is None:
            distances = np.zeros(len(boxes))
        distances = np.asarray(distances.cpu() if torch.is_tensor(distances) else distances, dtype=np.float64)

        # Greedy matching: highest overlap first, same class only
        iou = box_iou([t.box for t in self.tracks], boxes[:, :4])
        for i, t in enumerate(self.tracks):
            iou[i, boxes[:, 5] != t.cls] = 0
        matched_t, matched_d = set(), set()
        for flat in np.argsort(-iou, axis=None):
            i, j = np.unravel_index(flat, iou.shape)
            if iou[i, j] < self.iou_thr:
                break
            if i in matched_t or j in matched_d:
                continue
            matched_t.add(i)
            matched_d.add(j)
            t = self.tracks[i]
            t.box, t.conf = boxes[j, :4].tolist(), float(boxes[j, 4])
            d = float(distances[j])
            if d > 0:
                t.distance = d if t.distance <= 0 else self.smooth * d + (1 - self.smooth) * t.distance
            t.hits += 1
            t.misses = 0
            self._add_severity(t, d, road_mask)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for j in range(len(boxes)):
            if j not in matched_d:
                t = Track(self.next_id, int(boxes[j, 5]), boxes[j, :4].tolist(), float(boxes[j, 4]),
                          float(distances[j]), self.history)
                self.next_id += 1
                self._add_severity(t, float(distances[j]), road_mask)
                self.tracks.append(t)

        live = [t.as_dict() for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]
        return sorted(live, key=lambda d: d["distance"] if d["distance"] > 0 else float("inf"))

    def _add_severity(self, t, distance, road_mask):
        if t.cls != POTHOLE:
            return
        s = pothole_severity(t.box, distance, road_mask, self.hfov_deg)
        if s["level"] > 0:
            t.levels.append(s["level"])


def rider_alerts(tracks, pothole_level=3, pothole_dist=20.0, hump_dist=25.0,
                 near_dist=8.0, near_classes=(2, 3, 4, 5)):
    """Which confirmed hazards to warn the rider about now, nearest first:
    potholes of severity >= pothole_level within pothole_dist m, speed breakers within
    hump_dist m, and vehicles or people closer than near_dist m."""
    out = []
    for t in tracks:
        d = t["distance"]
        if d <= 0:
            continue
        if t["cls"] == POTHOLE and (t["severity"] or 0) >= pothole_level and d <= pothole_dist:
            out.append({**t, "alert": f"Pothole L{t['severity']} in {d:.0f} m"})
        elif t["cls"] == HUMP and d <= hump_dist:
            out.append({**t, "alert": f"Speed breaker in {d:.0f} m"})
        elif t["cls"] in near_classes and d <= near_dist:
            out.append({**t, "alert": f"{t['name'].replace('_', ' ').capitalize()} close: {d:.0f} m"})
    return sorted(out, key=lambda a: a["distance"])
