"""Camera-scene helpers for the demo app.

A camera on a two-wheeler usually sees part of the bike itself (tank, dashboard, handlebars,
mirrors, the rider's hands). Those parts never move relative to the camera, so they can be
found automatically from a few frames ("ego mask"). The app then
  1. crops the bike body off the bottom before the models run (they were trained on road views),
  2. ignores detections that sit on the bike (mirrors, handlebars called "car"),
  3. removes the bike from the road mask,
and keeps every overlay steady over time (smoothing), so the display does not flicker.

All coordinates here are "model space": the full frame letterboxed to 640 x 384, the same space
the page draws in (at 2x).
"""
import math

import cv2
import numpy as np

from datasets.contract import IMG_H, IMG_W
from datasets.transforms import letterbox
from models.postprocess import GENTLE_MAX_DEG, STRAIGHT_MAX_DEG


def lb_geom(w, h):
    """Letterbox of a w x h image into 640 x 384: scale, (pad_left, pad_top), (new_w, new_h)."""
    s = min(IMG_W / w, IMG_H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return s, ((IMG_W - nw) // 2, (IMG_H - nh) // 2), (nw, nh)


def focal_model(src_w, src_h, hfov_deg):
    """Focal length in model-space pixels for a camera with this horizontal field of view."""
    _, _, (nw, _) = lb_geom(src_w, src_h)
    return nw / 2 / math.tan(math.radians(hfov_deg) / 2)


def ground_distance(y, horizon_row, focal, camera_height):
    """Metres to a point on a flat road seen at model row y (0 = unknown / above the horizon)."""
    dy = y - horizon_row
    return 0.0 if dy <= 1 else min(80.0, focal * camera_height / dy)


def sample_frames(path, n=48, max_seconds=None):
    """n RGB frames spread evenly over the video (or its first max_seconds)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_seconds:
        total = min(total or 10**9, int(max_seconds * fps))
    out = []
    for i in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def estimate_ego(frames, min_frac=0.02):
    """Mask (model space, 1 = the rider's own bike / hands) of what stays still in every frame
    and is attached to the bottom of the picture. Returns all zeros when there is none
    (for example a dashcam that does not see the vehicle)."""
    ego = np.zeros((IMG_H, IMG_W), np.uint8)
    if len(frames) < 8:
        return ego
    h, w = frames[0].shape[:2]
    _, (pl, pt), (nw, nh) = lb_geom(w, h)
    sw, sh = max(8, nw // 2), max(8, nh // 2)
    stack = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
                      for f in frames]).astype(np.float32)
    std = stack.std(0)
    static = (std < max(6.0, 0.3 * float(np.median(std)))).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    keep = np.zeros_like(static)
    for i in range(1, n):
        touches_bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= sh - 2
        reaches_top = stats[i, cv2.CC_STAT_TOP] < 0.2 * sh      # e.g. a plain sky, not the bike
        if touches_bottom and not reaches_top and stats[i, cv2.CC_STAT_AREA] >= 0.002 * sw * sh:
            keep[lab == i] = 1
    if keep.sum() < min_frac * sw * sh:
        return ego
    # fill small areas closed in by the bike or the side/bottom edges (mirror glass, gaps in the bars)
    pad = np.pad(keep, 1, constant_values=1)
    pad[0, :] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pad == 0).astype(np.uint8), connectivity=4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_TOP] > 0 and stats[i, cv2.CC_STAT_AREA] < 0.05 * sw * sh:
            pad[lab == i] = 1
    keep = pad[1:-1, 1:-1]
    keep = cv2.resize(keep, (nw, nh), interpolation=cv2.INTER_NEAREST)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    ego[pt:pt + nh, pl:pl + nw] = keep
    return ego


def auto_keep_top(ego, src_w, src_h):
    """Share of the frame height (from the top) to give the models, so the bike body in the
    middle-bottom of the picture is left out. 1.0 = use the whole frame."""
    if not ego.any():
        return 1.0
    _, (pl, pt), (nw, nh) = lb_geom(src_w, src_h)
    band = ego[pt:pt + nh, pl + int(0.35 * nw):pl + int(0.65 * nw)].mean(1)
    r = nh - 1
    while r > 0 and band[r] >= 0.5:
        r -= 1
    keep = (r + 1) / nh
    return 1.0 if keep > 0.95 else float(min(1.0, max(0.5, keep + 0.02)))


def road_extent(edges):
    """How far up the picture the followed road reaches (model rows)."""
    return edges[0][0] - edges[-1][0] if edges else 0


def choose_keep_top(road_fn, samples, src_w, src_h, options=(1.0, 0.8, 0.65, 0.55)):
    """Try a few crops on sample frames and keep the one where the road model finds the road best.
    road_fn(crop, rgb) -> road mask in model space. A smaller crop must do clearly better (15%),
    so a normal dashcam view keeps the whole frame. Returns (keep_top, road tops for the horizon)."""
    best_k, best_score, best_tops = 1.0, -1.0, []
    for k in options:
        crop = CropMap(src_w, src_h, k)
        ext, tops = [], []
        for rgb in samples:
            e = road_edges(road_fn(crop, rgb))
            ext.append(road_extent(e))
            if e:
                tops.append(e[-1][0])
        score = float(np.median(ext)) if ext else 0.0
        if score > max(0.0, best_score) * 1.15 or best_score < 0:
            best_k, best_score, best_tops = k, score, tops
    return best_k, best_tops


class CropMap:
    """Runs the models on the top `keep_top` share of the frame and maps the results back
    into model space of the full frame."""

    def __init__(self, src_w, src_h, keep_top=1.0):
        self.src_w, self.src_h = src_w, src_h
        self.ch = src_h if keep_top >= 1.0 else max(16, int(round(src_h * keep_top)))
        self.sf, self.pf, self.nf = lb_geom(src_w, src_h)
        self.sc, self.pc, self.nc = lb_geom(src_w, self.ch)
        k = self.sf / self.sc
        self.k, self.tx, self.ty = k, self.pf[0] - self.pc[0] * k, self.pf[1] - self.pc[1] * k
        self.M = np.float32([[k, 0, self.tx], [0, k, self.ty]])
        self.bottom_row = self.pf[1] + self.ch * self.sf        # lowest model row the models saw

    def model_input(self, rgb):
        return letterbox(rgb[:self.ch])["image"]

    def boxes(self, b):
        b = np.asarray(b, dtype=np.float64).reshape(-1, 6).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.k + self.tx
        b[:, [1, 3]] = b[:, [1, 3]] * self.k + self.ty
        return b

    def mask(self, seg):
        m = (np.asarray(seg) == 1).astype(np.uint8)
        (px, py), (nw, nh) = self.pc, self.nc
        m[:py], m[py + nh:], m[:, :px], m[:, px + nw:] = 0, 0, 0, 0
        return cv2.warpAffine(m, self.M, (IMG_W, IMG_H), flags=cv2.INTER_NEAREST, borderValue=0)


def drop_ego_boxes(boxes, ego, crop):
    """Indices of boxes that are real objects, not parts of the rider's own bike.
    Dropped: boxes covering most of the picture; boxes resting on the bottom of what the models
    saw that also reach a side edge or are very wide (handlebars, mirrors, hands); boxes on the
    ego mask, when one was found."""
    (pl, pt), (nw, nh) = crop.pf, crop.nf
    bottom, keep = crop.bottom_row, []
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes)[:, :4]):
        x1, x2 = int(max(0, x1)), int(min(IMG_W - 1, x2))
        y1, y2 = int(max(0, y1)), int(min(IMG_H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) > 0.4 * nw * nh:
            continue
        on_bottom = y2 >= bottom - 0.03 * nh
        at_side = x1 <= pl + 0.02 * nw or x2 >= pl + 0.98 * nw
        if on_bottom and (at_side or x2 - x1 > 0.6 * nw):
            continue
        if ego.any():
            strip = ego[max(y1, y2 - max(2, (y2 - y1) // 6)):y2 + 1, x1:x2 + 1]
            if strip.mean() >= 0.5 or ego[(y1 + y2) // 2, (x1 + x2) // 2]:
                continue
        keep.append(i)
    return keep


def road_edges(mask, step=6, min_w_frac=0.03):
    """Left and right edge of the road ahead: [[y, x_left, x_right], ...] on a fixed row grid,
    nearest row first, smoothed down the road.

    Uses the largest piece of road, followed from where it disappears in the distance down
    towards the rider. It stops where the road is cut off (by the bike's tank or the rider's
    hands), so the bits of road seen beside the bike do not bend the lines."""
    if mask is None:
        return []
    m = (np.asarray(mask) == 1).astype(np.uint8)
    H, W = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return []
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[main, cv2.CC_STAT_AREA] < 0.005 * H * W:
        return []
    m = (lab == main)
    min_w = max(2, int(min_w_frac * W))
    rows = [y for y in range(H - 1, -1, -step)][::-1]          # same row grid every frame, top first
    out, prev = [], None
    for y in rows:
        cols = np.flatnonzero(m[y])
        if cols.size == 0:
            if out:
                break
            continue
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        if prev is None:
            run = max(runs, key=len)
        else:
            pl, pr = prev
            over = [r for r in runs if r[-1] >= pl - 0.1 * W and r[0] <= pr + 0.1 * W]
            if not over:
                break
            run = max(over, key=len)
            if len(out) > 4 and run.size < 0.6 * (pr - pl + 1):    # road cut off below this row
                break
        if run.size >= min_w or prev is None:
            out.append((y, run[0], run[-1]))
        prev = (run[0], run[-1])
    out = [o for o in out if o[2] - o[1] + 1 >= min_w]
    if len(out) < 3:
        return []
    a = np.array(out[::-1], dtype=np.float64)                  # nearest row first
    k = np.ones(5) / 5
    for c in (1, 2):
        a[:, c] = np.convolve(np.pad(a[:, c], 2, mode="edge"), k, mode="valid")
    return a.round().astype(int).tolist()


def edges_mask(edges, shape=(IMG_H, IMG_W), step=6):
    """Clean road mask drawn from the edges (used for curvature and pothole road coverage)."""
    m = np.zeros(shape, np.uint8)
    if len(edges) >= 2:
        left = [(e[1], e[0]) for e in edges]
        right = [(e[2], e[0]) for e in edges][::-1]
        cv2.fillPoly(m, [np.array(left + right, np.int32)], 1)
    return m


def horizon_from_edges(tops, src_w, src_h, default_frac=0.45):
    """Horizon row (model space) from where the road disappears in sample frames."""
    _, (_, pt), (_, nh) = lb_geom(src_w, src_h)
    if len(tops) < 3:
        return pt + default_frac * nh
    row = float(np.median(tops)) - 2
    return float(min(pt + 0.6 * nh, max(pt + 0.1 * nh, row)))


def _alpha(dt, tau):
    return 1.0 - math.exp(-max(dt, 1e-3) / tau)


class MaskSmoother:
    """Road mask averaged over the last ~tau seconds, so one bad frame does not show."""

    def __init__(self, tau=0.3):
        self.tau, self.acc = tau, None

    def update(self, mask, dt):
        m = np.asarray(mask, dtype=np.float32)
        self.acc = m if self.acc is None else self.acc + _alpha(dt, self.tau) * (m - self.acc)
        return (self.acc > 0.5).astype(np.uint8)


class BoxSmoother:
    """Each tracked object's box eased over ~tau seconds."""

    def __init__(self, tau=0.12):
        self.tau, self.boxes = tau, {}

    def update(self, tracks, dt):
        a, new = _alpha(dt, self.tau), {}
        for t in tracks:
            b = np.asarray(t["box"][:4], dtype=np.float64)
            prev = self.boxes.get(t["id"])
            new[t["id"]] = b if prev is None else prev + a * (b - prev)
        self.boxes = new
        return new


class CurveSmoother:
    """Bend angle eased over ~tau seconds; the last known value is held for `hold` seconds
    when the road is briefly not found."""

    def __init__(self, tau=0.5, hold=1.5):
        self.tau, self.hold, self.bend, self.missing = tau, hold, None, 0.0

    def update(self, code, bend, dt):
        if code:
            self.bend = bend if self.bend is None else self.bend + _alpha(dt, self.tau) * (bend - self.bend)
            self.missing = 0.0
        else:
            self.missing += dt
            if self.missing > self.hold:
                self.bend = None
        if self.bend is None:
            return 0, 0.0
        a = abs(self.bend)
        if a < STRAIGHT_MAX_DEG:
            return 1, self.bend
        if self.bend < 0:
            return (2 if a < GENTLE_MAX_DEG else 3), self.bend
        return (4 if a < GENTLE_MAX_DEG else 5), self.bend
