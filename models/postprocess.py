"""Turns the model's raw outputs into rider-level information (no training involved).

  box_distances(depth, boxes) -> distance in metres for every detected object
  road_curvature(road_mask, lean_deg) -> curvature code (see CURVATURE_NAMES) and bend angle
"""
import numpy as np
import torch
import torch.nn.functional as F

# Bend angle (degrees) between the near and far parts of the road's centre line
# (measured in the image; tune on real data if needed)
STRAIGHT_MAX_DEG = 8.0
GENTLE_MAX_DEG = 25.0


def box_distances(depth, boxes):
    """Distance of each object = median predicted distance in the bottom-centre of its box,
    where the object touches the ground (more reliable than the whole box, which mixes in
    background). depth: [H, W] metres. boxes: [K, >=4] x1, y1, x2, y2 in pixels.
    Returns a float tensor [K] (0 = no valid distance)."""
    H, W = depth.shape
    out = torch.zeros(len(boxes))
    for i, (x1, y1, x2, y2) in enumerate(boxes[:, :4].tolist()):
        w, h = x2 - x1, y2 - y1
        xa, xb = int(max(0, x1 + 0.25 * w)), int(min(W, x2 - 0.25 * w))
        ya, yb = int(max(0, y2 - 0.25 * h)), int(min(H, y2))
        region = depth[ya:max(yb, ya + 1), xa:max(xb, xa + 1)]
        vals = region[region > 0]
        if vals.numel() == 0:      # tiny box: fall back to the whole box
            whole = depth[int(max(0, y1)):int(min(H, y2)) + 1, int(max(0, x1)):int(min(W, x2)) + 1]
            vals = whole[whole > 0]
        if vals.numel() > 0:
            out[i] = vals.median()
    return out


def road_centre_line(road_mask, valid=None, step=8, min_width_frac=0.03):
    """Centre points (x, y) of the road ahead, walking up from the bottom of the image and
    following the road the rider is on (ignores side roads). Only rows where both road
    edges are visible are used: a road cut off by the image edge would bias the centre.
    road_mask: [H, W], 1 = road. valid: [H, W], 0 where there is no image (after straightening)."""
    m = road_mask.cpu().numpy() if torch.is_tensor(road_mask) else np.asarray(road_mask)
    v = None if valid is None else (valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid))
    H, W = m.shape
    min_w = max(2, int(min_width_frac * W))

    def edge_visible(y, x):        # the pixel just outside the road is real image, not border
        return 0 <= x < W and (v is None or v[y, x] > 0)

    cx, pts = W / 2.0, []
    for y in range(H - 1, -1, -step):
        cols = np.flatnonzero(m[y] == 1)
        if cols.size == 0:
            if pts:
                break
            continue
        splits = np.flatnonzero(np.diff(cols) > 1) + 1
        runs = [r for r in np.split(cols, splits) if r.size >= min_w]
        if not runs:
            if pts:
                break
            continue
        run = min(runs, key=lambda r: 0 if r[0] <= cx <= r[-1] else min(abs(r[0] - cx), abs(r[-1] - cx)))
        if pts and not (run[0] - 0.15 * W <= cx <= run[-1] + 0.15 * W):
            break                       # the road we were following has ended
        cx = (run[0] + run[-1]) / 2.0   # keep following, even through rows we don't use
        if edge_visible(y, run[0] - 1) and edge_visible(y, run[-1] + 1):
            pts.append((cx, y))
    return np.array(pts, dtype=np.float64)


def straighten_mask(road_mask, lean_deg):
    """Undo the rider's lean: rotate the road mask back by the predicted lean angle.
    Returns (road mask, valid area) — valid is 0 where the rotation left no image."""
    from train.lean_aug import _grid

    m = torch.as_tensor(np.asarray(road_mask) if not torch.is_tensor(road_mask) else road_mask)
    H, W = m.shape
    grid = _grid(torch.tensor([-float(lean_deg)]), H, W, "cpu")
    both = torch.stack([(m == 1).float(), torch.ones(H, W)])[None].cpu()
    out = F.grid_sample(both, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return out[0].long(), out[1]


def road_curvature(road_mask, lean_deg=0.0, min_points=8):
    """Returns (code, bend_deg). bend_deg > 0 means the road bends right.
    The mask is first straightened using the predicted lean, then the bend is the angle
    between the near and far halves of the road's centre line."""
    valid = None
    if abs(lean_deg) >= 1.0:
        road_mask, valid = straighten_mask(road_mask, lean_deg)
    pts = road_centre_line(road_mask, valid)
    if len(pts) < min_points:
        return 0, 0.0                                   # unknown
    half = len(pts) // 2
    near, far = pts[:half], pts[half:]

    def heading(p):   # angle of the line through the points, measured from straight up
        t = -p[:, 1]                                    # upward
        slope = np.polyfit(t, p[:, 0], 1)[0]            # dx per unit upward
        return np.degrees(np.arctan(slope))

    bend = float(heading(far) - heading(near))
    a = abs(bend)
    if a < STRAIGHT_MAX_DEG:
        return 1, bend                                  # straight
    if bend < 0:
        return (2 if a < GENTLE_MAX_DEG else 3), bend   # gentle_left / sharp_left
    return (4 if a < GENTLE_MAX_DEG else 5), bend       # gentle_right / sharp_right
