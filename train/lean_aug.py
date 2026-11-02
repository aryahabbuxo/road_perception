"""Self-supervised lean: tilt training images by a random angle, as a leaning two-wheeler's
camera would see the road. Boxes, road masks and distance maps are tilted the same way, and
the angle itself becomes the label for the lean head, so no sensor or annotation is needed.

Convention: lean +theta = rider leaning right = camera rolled clockwise, so the scene appears
rotated counterclockwise in the image.
"""
import math

import torch
import torch.nn.functional as F

from datasets.contract import SEG_IGNORE

PAD_IMAGE = 114 / 255.0   # same grey as the letterbox padding


def _grid(angles_deg, H, W, device):
    """Sampling grid: for every output pixel, where to read in the input image."""
    t = torch.deg2rad(angles_deg.to(device).float())
    c, s = torch.cos(t), torch.sin(t)
    theta = torch.zeros(len(t), 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = c, -(H / W) * s
    theta[:, 1, 0], theta[:, 1, 1] = (W / H) * s, c
    return F.affine_grid(theta, (len(t), 1, H, W), align_corners=False)


def _rotate_boxes(t, angles_deg, H, W, min_keep=0.4):
    """t: [M, 6] batch_idx, cls, cx, cy, w, h (0-1). Returns the boxes that stay visible,
    as the axis-aligned boxes around their tilted corners, clipped to the image."""
    if len(t) == 0:
        return t
    a = torch.deg2rad(angles_deg.to(t.device).float()[t[:, 0].long()])
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    cx, cy, w, h = t[:, 2] * W, t[:, 3] * H, t[:, 4] * W, t[:, 5] * H
    xs = torch.stack([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2], 1) - W / 2
    ys = torch.stack([cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2], 1) - H / 2
    xr, yr = xs * c + ys * s + W / 2, -xs * s + ys * c + H / 2
    x1, x2 = xr.min(1).values, xr.max(1).values
    y1, y2 = yr.min(1).values, yr.max(1).values
    area = (x2 - x1) * (y2 - y1)
    x1c, x2c = x1.clamp(0, W), x2.clamp(0, W)
    y1c, y2c = y1.clamp(0, H), y2.clamp(0, H)
    area_c = (x2c - x1c).clamp(min=0) * (y2c - y1c).clamp(min=0)
    keep = (area_c >= min_keep * area) & ((x2c - x1c) > 2) & ((y2c - y1c) > 2)
    out = torch.stack([t[:, 0], t[:, 1], (x1c + x2c) / 2 / W, (y1c + y2c) / 2 / H,
                       (x2c - x1c) / W, (y2c - y1c) / H], 1)
    return out[keep]


def apply_lean(batch, angles_deg):
    """Tilts every image in the batch by its angle (0 = unchanged) and adds batch["lean"]."""
    img = batch["img"]
    B, _, H, W = img.shape
    angles_deg = angles_deg.to(img.device).float()
    grid = _grid(angles_deg, H, W, img.device)
    out = dict(batch)
    out["img"] = F.grid_sample(img - PAD_IMAGE, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False) + PAD_IMAGE
    m = (batch["seg_mask"].float() + 1)[:, None]            # 0 after sampling = outside
    m = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0]
    out["seg_mask"] = torch.where(m == 0, torch.full_like(m, SEG_IGNORE), m - 1).long()
    d = F.grid_sample(batch["depth"][:, None].float(), grid, mode="nearest",
                      padding_mode="zeros", align_corners=False)[:, 0]
    out["depth"] = d                                        # 0 = no value
    out["det_targets"] = _rotate_boxes(batch["det_targets"], angles_deg, H, W)
    out["lean"] = angles_deg
    return out


def random_lean(batch, p=0.5, max_deg=30.0, generator=None):
    """Tilts a random half of the batch by a random angle in [-max_deg, max_deg]."""
    B = batch["img"].shape[0]
    angles = (torch.rand(B, generator=generator) * 2 - 1) * max_deg
    angles = angles * (torch.rand(B, generator=generator) < p).float()
    return apply_lean(batch, angles)


def fixed_lean_angles(n, max_deg=30.0):
    """Evenly spread test angles, for measuring lean error the same way every time."""
    return torch.linspace(-max_deg, max_deg, n) if n > 1 else torch.zeros(n)


def lean_deg_to_text(deg):
    return "upright" if abs(deg) < 3 else f"{abs(deg):.0f} deg {'right' if deg > 0 else 'left'}"
