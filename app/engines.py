"""The two ways the demo app can run the perception models. Both return the same thing per frame:

  {"boxes": [K, 6] x1, y1, x2, y2, conf, cls   (reported classes only, pixels in 640x384),
   "box_distance": [K] metres (0 = unknown),
   "seg": [384, 640] road mask (1 = road) or None,
   "surface": (code, confidence) or None,
   "lean": degrees or None}

MultiTaskEngine  - the final combined model (one checkpoint does everything).
BaselineEngine   - the trained baselines you already have: YOLOv11s for objects and
                   SegFormer for the road. They have no distance head, so distance is
                   estimated from where each object touches the road (flat-road geometry),
                   and surface type and lean are not available.
"""
import torch

from datasets.contract import IMG_H, IMG_W, REPORTED_DET_CLASSES
from models.hazards import DEFAULT_HFOV_DEG, focal_px

DEFAULT_CAMERA_HEIGHT_M = 1.0      # camera height above the road on a two-wheeler
DEFAULT_HORIZON_FRAC = 0.45        # horizon row as a share of image height (camera looking level)


def ground_distance(y_bottom, horizon_row, camera_height=DEFAULT_CAMERA_HEIGHT_M,
                    hfov_deg=DEFAULT_HFOV_DEG):
    """Distance (m) to a point on a flat road seen at image row y_bottom.
    Rows at or above the horizon have no ground distance (returns 0)."""
    dy = y_bottom - horizon_row
    if dy <= 1:
        return 0.0
    return min(80.0, focal_px(IMG_W, hfov_deg) * camera_height / dy)


def _keep_reported(boxes):
    if len(boxes) == 0:
        return boxes
    return boxes[torch.isin(boxes[:, 5].long(), torch.tensor(REPORTED_DET_CLASSES))]


class MultiTaskEngine:
    name = "multitask"

    def __init__(self, checkpoint, device=None, conf=0.25):
        from models.wrappers import MultiTaskWrapper

        self.model = MultiTaskWrapper(checkpoint, device=device, conf=conf)

    def __call__(self, img):
        """img: float32 [1, 3, 384, 640], RGB 0-1."""
        p = self.model.predict(img)
        surface = None
        if p.get("surface") is not None:
            surface = (int(p["surface"][0]), float(p["surface_conf"][0]))
        return {"boxes": p["boxes"][0], "box_distance": p["box_distance"][0],
                "seg": p["seg"][0] if p["seg"] is not None else None,
                "surface": surface,
                "lean": float(p["lean"][0]) if p.get("lean") is not None else None}


class BaselineEngine:
    name = "baselines"

    def __init__(self, yolo_checkpoint, seg_checkpoint=None, device=None, conf=0.25,
                 camera_height=DEFAULT_CAMERA_HEIGHT_M, horizon_frac=DEFAULT_HORIZON_FRAC,
                 hfov_deg=DEFAULT_HFOV_DEG):
        from models.wrappers import SegBaseline, YoloBaseline

        self.det = YoloBaseline(yolo_checkpoint, conf=conf, device=device)
        self.seg = SegBaseline(seg_checkpoint, device=device) if seg_checkpoint else None
        self.camera_height, self.hfov_deg = camera_height, hfov_deg
        self.horizon_row = horizon_frac * IMG_H

    def __call__(self, img):
        boxes = _keep_reported(self.det.predict(img)["boxes"][0])
        dist = torch.tensor([ground_distance(float(b[3]), self.horizon_row, self.camera_height,
                                             self.hfov_deg) for b in boxes], dtype=torch.float32)
        seg = self.seg.predict(img)["seg"][0] if self.seg is not None else None
        return {"boxes": boxes, "box_distance": dist, "seg": seg, "surface": None, "lean": None}


def make_engine(kind, **kw):
    if kind == "multitask":
        return MultiTaskEngine(kw["checkpoint"], device=kw.get("device"), conf=kw.get("conf", 0.25))
    if kind == "baselines":
        return BaselineEngine(kw["yolo"], kw.get("seg"), device=kw.get("device"),
                              conf=kw.get("conf", 0.25),
                              camera_height=kw.get("camera_height", DEFAULT_CAMERA_HEIGHT_M),
                              horizon_frac=kw.get("horizon_frac", DEFAULT_HORIZON_FRAC),
                              hfov_deg=kw.get("hfov_deg", DEFAULT_HFOV_DEG))
    raise ValueError(f"unknown engine '{kind}' (use 'multitask' or 'baselines')")


def lateral_metres(u, distance_m, hfov_deg=DEFAULT_HFOV_DEG):
    """Sideways offset (m, + = right) of an image column u at a given distance."""
    return (u - IMG_W / 2) / focal_px(IMG_W, hfov_deg) * distance_m
