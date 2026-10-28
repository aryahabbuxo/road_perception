"""Every model is wrapped so that predict(img) returns the same format (Rule 5):
  boxes: list of B tensors [K, 6] = x1, y1, x2, y2, score, class (pixels)
  seg:   int64 [B, 384, 640] (0 = not road, 1 = road), or None
  depth: float32 [B, 384, 640] in metres, or None
img is float32 [B, 3, 384, 640], values 0-1."""
import torch
from ultralytics import YOLO

from datasets.contract import IMG_H, IMG_W


class ModelWrapper:
    tasks = ()  # which jobs this model does: "det", "seg", "depth"

    def predict(self, img):
        raise NotImplementedError


class YoloBaseline(ModelWrapper):
    tasks = ("det",)

    def __init__(self, weights, conf=0.25, iou=0.7, device=None):
        self.model = YOLO(weights)
        self.conf, self.iou = conf, iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(self, img):
        results = self.model.predict(img, imgsz=(IMG_H, IMG_W), conf=self.conf,
                                     iou=self.iou, device=self.device, verbose=False)
        boxes = []
        for r in results:
            b = r.boxes
            if len(b):
                boxes.append(torch.cat([b.xyxy, b.conf[:, None], b.cls[:, None]], 1).cpu())
            else:
                boxes.append(torch.zeros((0, 6)))
        return {"boxes": boxes, "seg": None, "depth": None}