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


class SegBaseline(ModelWrapper):
    """SegFormer road baseline. checkpoint=None gives an untrained model (for tests)."""
    tasks = ("seg",)

    def __init__(self, checkpoint=None, device=None):
        from models.segformer_baseline import SegFormerRoad

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = "nvidia/mit-b0"
        state = None
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            backbone = ckpt["config"]["model"]
            state = ckpt["model"]
        self.model = SegFormerRoad(backbone, pretrained=False)
        if state is not None:
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, img):
        logits = self.model(img.to(self.device))
        seg = logits.argmax(1).cpu()
        boxes = [torch.zeros((0, 6)) for _ in range(img.shape[0])]
        return {"boxes": boxes, "seg": seg, "depth": None}


class MultiTaskWrapper(ModelWrapper):
    """The combined model. checkpoint=None gives an untrained model (for tests).

    predict() returns the Rule 5 keys (boxes, seg, depth) plus, per image:
      box_distance: list of [K] tensors, metres to each object (0 = unknown)
      surface:      int64 [B] surface type (SURFACE_NAMES), surface_conf: float [B]
      lean:         float [B] degrees, positive = leaning right
      curvature:    int64 [B] code (CURVATURE_NAMES), curvature_deg: float [B]
    Boxes only include REPORTED_DET_CLASSES (traffic lights and signs are left out for now).
    """

    def __init__(self, checkpoint=None, device=None, conf=0.25, iou=0.7, all_classes=False):
        from models.multitask import ALL_TASKS, MultiTaskNet

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf, self.iou, self.all_classes = conf, iou, all_classes
        tasks, state = ALL_TASKS, None
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            tasks, state = tuple(ckpt["config"]["tasks"]), ckpt["model"]
        self.model = MultiTaskNet(pretrained=False, tasks=tasks)
        if state is not None:
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tasks = tasks

    @torch.no_grad()
    def predict(self, img):
        from ultralytics.utils.nms import non_max_suppression

        from datasets.contract import REPORTED_DET_CLASSES
        from models.postprocess import box_distances, road_curvature

        out = self.model(img.to(self.device))
        dets = non_max_suppression(out["det"][0].float(), conf_thres=self.conf, iou_thres=self.iou)
        boxes = []
        for d in dets:
            d = d[:, :6].float().cpu()
            if not self.all_classes:
                d = d[torch.isin(d[:, 5].long(), torch.tensor(REPORTED_DET_CLASSES))]
            boxes.append(d)
        B = img.shape[0]
        seg = out["seg"].argmax(1).cpu() if out["seg"] is not None else None
        depth = out["depth"][:, 0].float().cpu() if out["depth"] is not None else None
        lean = out["lean"].float().cpu() if out["lean"] is not None else None
        res = {"boxes": boxes, "seg": seg, "depth": depth, "lean": lean,
               "box_distance": [box_distances(depth[i], boxes[i]) if depth is not None
                                else torch.zeros(len(boxes[i])) for i in range(B)]}
        if out["surface"] is not None:
            prob = out["surface"].float().softmax(1).cpu()
            res["surface_conf"], res["surface"] = prob.max(1)
        else:
            res["surface"], res["surface_conf"] = None, None
        if seg is not None:
            curv = [road_curvature(seg[i], lean_deg=float(lean[i]) if lean is not None else 0.0)
                    for i in range(B)]
            res["curvature"] = torch.tensor([c for c, _ in curv], dtype=torch.int64)
            res["curvature_deg"] = torch.tensor([a for _, a in curv], dtype=torch.float32)
        else:
            res["curvature"], res["curvature_deg"] = None, None
        return res
