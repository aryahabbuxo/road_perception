"""Joint test (owned by Person A). Uses fake data now; on Integration Day 1
the fake dataset is replaced by Person B's MultiTaskDataset."""
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.contract import IMG_H, IMG_W, empty_labels, check_sample, collate_fn
from models.wrappers import YoloBaseline


class FakeDataset(Dataset):
    """Samples that follow the contract, with random content."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        s = {"image": torch.rand(3, IMG_H, IMG_W), **empty_labels(),
             "has_det": True, "has_seg": True, "has_depth": True,
             "dataset": "fake", "stem": f"fake_{i}"}
        s["boxes"] = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        s["labels"] = torch.tensor([0])
        s["seg_mask"] = torch.randint(0, 2, (IMG_H, IMG_W))
        s["depth"] = torch.rand(IMG_H, IMG_W) * 80
        return s


def check_prediction(pred, batch_size):
    """Check a model's output follows Rule 5."""
    assert {"boxes", "seg", "depth"} <= set(pred)
    assert len(pred["boxes"]) == batch_size
    for b in pred["boxes"]:
        assert b.ndim == 2 and b.shape[1] == 6
    if pred["seg"] is not None:
        assert pred["seg"].shape == (batch_size, IMG_H, IMG_W)
    if pred["depth"] is not None:
        assert pred["depth"].shape == (batch_size, IMG_H, IMG_W)


def test_data_follows_contract():
    ds = FakeDataset()
    for i in range(len(ds)):
        check_sample(ds[i])


def test_batch_shapes():
    batch = next(iter(DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)))
    assert batch["img"].shape == (2, 3, IMG_H, IMG_W)
    assert batch["det_targets"].shape[1] == 6


def test_yolo_baseline_output():
    batch = next(iter(DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)))
    model = YoloBaseline("yolo11n.pt", device="cpu")
    check_prediction(model.predict(batch["img"]), batch_size=2)
    

def test_seg_baseline_output():
    from models.wrappers import SegBaseline

    batch = next(iter(DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)))
    model = SegBaseline(device="cpu")   # untrained, no download needed
    pred = model.predict(batch["img"])
    check_prediction(pred, batch_size=2)
    assert pred["seg"].dtype == torch.int64


def test_multitask_output():
    from datasets.contract import REPORTED_DET_CLASSES
    from models.wrappers import MultiTaskWrapper

    batch = next(iter(DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)))
    model = MultiTaskWrapper(device="cpu", conf=0.0)   # untrained; conf 0 so boxes appear
    pred = model.predict(batch["img"])
    check_prediction(pred, batch_size=2)
    assert pred["depth"].min() >= 0 and pred["depth"].max() <= 80
    for b, d in zip(pred["boxes"], pred["box_distance"]):
        assert len(d) == len(b)
        assert set(b[:, 5].long().tolist()) <= set(REPORTED_DET_CLASSES)   # no lights/signs
    assert pred["surface"].shape == (2,) and pred["lean"].shape == (2,)
    assert pred["curvature"].shape == (2,) and pred["curvature"].dtype == torch.int64


def test_multitask_loss_skips_missing_labels():
    from losses.multitask_loss import MultiTaskLoss
    from models.multitask import MultiTaskNet
    from train.lean_aug import random_lean

    model = MultiTaskNet(pretrained=False).train()
    criterion = MultiTaskLoss(model)
    batch = next(iter(DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)))
    batch["has_seg"][:] = False          # pretend this batch has no road masks
    batch["surface"][:] = torch.tensor([2, -1])
    batch = random_lean(batch, p=1.0)
    total, parts = criterion(model(batch["img"]), batch)
    total.backward()
    assert parts["seg"] is None
    assert all(parts[t] is not None for t in ("det", "depth", "surface", "lean"))
    assert torch.isfinite(total)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.seg_head.parameters())
    for head in (model.depth_head, model.surface_head, model.lean_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_lean_tilt_moves_labels_with_image():
    from train.lean_aug import apply_lean

    img = torch.zeros(1, 3, IMG_H, IMG_W)
    y0, x0 = IMG_H // 2, IMG_W // 2 + 200          # bright dot right of centre, inside a box
    img[0, :, y0 - 3:y0 + 4, x0 - 3:x0 + 4] = 1
    batch = {"img": img, "seg_mask": torch.ones(1, IMG_H, IMG_W, dtype=torch.long),
             "depth": torch.full((1, IMG_H, IMG_W), 10.0),
             "det_targets": torch.tensor([[0, 2, x0 / IMG_W, y0 / IMG_H, 20 / IMG_W, 20 / IMG_H]])}
    out = apply_lean(batch, torch.tensor([30.0]))
    ys, xs = torch.nonzero(out["img"][0, 0] > 0.8, as_tuple=True)
    dy, dx = ys.float().mean().item(), xs.float().mean().item()
    assert dy < y0 - 50                             # leaning right: scene turns anticlockwise
    _, _, cx, cy, w, h = out["det_targets"][0].tolist()
    assert abs(dx - cx * IMG_W) < w * IMG_W / 2 and abs(dy - cy * IMG_H) < h * IMG_H / 2
    assert out["seg_mask"][0, 0, 0] == 255 and out["depth"][0, 0, 0] == 0   # corners: no label
    assert out["lean"].tolist() == [30.0]


def test_curvature_direction():
    import numpy as np

    from models.postprocess import road_curvature

    def road(curve):
        m = np.zeros((IMG_H, IMG_W), np.uint8)
        for y in range(int(IMG_H * 0.45), IMG_H):
            t = (IMG_H - y) / (IMG_H * 0.55)
            half, c = 260 * (1 - t) + 20, IMG_W / 2 + curve * t ** 2 * IMG_W
            m[y, max(0, int(c - half)):min(IMG_W, int(c + half))] = 1
        return m

    assert road_curvature(road(0.0))[0] == 1           # straight
    assert road_curvature(road(0.35))[0] == 5          # sharp right
    assert road_curvature(road(-0.12))[0] == 2         # gentle left
    assert road_curvature(np.zeros((IMG_H, IMG_W)))[0] == 0   # no road: unknown
