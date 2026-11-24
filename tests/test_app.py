"""Tests for the Saarth demo app (app/): geometry, the rider's-bike handling, smoothing,
CAN frames and the video processor end to end (with a stand-in engine, no checkpoints needed)."""
import argparse
import json
import math

import cv2
import numpy as np
import torch

from app import process_video
from app.can_frames import encode_frame
from app.scene import (CropMap, CurveSmoother, MaskSmoother, drop_ego_boxes, edges_mask,
                       estimate_ego, focal_model, ground_distance, lb_geom, road_edges)
from datasets.contract import IMG_H, IMG_W

W, H = 640, 360                      # test video size (16:9, like most clips)
HORIZON = 0.30                       # where the road meets the sky in the test video
TANK_TOP = 0.62                      # the bike's tank / dashboard covers the frame below this


def test_ground_distance_flat_road():
    f = focal_model(1280, 720, 70.0)
    assert math.isclose(ground_distance(100 + f / 10, 100, f, 1.0), 10.0, rel_tol=1e-6)
    assert ground_distance(90, 100, f, 1.0) == 0.0               # above the horizon: unknown
    assert ground_distance(102, 100, f, 1.0) == 80.0             # capped


def test_crop_maps_back_to_full_frame():
    full, crop = CropMap(W, H, 1.0), CropMap(W, H, 0.5)
    assert crop.bottom_row < full.bottom_row
    # a box at the bottom of the cropped view lands half way down the full frame
    (px, py), (nw, nh) = crop.pc, crop.nc
    b = crop.boxes([[px, py + nh - 10, px + 20, py + nh, 0.9, 2]])
    assert abs(b[0, 3] - (full.pf[1] + 0.5 * full.nf[1])) < 2
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    m[py:py + nh, px:px + nw] = 1                                  # everything the cropped view saw
    back = crop.mask(m)
    rows = np.flatnonzero(back.any(1))
    assert abs(rows[0] - full.pf[1]) <= 1 and abs(rows[-1] - crop.bottom_row) <= 2


def test_bike_parts_are_not_objects():
    crop = CropMap(W, H, 1.0)
    (pl, pt), (nw, nh) = crop.pf, crop.nf
    bottom = pt + nh - 1
    boxes = np.array([[pl + 5, pt + 5, pl + nw - 5, bottom, 0.6, 2],      # the whole bike
                      [pl, pt + 100, pl + 90, bottom, 0.5, 2],            # mirror + hand at the side
                      [300, pt + 120, 340, pt + 150, 0.8, 2],             # a real car ahead
                      [280, bottom - 60, 360, bottom, 0.8, 2]])           # a real car very close, centred
    assert drop_ego_boxes(boxes, np.zeros((IMG_H, IMG_W), np.uint8), crop) == [2, 3]


def test_road_edges_follow_road_and_stop_at_the_bike():
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(m, [np.array([[60, 383], [580, 383], [340, 120], [300, 120]], np.int32)], 1)
    m[300:, 200:440] = 0                                           # tank splits the near road
    m[330:370, 20:90] = 1                                          # stray blob (a glove marked as road)
    e = road_edges(m)
    ys = [r[0] for r in e]
    assert ys == sorted(ys, reverse=True)                          # nearest row first
    assert min(ys) <= 126 and max(ys) < 300                        # from the far end down to the tank
    assert all(r[1] < 320 < r[2] for r in e)                       # straight road centred ahead
    assert edges_mask(e)[200, 320] == 1 and edges_mask(e)[350, 50] == 0


def test_smoothers():
    ms = MaskSmoother(tau=0.3)
    on, off = np.ones((4, 4)), np.zeros((4, 4))
    ms.update(on, 1 / 30)
    assert ms.update(off, 1 / 30).all()                            # one bad frame does not show
    cs = CurveSmoother(tau=0.5, hold=1.0)
    assert cs.update(1, 2.0, 1 / 30) == (1, 2.0)
    assert cs.update(0, 0.0, 0.5)[0] == 1                          # held while briefly lost
    assert cs.update(0, 0.0, 0.6) == (0, 0.0)                      # then unknown


def test_can_frames():
    tracks = [{"id": 3, "cls": 0, "box": [300, 300, 340, 320], "conf": 0.8, "distance": 12.3, "severity": 3}]
    msgs = encode_frame(tracks, 4, 12.0, None, None, 9.5, counter=0)
    assert [m["id"] for m in msgs] == ["0x100", "0x110", "0x120", "0x130"]
    assert all(len(m["data"].split()) == 8 for m in msgs)
    obj = [int(b, 16) for b in msgs[0]["data"].split()]
    assert obj[1] == 0 and obj[2] + (obj[3] << 8) == 123
    road = [int(b, 16) for b in msgs[1]["data"].split()]
    assert road[0] == 4 and road[2] == 7
    rider = [int(b, 16) for b in msgs[2]["data"].split()]
    assert rider[0] == 0x80 and rider[1] == 3


# ---------- end to end: a ride filmed with the bike in view ----------

def road_poly():
    return np.array([[30, H], [W - 30, H], [W // 2 + 12, int(HORIZON * H)], [W // 2 - 12, int(HORIZON * H)]], np.int32)


def make_bike_video(path, n=90, fps=30):
    """Moving road stripes and trees, with a still tank, handlebars and mirrors in front."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    rng = np.random.default_rng(0)
    trees = cv2.resize(rng.integers(30, 160, (H // 8, W // 8, 3)).astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST)
    asphalt = cv2.resize(rng.integers(95, 145, (H // 4, W // 4)).astype(np.uint8), (W, H),
                         interpolation=cv2.INTER_NEAREST)
    for i in range(n):
        f = np.zeros((H, W, 3), np.uint8)
        f[:] = np.roll(trees, i * 13, axis=1)                         # trees and fields sliding past
        road = np.zeros((H, W), np.uint8)
        cv2.fillPoly(road, [road_poly()], 1)
        tex = np.roll(asphalt, i * 11, axis=0)                      # asphalt texture rushing past
        f[road == 1] = np.repeat(tex[road == 1][:, None], 3, axis=1)
        for k in range(8):                                          # dashes rushing towards the rider
            y = int(HORIZON * H) + ((k * 30 + i * 9) % (H - int(HORIZON * H)))
            cv2.line(f, (W // 2 - 2, y), (W // 2 + 2, y + 8), (240, 240, 240), 3)
            cv2.circle(f, (int(rng.integers(0, W)), int(rng.integers(int(HORIZON * H), H))), 6,
                       (int(rng.integers(60, 200)),) * 3, -1)
        # the rider's own bike: never moves in the picture
        cv2.ellipse(f, (W // 2, H + 40), (150, int((1 - TANK_TOP) * H) + 40), 0, 180, 360, (20, 20, 25), -1)
        cv2.rectangle(f, (60, int(0.66 * H)), (W - 60, int(0.70 * H)), (30, 30, 30), -1)       # handlebar
        cv2.rectangle(f, (0, int(0.70 * H)), (70, H), (25, 25, 25), -1)                        # gloves
        cv2.rectangle(f, (W - 70, int(0.70 * H)), (W, H), (25, 25, 25), -1)
        vw.write(f)
    vw.release()


class BikeEngine:
    """Stand-in for the trained models: finds the road (but also marks the glove as road, like the
    real SegFormer did), calls the whole bike a car, and sees one real car ahead."""
    name = "baselines"

    def __call__(self, img):
        x = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        c = x.astype(int)
        grey = ((abs(c[..., 0] - c[..., 1]) < 10) & (abs(c[..., 1] - c[..., 2]) < 10)
                & (c[..., 0] >= 88) & (c[..., 0] <= 152) & (c[..., 0] != 114))
        road = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ys = np.flatnonzero((x[:, :, 0] != 114).any(1))
        top, bot = ys.min(), ys.max()
        road[bot - 20:bot + 1, 0:40] = 1                             # the "glove is road" mistake
        boxes = torch.tensor([[2, top + 5, 637, bot, 0.6, 2],        # the whole bike as a car
                              [300, top + 125, 340, top + 145, 0.8, 2]], dtype=torch.float32)
        return {"boxes": boxes, "box_distance": torch.zeros(2), "seg": torch.from_numpy(road).long(),
                "surface": None, "lean": None}


def run_args(video, name, **kw):
    a = dict(video=str(video), name=name, engine="baselines", checkpoint=None, yolo="x", seg="x",
             device="cpu", conf=0.25, every=1, max_seconds=None, hfov=70.0, camera_height=1.0,
             horizon="auto", keep_top="auto", ego=True)
    a.update(kw)
    return argparse.Namespace(**a)


def test_estimate_ego_finds_the_bike(tmp_path):
    video = tmp_path / "bike.avi"
    make_bike_video(video, n=60)
    from app.scene import sample_frames
    ego = estimate_ego(sample_frames(str(video), n=48))
    s, (pl, pt), (nw, nh) = lb_geom(W, H)
    assert ego[pt + nh - 3, pl + nw // 2] == 1                     # tank
    assert ego[pt + int(0.4 * nh), pl + nw // 2] == 0              # road ahead
    assert 0.1 < ego.mean() < 0.5


def test_process_bike_video_end_to_end(tmp_path, monkeypatch):
    video = tmp_path / "bike.avi"
    make_bike_video(video)
    monkeypatch.setattr(process_video, "WEB_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(process_video, "make_engine", lambda kind, **kw: BikeEngine())
    monkeypatch.setattr(process_video, "browser_copy", lambda *a, **k: True)
    out = process_video.process(run_args(video, "bike"))

    data = json.load(open(f"{out}/frames.json"))
    meta, frames = data["meta"], data["frames"]
    s, (pl, pt), (nw, nh) = lb_geom(W, H)
    assert meta["ego_share"] > 0.1 and meta["keep_top"] < 0.9      # bike found and cropped off
    assert abs((meta["horizon_row"] - pt) / nh - HORIZON) < 0.05   # horizon found from the road
    last = frames[-1]
    assert [o[1] for o in last["objs"]] == [2]                     # only the real car, not the bike
    assert last["objs"][0][6] > 5                                  # with a sensible distance
    assert last["curv"][0] == 1                                    # straight road
    assert len(last["edges"]) > 10 and all(e[1] > 40 for e in last["edges"])   # glove ignored
    assert json.load(open(tmp_path / "runs" / "index.json"))[0]["name"] == "bike"


def test_every_nth_frame(tmp_path, monkeypatch):
    video = tmp_path / "bike.avi"
    make_bike_video(video, n=12)
    monkeypatch.setattr(process_video, "WEB_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(process_video, "make_engine", lambda kind, **kw: BikeEngine())
    monkeypatch.setattr(process_video, "browser_copy", lambda *a, **k: True)
    frames = json.load(open(f"{process_video.process(run_args(video, 't2', every=3, ego=False, keep_top='1'))}/frames.json"))["frames"]
    assert [f["t"] for f in frames] == [0.0, 0.1, 0.2, 0.3]
