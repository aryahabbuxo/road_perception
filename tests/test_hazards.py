"""Tests for pothole severity, hazard tracking and rider alerts (models/hazards.py)."""
import math

import numpy as np
import torch

from datasets.contract import IMG_H, IMG_W
from models.hazards import (HazardTracker, focal_px, pothole_severity, rider_alerts,
                            road_width_px)


def road_mask(left=170, right=470):
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    m[200:, left:right] = 1                      # road 300 px wide in the lower part
    return m


def box_for_width_m(width_m, distance_m, x_centre=320, y_bottom=360, height=20):
    w = width_m * focal_px() / distance_m        # pixels this real width would cover
    return [x_centre - w / 2, y_bottom - height, x_centre + w / 2, y_bottom]


def test_focal_length():
    assert math.isclose(focal_px(640, 90.0), 320.0, rel_tol=1e-6)


def test_severity_from_real_width():
    for width_m, level in [(0.2, 1), (0.5, 2), (0.8, 3), (1.2, 4), (2.0, 5)]:
        s = pothole_severity(box_for_width_m(width_m, 12.0), 12.0)
        assert s["level"] == level, (width_m, s)
        assert math.isclose(s["width_m"], width_m, rel_tol=1e-6)


def test_same_pothole_same_level_at_any_distance():
    near = pothole_severity(box_for_width_m(0.8, 6.0), 6.0)["level"]
    far = pothole_severity(box_for_width_m(0.8, 25.0), 25.0)["level"]
    assert near == far == 3


def test_pothole_blocking_the_road_is_at_least_l4():
    box = [250, 340, 410, 360]                    # 160 px of a 300 px road = 53%
    s = pothole_severity(box, 0.0, road_mask())   # distance unknown: coverage alone decides
    assert s["level"] == 4 and s["coverage"] > 0.5


def test_unknown_when_no_distance_and_no_road():
    assert pothole_severity([300, 340, 340, 360], 0.0)["level"] == 0


def test_road_width():
    assert road_width_px(road_mask(), 300, 320) == 300
    assert road_width_px(road_mask(), 100, 320) == 0          # above the road


def frame(*objs):
    """objs: (x1, y1, x2, y2, conf, cls, distance)"""
    boxes = torch.tensor([o[:6] for o in objs], dtype=torch.float32).reshape(-1, 6)
    dist = torch.tensor([o[6] for o in objs], dtype=torch.float32)
    return boxes, dist


def test_tracker_keeps_id_and_confirms_after_3_frames():
    tr = HazardTracker()
    outs = [tr.update(*frame((300 + k, 330, 360 + k, 350, 0.9, 0, 15.0 - k))) for k in range(4)]
    assert outs[0] == [] and outs[1] == []                    # not confirmed yet
    assert len(outs[2]) == 1 and len(outs[3]) == 1
    assert outs[2][0]["id"] == outs[3][0]["id"] == 1


def test_tracker_separates_objects_and_classes():
    tr = HazardTracker(min_hits=1)
    out = tr.update(*frame((100, 300, 160, 340, 0.9, 0, 10.0),      # pothole
                           (100, 300, 160, 340, 0.8, 2, 10.0),      # car, same place
                           (400, 300, 460, 340, 0.9, 0, 20.0)))     # another pothole
    assert len({t["id"] for t in out}) == 3
    assert [t["distance"] for t in out] == [10.0, 10.0, 20.0]       # nearest first


def test_tracker_drops_lost_objects():
    tr = HazardTracker(min_hits=1, max_misses=2)
    tr.update(*frame((100, 300, 160, 340, 0.9, 2, 10.0)))
    for _ in range(3):
        out = tr.update(*frame())
    assert out == [] and tr.tracks == []


def test_one_frame_spike_does_not_change_severity():
    tr = HazardTracker(min_hits=1)
    widths = [0.5, 0.5, 2.0, 0.5, 0.5]                             # L2, L2, L5 (noise), L2, L2
    for w in widths:
        out = tr.update(*frame((*box_for_width_m(w, 12.0), 0.9, 0, 12.0)))
    assert out[0]["severity"] == 2


def test_alerts():
    tracks = [
        {"id": 1, "cls": 0, "name": "pothole", "box": [], "conf": 0.9, "distance": 15.0, "severity": 3},
        {"id": 2, "cls": 0, "name": "pothole", "box": [], "conf": 0.9, "distance": 12.0, "severity": 1},
        {"id": 3, "cls": 1, "name": "speed_breaker", "box": [], "conf": 0.9, "distance": 30.0, "severity": None},
        {"id": 4, "cls": 5, "name": "person", "box": [], "conf": 0.9, "distance": 6.0, "severity": None},
    ]
    alerts = rider_alerts(tracks)
    assert [a["id"] for a in alerts] == [4, 1]                    # small pothole and far breaker ignored
    assert alerts[1]["alert"] == "Pothole L3 in 15 m"
    assert alerts[0]["alert"] == "Person close: 6 m"
