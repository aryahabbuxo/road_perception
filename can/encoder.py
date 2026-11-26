"""CAN message encoder using CAN DBC definitions.

Encodes outputs from MultiTaskWrapper.predict() and HazardTracker into standard
11-bit CAN messages at 10 Hz according to can/road_perception.dbc:
  - 0x100 OBJECT (up to 8, nearest first)
  - 0x110 ROAD_STATE
  - 0x120 RIDER_STATE
  - 0x130 STATUS
"""
import os
import cantools
import torch
from datasets.contract import IMG_W, REPORTED_DET_CLASSES

DBC_PATH = os.path.join(os.path.dirname(__file__), "road_perception.dbc")
_DB = None


def get_dbc():
    global _DB
    if _DB is None:
        _DB = cantools.database.load_file(DBC_PATH)
    return _DB


def _b(n):
    return int(round(n)) & 0xFF


def _hex(bytes_):
    return " ".join(f"{_b(x):02X}" for x in bytes_)


SHORT = {0: "pothole", 1: "breaker", 2: "car", 3: "2-wheel", 4: "bus", 5: "person"}
CURVE_SHORT = ["unk", "str", "gL", "sL", "gR", "sR"]
SURFACE_SHORT = ["asph", "conc", "pave", "unpv"]


class CANEncoder:
    """Encodes MultiTask predictions and Hazard tracks into CAN bytes/messages."""

    def __init__(self, dbc_path=None):
        self.db = cantools.database.load_file(dbc_path or DBC_PATH)
        self.msg_object = self.db.get_message_by_name("OBJECT")
        self.msg_road = self.db.get_message_by_name("ROAD_STATE")
        self.msg_rider = self.db.get_message_by_name("RIDER_STATE")
        self.msg_status = self.db.get_message_by_name("STATUS")

    def encode_tracks(self, tracks, curvature_code=0, bend_deg=0, surface=None,
                      lean=None, fps=30, counter=0, degraded=False, periodic_status=False):
        """Encodes tracking dictionaries into raw CAN payloads.
        Returns list of dicts: {"id": int, "data": bytes, "name": str}
        """
        k = counter % 16
        payloads = []

        # Sort nearest first (positive distance first, then unknown 0)
        sorted_tracks = sorted(
            tracks[:8],
            key=lambda t: (t.get("distance", 0) <= 0, t.get("distance", 0))
        )

        # 0x100 OBJECT (max 8)
        for i, t in enumerate(sorted_tracks):
            cls_id = int(t.get("cls", 0))
            if cls_id not in REPORTED_DET_CLASSES:
                continue

            dist = float(t.get("distance", 0.0))
            dist = max(0.0, min(dist, 6553.5))

            box = t.get("box", [0, 0, 0, 0])
            u = (box[0] + box[2]) / 2.0
            pos = int(round((u - IMG_W / 2.0) / (IMG_W / 2.0) * 100.0))
            pos = max(-100, min(100, pos))

            conf = int(round(float(t.get("conf", 1.0)) * 100.0))
            conf = max(0, min(100, conf))

            track_id = int(t.get("id", i)) & 0xFF

            data = self.msg_object.encode({
                "obj_index": i,
                "obj_class": cls_id,
                "obj_distance": dist,
                "obj_position": pos,
                "obj_confidence": conf,
                "obj_track_id": track_id,
                "obj_counter": k,
            })
            payloads.append({"id": 0x100, "data": data, "name": "OBJECT"})

        # 0x110 ROAD_STATE
        s_code, s_conf = surface if surface is not None else (7, 0.0)
        s_code = int(s_code) if 0 <= int(s_code) <= 3 else 7
        s_conf_pct = int(round(float(s_conf) * 100.0))
        data_road = self.msg_road.encode({
            "curvature_code": int(curvature_code) % 6,
            "bend_angle": max(-128, min(127, int(round(bend_deg)))),
            "surface_type": s_code,
            "surface_conf": max(0, min(100, s_conf_pct)),
            "road_counter": k,
        })
        payloads.append({"id": 0x110, "data": data_road, "name": "ROAD_STATE"})

        # 0x120 RIDER_STATE
        lean_deg = -128 if lean is None else int(round(lean))
        lean_deg = max(-128, min(127, lean_deg))
        # Pothole severity
        pothole_sevs = [t.get("severity", 0) or 0 for t in tracks if t.get("cls") == 0 and 0 < t.get("distance", 0) <= 30]
        worst_sev = max(pothole_sevs + [0])
        data_rider = self.msg_rider.encode({
            "rider_lean": lean_deg,
            "pothole_severity": min(5, max(0, int(worst_sev))),
            "rider_counter": k,
        })
        payloads.append({"id": 0x120, "data": data_rider, "name": "RIDER_STATE"})

        # 0x130 STATUS
        if not periodic_status or (k % 5 == 0):
            data_status = self.msg_status.encode({
                "system_fps": max(0, min(255, int(round(fps)))),
                "object_count": min(255, len(tracks)),
                "degraded_flag": 1 if degraded else 0,
                "status_counter": k,
            })
            payloads.append({"id": 0x130, "data": data_status, "name": "STATUS"})

        return payloads

    def encode_prediction(self, pred, fps=30, counter=0, degraded=False):
        """Encodes output dict from MultiTaskWrapper.predict() (for first batch item)."""
        boxes = pred.get("boxes", [])
        if isinstance(boxes, list):
            boxes = boxes[0] if len(boxes) > 0 else torch.zeros((0, 6))
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        distances = pred.get("box_distance", [])
        if isinstance(distances, list):
            distances = distances[0] if len(distances) > 0 else torch.zeros(len(boxes))
        if isinstance(distances, torch.Tensor):
            distances = distances.cpu().numpy()

        tracks = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2, conf, cls_id = box[:6]
            dist = float(distances[i]) if i < len(distances) else 0.0
            tracks.append({
                "id": i + 1,
                "cls": int(cls_id),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "distance": dist,
                "severity": 0,
            })

        surface_val = pred.get("surface")
        if isinstance(surface_val, torch.Tensor):
            surface_val = int(surface_val[0].item()) if surface_val.numel() > 0 else 7
        surface_conf = pred.get("surface_conf")
        if isinstance(surface_conf, torch.Tensor):
            surface_conf = float(surface_conf[0].item()) if surface_conf.numel() > 0 else 0.0
        surface_tuple = (surface_val, surface_conf) if surface_val is not None and surface_val >= 0 else (7, 0.0)

        lean = pred.get("lean")
        if isinstance(lean, torch.Tensor):
            lean = float(lean[0].item()) if lean.numel() > 0 else 0.0

        curve = pred.get("curvature")
        if isinstance(curve, torch.Tensor):
            curve = int(curve[0].item()) if curve.numel() > 0 else 0
        curve_deg = pred.get("curvature_deg")
        if isinstance(curve_deg, torch.Tensor):
            curve_deg = float(curve_deg[0].item()) if curve_deg.numel() > 0 else 0.0

        return self.encode_tracks(
            tracks=tracks,
            curvature_code=curve or 0,
            bend_deg=curve_deg or 0.0,
            surface=surface_tuple,
            lean=lean or 0.0,
            fps=fps,
            counter=counter,
            degraded=degraded,
        )


def encode_frame(tracks, curvature_code, bend_deg, surface, lean, fps, counter, degraded=False):
    """Drop-in replacement for app/can_frames.py encode_frame using the DBC encoder.
    Returns [{"id": "0x100", "data": "hex string", "text": "description"}].
    """
    encoder = CANEncoder()
    payloads = encoder.encode_tracks(
        tracks=tracks,
        curvature_code=curvature_code,
        bend_deg=bend_deg,
        surface=surface,
        lean=lean,
        fps=fps,
        counter=counter,
        degraded=degraded,
        periodic_status=True,
    )

    out = []
    db = get_dbc()
    for p in payloads:
        hex_data = " ".join(f"{b:02X}" for b in p["data"])
        msg_name = p["name"]
        msg_id_hex = f"0x{p['id']:03X}"

        decoded = db.decode_message(p["id"], p["data"])

        if msg_name == "OBJECT":
            cls_name = SHORT.get(decoded["obj_class"], f"c{decoded['obj_class']}")
            dist_str = f"{decoded['obj_distance']:.0f}m" if decoded['obj_distance'] > 0 else "?m"
            text = f"{cls_name} {dist_str}"
        elif msg_name == "ROAD_STATE":
            c_code = decoded["curvature_code"]
            s_type = decoded["surface_type"]
            c_str = CURVE_SHORT[c_code] if c_code < len(CURVE_SHORT) else "unk"
            s_str = SURFACE_SHORT[s_type] if s_type < len(SURFACE_SHORT) else "-"
            text = f"{c_str} {s_str}"
        elif msg_name == "RIDER_STATE":
            l_val = decoded["rider_lean"]
            sev = decoded["pothole_severity"]
            l_str = "-" if l_val == -128 else f"{l_val:.0f}°"
            text = f"lean {l_str} L{sev}"
        else:
            text = f"{decoded['object_count']} obj {'DEGRADED' if decoded['degraded_flag'] else 'ok'}"

        out.append({"id": msg_id_hex, "data": hex_data, "text": text})

    return out
