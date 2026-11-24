"""CAN frames for one video frame, in the layout proposed for Saarth (classic CAN, 8 data
bytes, little-endian). When Person B's can/ module and .dbc file are merged, replace
encode_frame() with their encoder so the demo shows exactly the frames the DBC defines.

  0x100 OBJECT       index, class, distance (0.1 m, 2 bytes), position (-100..100 % from centre),
                     confidence %, track id, counter        (one per object, nearest first, max 8)
  0x110 ROAD_STATE   curvature code, bend (deg), surface code (7 = unknown), surface confidence %,
                     0, 0, 0, counter
  0x120 RIDER_STATE  lean (deg, -128 = unknown), worst pothole severity within 30 m, 0 x5, counter
  0x130 STATUS       frames per second, object count, degraded flag, 0 x4, counter
"""
from datasets.contract import IMG_W

SHORT = {0: "pothole", 1: "breaker", 2: "car", 3: "2-wheel", 4: "bus", 5: "person"}
CURVE_SHORT = ["unk", "str", "gL", "sL", "gR", "sR"]
SURFACE_SHORT = ["asph", "conc", "pave", "unpv"]


def _b(n):
    return int(round(n)) & 0xFF


def _hex(bytes_):
    return " ".join(f"{_b(x):02X}" for x in bytes_)


def encode_frame(tracks, curvature_code, bend_deg, surface, lean, fps, counter, degraded=False):
    """tracks: HazardTracker output (dicts with cls, box, distance, conf, id, severity).
    surface: (code, conf) or None. lean: degrees or None. Returns [{"id", "data", "text"}]."""
    k = counter % 16
    out = []
    for i, t in enumerate(tracks[:8]):
        dm = int(round(t["distance"] * 10)) if t["distance"] > 0 else 0
        u = (t["box"][0] + t["box"][2]) / 2
        pos = round((u - IMG_W / 2) / (IMG_W / 2) * 100)
        data = [i, t["cls"], dm & 0xFF, dm >> 8, pos, t["conf"] * 100, t["id"], k]
        dist = f"{t['distance']:.0f}m" if t["distance"] > 0 else "?m"
        out.append({"id": "0x100", "data": _hex(data), "text": f"{SHORT.get(t['cls'], t['cls'])} {dist}"})
    s_code, s_conf = surface if surface is not None else (7, 0.0)
    out.append({"id": "0x110", "data": _hex([curvature_code, bend_deg, s_code, s_conf * 100, 0, 0, 0, k]),
                "text": f"{CURVE_SHORT[curvature_code]} {SURFACE_SHORT[s_code] if s_code < 4 else '-'}"})
    sev = max([t["severity"] or 0 for t in tracks if t["cls"] == 0 and 0 < t["distance"] <= 30] + [0])
    out.append({"id": "0x120", "data": _hex([-128 if lean is None else lean, sev, 0, 0, 0, 0, 0, k]),
                "text": f"lean {'-' if lean is None else f'{lean:.0f}°'} L{sev}"})
    if k % 5 == 0:
        out.append({"id": "0x130", "data": _hex([fps, len(tracks), int(degraded), 0, 0, 0, 0, k]),
                    "text": f"{len(tracks)} obj {'DEGRADED' if degraded else 'ok'}"})
    return out
