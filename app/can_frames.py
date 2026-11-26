"""CAN frames for one video frame using the CAN module and .dbc file (Person B, Nov 2026).

Encodes outputs from HazardTracker and model predictions into standard 11-bit CAN messages:
  0x100 OBJECT       index, class, distance (0.1 m, 2 bytes), position (-100..100 % from centre),
                     confidence %, track id, counter        (one per object, nearest first, max 8)
  0x110 ROAD_STATE   curvature code, bend (deg), surface code (7 = unknown), surface confidence %,
                     0, 0, 0, counter
  0x120 RIDER_STATE  lean (deg, -128 = unknown), worst pothole severity within 30 m, 0 x5, counter
  0x130 STATUS       frames per second, object count, degraded flag, 0 x4, counter
"""
from can.encoder import encode_frame

__all__ = ["encode_frame"]
