"""CAN module for Road Perception system (Saarth)."""
import sys
import os

local_can_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(local_can_dir)

# Temporarily remove project root from sys.path to load official python-can from site-packages
removed = []
for p in [parent_dir, "", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)
        removed.append(p)

if "can" in sys.modules:
    del sys.modules["can"]

import can as _can

for p in reversed(removed):
    sys.path.insert(0, p)

if local_can_dir not in _can.__path__:
    _can.__path__.append(local_can_dir)

sys.modules["can"] = _can

from can.encoder import CANEncoder, encode_frame, get_dbc
from can.comms import CANSender, CANReceiver, load_can_config

_can.CANEncoder = CANEncoder
_can.CANSender = CANSender
_can.CANReceiver = CANReceiver
_can.encode_frame = encode_frame
_can.get_dbc = get_dbc
_can.load_can_config = load_can_config

__all__ = ["CANEncoder", "CANSender", "CANReceiver", "encode_frame", "get_dbc", "load_can_config"]
