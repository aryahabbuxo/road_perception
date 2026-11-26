"""CAN bus Sender and Receiver with DBC encoding and alert generation."""
import os
import yaml
import can
import cantools

from can.encoder import CANEncoder, DBC_PATH

DEFAULT_CONFIG = {
    "interface": "virtual",
    "channel": 0,
    "bitrate": 500000,
    "dbc_path": DBC_PATH,
}


def load_can_config(config_path="can/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return {**DEFAULT_CONFIG, **(cfg or {})}
    return DEFAULT_CONFIG


class CANSender:
    """Sends encoded messages over CAN bus (virtual or hardware)."""

    def __init__(self, config_path="can/config.yaml"):
        self.cfg = load_can_config(config_path)
        self.encoder = CANEncoder(self.cfg["dbc_path"])
        self.bus = can.Bus(
            interface=self.cfg["interface"],
            channel=self.cfg["channel"],
            bitrate=self.cfg["bitrate"],
        )

    def send_tracks(self, tracks, curvature_code=0, bend_deg=0, surface=None,
                    lean=0.0, fps=30, counter=0, degraded=False):
        payloads = self.encoder.encode_tracks(
            tracks=tracks,
            curvature_code=curvature_code,
            bend_deg=bend_deg,
            surface=surface,
            lean=lean,
            fps=fps,
            counter=counter,
            degraded=degraded,
        )
        for p in payloads:
            msg = can.Message(
                arbitration_id=p["id"],
                data=p["data"],
                is_extended_id=False,
            )
            self.bus.send(msg)

    def send_prediction(self, pred, fps=30, counter=0, degraded=False):
        payloads = self.encoder.encode_prediction(pred, fps=fps, counter=counter, degraded=degraded)
        for p in payloads:
            msg = can.Message(
                arbitration_id=p["id"],
                data=p["data"],
                is_extended_id=False,
            )
            self.bus.send(msg)

    def close(self):
        self.bus.shutdown()


class CANReceiver:
    """Receives CAN messages, decodes with DBC and generates rider alerts."""

    def __init__(self, config_path="can/config.yaml"):
        self.cfg = load_can_config(config_path)
        self.db = cantools.database.load_file(self.cfg["dbc_path"])
        self.bus = can.Bus(
            interface=self.cfg["interface"],
            channel=self.cfg["channel"],
            bitrate=self.cfg["bitrate"],
        )

    def receive_and_decode(self, timeout=1.0):
        """Receives a single message from the bus and decodes it.
        Returns (msg_name, decoded_dict) or None if timeout.
        """
        msg = self.bus.recv(timeout=timeout)
        if msg is None:
            return None
        try:
            db_msg = self.db.get_message_by_frame_id(msg.arbitration_id)
            decoded = db_msg.decode(msg.data)
            return db_msg.name, decoded
        except Exception:
            return None

    def check_alerts(self, decoded_msg):
        """Analyzes decoded message fields and returns human-readable alerts."""
        alerts = []
        name, data = decoded_msg

        if name == "OBJECT":
            dist = data.get("obj_distance", 0.0)
            cls_id = data.get("obj_class", 0)
            if 0 < dist < 10.0:
                alerts.append(f"CRITICAL: Close object (class {cls_id}) at {dist:.1f} m!")
            elif 0 < dist < 20.0 and cls_id in (0, 1):
                alerts.append(f"WARNING: Road hazard ahead at {dist:.1f} m!")

        elif name == "ROAD_STATE":
            code = data.get("curvature_code", 0)
            bend = abs(data.get("bend_angle", 0))
            surf = data.get("surface_type", 7)
            if code in (3, 5) or bend > 25:
                alerts.append(f"ALERT: Sharp bend ahead ({bend}°)")
            if surf == 3:
                alerts.append("ALERT: Unpaved surface ahead, reduce speed!")
            elif surf == 2:
                alerts.append("CAUTION: Paving stones/slippery surface!")

        elif name == "RIDER_STATE":
            sev = data.get("pothole_severity", 0)
            lean = abs(data.get("rider_lean", 0))
            if sev >= 3:
                alerts.append(f"HAZARD: Severe pothole (Level {sev}) detected!")
            if lean > 25:
                alerts.append(f"RIDER: High lean angle ({lean}°)")

        elif name == "STATUS":
            if data.get("degraded_flag", 0) == 1:
                alerts.append("SYSTEM: Perception degraded mode active!")

        return alerts

    def close(self):
        self.bus.shutdown()
