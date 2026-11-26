"""Tests for can/ module: DBC validation, encode -> send -> receive -> decode."""
import pytest
from can.encoder import CANEncoder, encode_frame, get_dbc
from can.comms import CANSender, CANReceiver


def test_dbc_loading():
    db = get_dbc()
    expected = {"OBJECT", "ROAD_STATE", "RIDER_STATE", "STATUS"}
    assert expected.issubset(set(m.name for m in db.messages))


def test_encoder_tracks():
    encoder = CANEncoder()
    tracks = [
        {"id": 1, "cls": 0, "box": [100, 100, 200, 200], "conf": 0.85, "distance": 8.5, "severity": 3},
        {"id": 2, "cls": 2, "box": [300, 150, 400, 250], "conf": 0.92, "distance": 22.0, "severity": 0},
    ]
    payloads = encoder.encode_tracks(
        tracks=tracks,
        curvature_code=2,
        bend_deg=-15,
        surface=(0, 0.95),
        lean=12.5,
        fps=30,
        counter=7,
        degraded=False,
    )

    names = [p["name"] for p in payloads]
    assert "OBJECT" in names
    assert "ROAD_STATE" in names
    assert "RIDER_STATE" in names
    assert "STATUS" in names
    # 2 objects + 3 state messages = 5 messages
    assert len(payloads) == 5


def test_can_virtual_bus_roundtrip():
    """Verify that a message survives encode -> send -> receive -> decode over virtual bus."""
    sender = CANSender()
    receiver = CANReceiver()

    tracks = [
        {"id": 5, "cls": 0, "box": [250, 150, 350, 250], "conf": 0.88, "distance": 6.4, "severity": 4}
    ]

    sender.send_tracks(
        tracks=tracks,
        curvature_code=3,
        bend_deg=-30,
        surface=(3, 0.90),
        lean=15.0,
        fps=28,
        counter=1,
    )

    received_messages = {}
    all_alerts = []

    # Drain sent messages (up to 4 messages sent: 1 OBJECT + 3 states)
    for _ in range(10):
        res = receiver.receive_and_decode(timeout=0.3)
        if res is None:
            break
        name, data = res
        received_messages[name] = data
        alerts = receiver.check_alerts(res)
        all_alerts.extend(alerts)

    sender.close()
    receiver.close()

    assert "OBJECT" in received_messages
    assert "ROAD_STATE" in received_messages
    assert "RIDER_STATE" in received_messages
    assert "STATUS" in received_messages

    # Verify signal values survive roundtrip
    obj_data = received_messages["OBJECT"]
    assert obj_data["obj_class"] == 0
    assert abs(obj_data["obj_distance"] - 6.4) < 0.15
    assert obj_data["obj_track_id"] == 5
    assert obj_data["obj_counter"] == 1

    road_data = received_messages["ROAD_STATE"]
    assert road_data["curvature_code"] == 3
    assert road_data["bend_angle"] == -30
    assert road_data["surface_type"] == 3

    rider_data = received_messages["RIDER_STATE"]
    assert rider_data["pothole_severity"] == 4
    assert rider_data["rider_lean"] == 15

    # Check alert generation
    assert any("pothole" in a.lower() for a in all_alerts)
    assert any("unpaved" in a.lower() for a in all_alerts)


def test_encode_frame_demo_app_format():
    """Verify encode_frame matches app/can_frames.py format."""
    tracks = [
        {"id": 1, "cls": 2, "box": [100, 100, 200, 200], "conf": 0.9, "distance": 14.2, "severity": 0}
    ]
    frames = encode_frame(
        tracks=tracks,
        curvature_code=1,
        bend_deg=0,
        surface=(0, 0.99),
        lean=0,
        fps=30,
        counter=0,
    )
    assert len(frames) >= 4
    for f in frames:
        assert "id" in f and f["id"].startswith("0x")
        assert "data" in f
        assert "text" in f
