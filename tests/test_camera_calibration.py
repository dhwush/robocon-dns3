from types import SimpleNamespace

import pytest

from view_field import _camera_calibration


def calibration_profile(fx=610.0):
    raw = SimpleNamespace(width=1280, height=720, fx=fx, fy=611.0,
                          ppx=640.5, ppy=360.5, model="brown_conrady",
                          coeffs=[0.1, 0.0, 0.0, 0.0, 0.0])
    video = SimpleNamespace(get_intrinsics=lambda: raw)
    stream = SimpleNamespace(as_video_stream_profile=lambda: video)
    device = SimpleNamespace(get_info=lambda key: "camera-123")
    profile = SimpleNamespace(get_stream=lambda key: stream, get_device=lambda: device)
    rs = SimpleNamespace(stream=SimpleNamespace(color="color"),
                         camera_info=SimpleNamespace(serial_number="serial"))
    return profile, rs


def test_calibration_uses_active_stream_and_records_device():
    intr, metadata = _camera_calibration(*calibration_profile())
    assert (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy) == (
        1280, 720, 610.0, 611.0, 640.5, 360.5)
    assert metadata["serial_number"] == "camera-123"
    assert metadata["distortion_coefficients"] == [0.1, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("fx", [0.0, -1.0, float("nan")])
def test_invalid_device_calibration_is_rejected(fx):
    with pytest.raises(ValueError, match="内参无效"):
        _camera_calibration(*calibration_profile(fx))
