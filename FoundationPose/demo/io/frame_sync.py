from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TimedFrame:
  data: Any
  timestamp_ms: float
  frame_id: int


class TimestampSyncer:
  """Pair RGB/Depth/Mask frames by nearest timestamp in a tolerance window."""

  def __init__(self, tolerance_ms: float = 33.0):
    self.tolerance_ms = float(tolerance_ms)
    self._rgb: Optional[TimedFrame] = None
    self._depth: Optional[TimedFrame] = None
    self._mask: Optional[TimedFrame] = None

  def update_rgb(self, data: Any, timestamp_ms: float, frame_id: int) -> None:
    self._rgb = TimedFrame(data=data, timestamp_ms=timestamp_ms, frame_id=frame_id)

  def update_depth(self, data: Any, timestamp_ms: float, frame_id: int) -> None:
    self._depth = TimedFrame(data=data, timestamp_ms=timestamp_ms, frame_id=frame_id)

  def update_mask(self, data: Any, timestamp_ms: float, frame_id: int) -> None:
    self._mask = TimedFrame(data=data, timestamp_ms=timestamp_ms, frame_id=frame_id)

  def _delta(self, a: TimedFrame, b: TimedFrame) -> float:
    return abs(a.timestamp_ms - b.timestamp_ms)

  def try_get_synced_sample(self) -> Optional[Dict[str, Any]]:
    if self._rgb is None or self._depth is None or self._mask is None:
      return None

    rgb_depth_dt = self._delta(self._rgb, self._depth)
    rgb_mask_dt = self._delta(self._rgb, self._mask)
    depth_mask_dt = self._delta(self._depth, self._mask)
    max_dt = max(rgb_depth_dt, rgb_mask_dt, depth_mask_dt)
    if max_dt > self.tolerance_ms:
      return None

    mean_ts = (self._rgb.timestamp_ms + self._depth.timestamp_ms + self._mask.timestamp_ms) / 3.0
    return {
      "rgb": self._rgb.data,
      "depth": self._depth.data,
      "mask": self._mask.data,
      "timestamp_ms": mean_ts,
      "rgb_frame_id": self._rgb.frame_id,
      "depth_frame_id": self._depth.frame_id,
      "mask_frame_id": self._mask.frame_id,
      "sync_error_ms": max_dt,
    }
