import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import cv2
import numpy as np


try:
  import pyrealsense2 as rs
except Exception:
  rs = None


@dataclass
class RGBDFrame:
  rgb: np.ndarray
  depth: np.ndarray
  K: np.ndarray
  timestamp_ms: float
  frame_id: int


class RealSenseStream:
  """D435i RGB-D stream with depth aligned to color."""

  def __init__(
    self,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    depth_scale_override: Optional[float] = None,
    async_queue: bool = False,
    first_frame_timeout_ms: int = 8000,
    frame_timeout_ms: int = 10000,
    wait_for_frames_retries: int = 3,
    grab_failures_before_abort: int = 200,
  ):
    if rs is None:
      raise ImportError("pyrealsense2 is not installed. Please install librealsense Python bindings first.")
    self.width = int(width)
    self.height = int(height)
    self.fps = int(fps)
    self.depth_scale_override = depth_scale_override
    self.async_queue = bool(async_queue)
    self.first_frame_timeout_ms = int(first_frame_timeout_ms)
    self.frame_timeout_ms = int(frame_timeout_ms)
    self.wait_for_frames_retries = max(1, int(wait_for_frames_retries))
    self.grab_failures_before_abort = int(grab_failures_before_abort)
    self._frame_q: Optional[queue.Queue] = None
    self._grab_stop: Optional[threading.Event] = None
    self._grab_thread: Optional[threading.Thread] = None
    self._async_delivered_frame: bool = False
    self._K: Optional[np.ndarray] = None
    self.pipeline = rs.pipeline()
    self.config = rs.config()
    self.align = rs.align(rs.stream.color)
    self.profile = None
    self.depth_scale = 0.001

  def _compute_intrinsics_matrix(self) -> np.ndarray:
    if self.profile is None:
      raise RuntimeError("Stream has not started.")
    color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    return np.array(
      [
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
      ],
      dtype=np.float32,
    )

  def start(self) -> None:
    self._async_delivered_frame = False
    self._K = None
    self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
    self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
    self.profile = self.pipeline.start(self.config)
    if self.depth_scale_override is not None:
      self.depth_scale = float(self.depth_scale_override)
    else:
      depth_sensor = self.profile.get_device().first_depth_sensor()
      self.depth_scale = float(depth_sensor.get_depth_scale())
    self._K = self._compute_intrinsics_matrix()
    if self.async_queue:
      self._frame_q = queue.Queue(maxsize=1)
      self._grab_stop = threading.Event()
      self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
      self._grab_thread.start()

  def _grab_loop(self) -> None:
    assert self._frame_q is not None
    consecutive_failures = 0
    while True:
      stop_flag = self._grab_stop
      if stop_flag is None or stop_flag.is_set():
        break
      try:
        frame = self._read_one_frame(timeout_ms=max(500, self.frame_timeout_ms))
        consecutive_failures = 0
      except Exception as e:
        consecutive_failures += 1
        if consecutive_failures == 1 or consecutive_failures % 30 == 0:
          logging.warning(
            "RealSense grab loop: read failed (%d consecutive): %s",
            consecutive_failures,
            e,
          )
        if consecutive_failures >= self.grab_failures_before_abort:
          logging.error(
            "RealSense grab loop exiting after %d consecutive read failures.",
            consecutive_failures,
          )
          break
        continue
      try:
        self._frame_q.put_nowait(frame)
      except queue.Full:
        try:
          self._frame_q.get_nowait()
        except queue.Empty:
          pass
        try:
          self._frame_q.put_nowait(frame)
        except queue.Full:
          pass

  def stop(self) -> None:
    if self._grab_stop is not None:
      self._grab_stop.set()
    if self.profile is not None:
      try:
        self.pipeline.stop()
      except Exception as e:
        logging.warning("RealSense pipeline.stop() failed: %s", e)
    if self._grab_thread is not None:
      self._grab_thread.join()
      self._grab_thread = None
      self._grab_stop = None
      self._frame_q = None
    self.profile = None
    self._K = None

  def get_intrinsics_matrix(self) -> np.ndarray:
    if self._K is not None:
      return self._K.copy()
    return self._compute_intrinsics_matrix()

  def _read_one_frame(self, timeout_ms: Optional[int] = None) -> RGBDFrame:
    if self.profile is None:
      raise RuntimeError("Stream has not started.")
    if self._K is None:
      raise RuntimeError("Intrinsics not ready.")
    t = int(timeout_ms) if timeout_ms is not None else self.frame_timeout_ms
    frames = None
    last_err: Optional[BaseException] = None
    for attempt in range(self.wait_for_frames_retries):
      try:
        frames = self.pipeline.wait_for_frames(timeout_ms=t)
        break
      except RuntimeError as e:
        last_err = e
        msg = str(e).lower()
        transient = "didn't arrive" in msg
        if transient and attempt + 1 < self.wait_for_frames_retries:
          logging.warning(
            "RealSense wait_for_frames failed (attempt %d/%d, %d ms): %s",
            attempt + 1,
            self.wait_for_frames_retries,
            t,
            e,
          )
          continue
        raise
    if frames is None:
      if last_err is not None:
        raise last_err
      raise RuntimeError("wait_for_frames returned no frames")
    aligned_frames = self.align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
      raise RuntimeError("Failed to fetch aligned color/depth frame.")
    rgb = np.asanyarray(color_frame.get_data())[:, :, ::-1].copy()  # BGR -> RGB
    depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth = depth_raw * self.depth_scale  # meters
    timestamp_ms = float(color_frame.get_timestamp())
    frame_id = int(color_frame.get_frame_number())
    return RGBDFrame(
      rgb=rgb,
      depth=depth,
      K=self._K.copy(),
      timestamp_ms=timestamp_ms,
      frame_id=frame_id,
    )

  def wait_for_frame(self, timeout_ms: Optional[int] = None) -> RGBDFrame:
    if not self.async_queue or self._frame_q is None:
      t = self.frame_timeout_ms if timeout_ms is None else int(timeout_ms)
      return self._read_one_frame(timeout_ms=t)

    effective_ms = int(timeout_ms) if timeout_ms is not None else (
      self.first_frame_timeout_ms if not self._async_delivered_frame else self.frame_timeout_ms
    )
    deadline = time.perf_counter() + max(0.001, effective_ms / 1000.0)

    while True:
      remaining = deadline - time.perf_counter()
      if remaining <= 0:
        break
      try:
        frame = self._frame_q.get(timeout=max(1e-3, remaining))
        self._async_delivered_frame = True
        return frame
      except queue.Empty:
        if self._grab_thread is None or not self._grab_thread.is_alive():
          logging.warning("Grab thread stopped; falling back to synchronous frame read.")
          return self._read_one_frame(timeout_ms=self.frame_timeout_ms)
        continue

    logging.warning(
      "Async camera queue empty after %d ms (first_frame=%s); waiting one extra 5s block.",
      effective_ms,
      not self._async_delivered_frame,
    )
    try:
      frame = self._frame_q.get(timeout=5.0)
      self._async_delivered_frame = True
      return frame
    except queue.Empty:
      if self._grab_thread is not None and self._grab_thread.is_alive():
        raise RuntimeError(
          "Async camera queue still empty after extended wait; grab thread is running. "
          "Increase camera.async_first_frame_timeout_ms / async_frame_timeout_ms or set performance.async_camera: false."
        ) from None
      logging.warning("Grab thread stopped; falling back to synchronous frame read.")
      return self._read_one_frame(timeout_ms=self.frame_timeout_ms)


class ReplayStream:
  """Replay frames recorded by this demo recorder."""

  def __init__(self, replay_dir: str):
    self.replay_dir = Path(replay_dir)
    self.rgb_dir = self.replay_dir / "rgb"
    self.depth_dir = self.replay_dir / "depth"
    self.meta_path = self.replay_dir / "meta.npz"
    if not self.meta_path.exists():
      raise FileNotFoundError(f"Missing replay meta file: {self.meta_path}")
    meta = np.load(self.meta_path)
    self.ids = [int(x) for x in meta["frame_ids"]]
    self.timestamps_ms = [float(x) for x in meta["timestamps_ms"]]
    self.K = np.array(meta["K"], dtype=np.float32)
    self._idx = 0

  def __iter__(self) -> Iterator[RGBDFrame]:
    self._idx = 0
    return self

  def __next__(self) -> RGBDFrame:
    if self._idx >= len(self.ids):
      raise StopIteration
    frame_id = self.ids[self._idx]
    ts = self.timestamps_ms[self._idx]
    rgb = cv2.cvtColor(cv2.imread(str(self.rgb_dir / f"{frame_id:06d}.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(self.depth_dir / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    out = RGBDFrame(rgb=rgb, depth=depth, K=self.K.copy(), timestamp_ms=ts, frame_id=frame_id)
    self._idx += 1
    return out


class ReplayRecorder:
  """Record aligned RGB-D frames for offline debugging."""

  def __init__(self, output_dir: str):
    self.output_dir = Path(output_dir)
    self.rgb_dir = self.output_dir / "rgb"
    self.depth_dir = self.output_dir / "depth"
    self.rgb_dir.mkdir(parents=True, exist_ok=True)
    self.depth_dir.mkdir(parents=True, exist_ok=True)
    self.frame_ids = []
    self.timestamps_ms = []
    self.K: Optional[np.ndarray] = None

  def push(self, frame: RGBDFrame) -> None:
    rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(self.rgb_dir / f"{frame.frame_id:06d}.png"), rgb_bgr)
    depth_mm = np.clip(frame.depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(self.depth_dir / f"{frame.frame_id:06d}.png"), depth_mm)
    self.frame_ids.append(int(frame.frame_id))
    self.timestamps_ms.append(float(frame.timestamp_ms))
    self.K = frame.K.copy()

  def close(self) -> Dict[str, str]:
    if self.K is None:
      return {"status": "empty"}
    np.savez(
      self.output_dir / "meta.npz",
      frame_ids=np.asarray(self.frame_ids, dtype=np.int32),
      timestamps_ms=np.asarray(self.timestamps_ms, dtype=np.float64),
      K=self.K.astype(np.float32),
      saved_at=time.time(),
    )
    return {"status": "ok", "dir": str(self.output_dir)}
