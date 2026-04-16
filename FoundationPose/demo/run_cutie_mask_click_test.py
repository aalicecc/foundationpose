import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from Utils import set_logging_format, set_seed
from demo.fpp.cutie_tracker import CutieTracker
from demo.io.realsense_stream import RealSenseStream, ReplayStream
from demo.seg.fastsam_bridge import build_fastsam_bridge

try:
  import yaml
except Exception:
  yaml = None


@dataclass
class ClickPromptState:
  points: list = field(default_factory=list)
  labels: list = field(default_factory=list)
  request_reinit: bool = False

  def clear(self) -> None:
    self.points.clear()
    self.labels.clear()

  def add_positive(self, x: int, y: int) -> None:
    self.points.append((int(x), int(y)))
    self.labels.append(1)

  def add_negative(self, x: int, y: int) -> None:
    self.points.append((int(x), int(y)))
    self.labels.append(0)


def load_config(config_path: str) -> Dict:
  with open(config_path, "r", encoding="utf-8") as f:
    text = f.read()
  if yaml is None:
    raise RuntimeError("PyYAML is required to read demo config file.")
  cfg = yaml.safe_load(text)
  return cfg or {}


def _resolve_from_config_dir(path_like: str, config_dir: Path) -> str:
  if not path_like:
    return path_like
  p = Path(path_like).expanduser()
  if p.is_absolute():
    return str(p.resolve())
  return str((config_dir / p).resolve())


def normalize_config_paths(cfg: Dict, config_path: str) -> Dict:
  config_dir = Path(config_path).resolve().parent

  fastsam_cfg = cfg.get("fastsam", {})
  if isinstance(fastsam_cfg, dict) and fastsam_cfg.get("model_path"):
    fastsam_cfg["model_path"] = _resolve_from_config_dir(fastsam_cfg["model_path"], config_dir)

  offline_cfg = cfg.get("offline", {})
  if isinstance(offline_cfg, dict) and offline_cfg.get("replay_dir"):
    offline_cfg["replay_dir"] = _resolve_from_config_dir(offline_cfg["replay_dir"], config_dir)

  return cfg


def parse_args():
  here = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
    description="Click FastSAM first-frame mask, then test Cutie mask tracking."
  )
  parser.add_argument("--config", type=str, default=str(here / "configs" / "demo.yaml"))
  parser.add_argument("--mode", type=str, default="online", choices=["online", "offline"])
  parser.add_argument("--replay_dir", type=str, default="")
  parser.add_argument("--fastsam_model_path", type=str, default="")
  parser.add_argument("--device", type=str, default="auto")
  parser.add_argument("--no_gui", action="store_true")
  return parser.parse_args()


def create_click_callback(state: ClickPromptState):
  def _mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      state.add_positive(x, y)
      state.request_reinit = True
    elif event == cv2.EVENT_RBUTTONDOWN:
      state.add_negative(x, y)
      state.request_reinit = True
  return _mouse


def draw_click_points(rgb: np.ndarray, click: ClickPromptState) -> np.ndarray:
  vis = rgb.copy()
  for (x, y), label in zip(click.points, click.labels):
    color = (0, 255, 0) if label == 1 else (255, 64, 64)
    cv2.circle(vis, (int(x), int(y)), 5, color, -1)
    cv2.circle(vis, (int(x), int(y)), 7, (255, 255, 255), 1)
  return vis


def normalize_prompt_points(points, image_shape_hw):
  h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
  normalized = []
  corrected = False
  for x, y in points:
    xx = int(x)
    yy = int(y)
    mapped_x = xx % max(1, w)
    mapped_y = int(np.clip(yy, 0, max(0, h - 1)))
    if (mapped_x != xx) or (mapped_y != yy):
      corrected = True
    normalized.append((mapped_x, mapped_y))
  return normalized, corrected


def overlay_mask(rgb: np.ndarray, mask_u8: Optional[np.ndarray], color_rgb, alpha: float = 0.4) -> np.ndarray:
  vis = rgb.copy()
  if mask_u8 is None:
    return vis
  mask_bool = mask_u8 > 0
  overlay = np.zeros_like(vis)
  overlay[..., 0] = int(color_rgb[0])
  overlay[..., 1] = int(color_rgb[1])
  overlay[..., 2] = int(color_rgb[2])
  vis[mask_bool] = ((1.0 - alpha) * vis[mask_bool] + alpha * overlay[mask_bool]).astype(np.uint8)
  return vis


def resize_to_target(rgb: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
  target_w, target_h = target_size
  h, w = rgb.shape[:2]
  if (w, h) == (target_w, target_h):
    return rgb
  return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def add_panel_title(panel_rgb: np.ndarray, title: str) -> np.ndarray:
  out = panel_rgb.copy()
  cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
  cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  return out


def main():
  args = parse_args()
  cfg = normalize_config_paths(load_config(args.config), args.config)

  set_logging_format()
  set_seed(0)
  logging.info("Cutie mask test started, mode=%s", args.mode)

  fastsam_bridge = build_fastsam_bridge(cfg, args)
  cutie_cfg = cfg.get("cutie", {})
  cutie_tracker = None
  click = ClickPromptState()
  fastsam_init_mask_u8: Optional[np.ndarray] = None
  cutie_mask_u8: Optional[np.ndarray] = None
  fastsam_ms_last = 0.0
  cutie_ms_last = 0.0
  frame_idx = 0

  camera_cfg = cfg.get("camera", {})
  target_w = int(camera_cfg.get("target_width", camera_cfg.get("width", 640)))
  target_h = int(camera_cfg.get("target_height", camera_cfg.get("height", 480)))

  if not args.no_gui:
    cv2.namedWindow("cutie_mask_test", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("cutie_mask_test", create_click_callback(click))

  replay_stream = None
  rs_stream = None
  frame_iter = None
  if args.mode == "offline":
    replay_dir = args.replay_dir or cfg.get("offline", {}).get("replay_dir", "")
    if not replay_dir:
      raise ValueError("Offline mode requires --replay_dir or offline.replay_dir in config.")
    replay_stream = ReplayStream(replay_dir)
    frame_iter = iter(replay_stream)
  else:
    rs_stream = RealSenseStream(
      width=int(camera_cfg.get("width", 640)),
      height=int(camera_cfg.get("height", 480)),
      fps=int(camera_cfg.get("fps", 30)),
      depth_scale_override=camera_cfg.get("depth_scale_override"),
      async_queue=bool(cfg.get("performance", {}).get("async_camera", False)),
      first_frame_timeout_ms=int(camera_cfg.get("async_first_frame_timeout_ms", 8000)),
      frame_timeout_ms=int(camera_cfg.get("async_frame_timeout_ms", 10000)),
      wait_for_frames_retries=int(camera_cfg.get("wait_for_frames_retries", 3)),
      grab_failures_before_abort=int(camera_cfg.get("grab_failures_before_abort", 200)),
    )
    rs_stream.start()

  try:
    while True:
      if args.mode == "offline":
        try:
          frame = next(frame_iter)
        except StopIteration:
          break
      else:
        frame = rs_stream.wait_for_frame(timeout_ms=int(camera_cfg.get("main_loop_frame_timeout_ms", 1000)))

      rgb = resize_to_target(frame.rgb, (target_w, target_h))
      frame_idx += 1

      prompt_points_xy, points_corrected = normalize_prompt_points(click.points, rgb.shape[:2])
      if points_corrected and len(prompt_points_xy) > 0:
        click.points = prompt_points_xy

      if click.request_reinit and len(click.points) > 0:
        t0 = time.perf_counter()
        fastsam_init_mask_u8 = fastsam_bridge.infer_point_prompt(
          rgb=rgb,
          points_xy=prompt_points_xy,
          point_labels=click.labels,
        )
        fastsam_ms_last = (time.perf_counter() - t0) * 1000.0
        click.request_reinit = False
        init_fg_pixels = int((fastsam_init_mask_u8 > 0).sum())
        if init_fg_pixels <= 0:
          cutie_mask_u8 = None
          if cutie_tracker is not None:
            del cutie_tracker
            torch.cuda.empty_cache()
            cutie_tracker = None
          logging.warning("FastSAM init mask is empty, skip Cutie initialize. Please click on object again.")
          continue

        if cutie_tracker is not None:
          del cutie_tracker
          torch.cuda.empty_cache()
          cutie_tracker = None
        cutie_tracker = CutieTracker(
          cutie_seg_threshold=float(cutie_cfg.get("seg_threshold", 0.1)),
          erosion_size=int(cutie_cfg.get("erosion_size", 5)),
          max_internal_size=int(cutie_cfg.get("max_internal_size", 480)),
        )
        init_mask_binary = (fastsam_init_mask_u8 > 0).astype(np.uint8)
        t1 = time.perf_counter()
        cutie_tracker.initialize(rgb, {"mask": init_mask_binary})
        cutie_ms_last = (time.perf_counter() - t1) * 1000.0
        cutie_mask_u8 = cutie_tracker.last_mask_u8.copy() if cutie_tracker.last_mask_u8 is not None else None
        logging.info(
          "Re-init done: points=%d fastsam_ms=%.1f cutie_init_ms=%.1f",
          len(click.points),
          fastsam_ms_last,
          cutie_ms_last,
        )

      elif cutie_tracker is not None:
        t1 = time.perf_counter()
        cutie_tracker.track(rgb)
        cutie_ms_last = (time.perf_counter() - t1) * 1000.0
        cutie_mask_u8 = cutie_tracker.last_mask_u8.copy() if cutie_tracker.last_mask_u8 is not None else None

      base_with_points = draw_click_points(rgb, click)
      left_panel = overlay_mask(base_with_points, fastsam_init_mask_u8, color_rgb=(255, 210, 0), alpha=0.42)
      right_panel = overlay_mask(base_with_points, cutie_mask_u8, color_rgb=(0, 255, 0), alpha=0.40)
      left_panel = add_panel_title(left_panel, "FastSAM init mask (click L+:R-)")
      right_panel = add_panel_title(right_panel, "Cutie tracking mask")
      vis_rgb = np.concatenate([left_panel, right_panel], axis=1)
      vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
      cv2.putText(
        vis_bgr,
        f"frame={frame_idx} prompts={len(click.points)} fastsam_ms={fastsam_ms_last:.1f} cutie_ms={cutie_ms_last:.1f}",
        (10, vis_bgr.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
      )

      if not args.no_gui:
        cv2.imshow("cutie_mask_test", vis_bgr)
        key = cv2.waitKey(1) & 0xFF
      else:
        key = 0

      if key == ord("q"):
        break
      if key == ord("c"):
        click.clear()
        click.request_reinit = False
        fastsam_init_mask_u8 = None
        cutie_mask_u8 = None
        if cutie_tracker is not None:
          del cutie_tracker
          torch.cuda.empty_cache()
          cutie_tracker = None
        logging.info("State cleared by user.")
  finally:
    if rs_stream is not None:
      rs_stream.stop()
    if not args.no_gui:
      cv2.destroyAllWindows()
    if cutie_tracker is not None:
      del cutie_tracker
      torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
