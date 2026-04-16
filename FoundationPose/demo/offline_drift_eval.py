#!/usr/bin/env python3
"""Replay recorded RGB-D (ReplayRecorder layout) and log drift metrics to CSV for tuning.

Expects: replay_dir/{rgb,depth}/*.png and meta.npz (see ReplayStream).

Uses a fixed center ROI mask for all frames (no Cutie). Suitable for repeatable
metrics; for production-like mask quality, run the full demo online with Cutie.

Usage:
  python3 demo/offline_drift_eval.py --config demo/configs/demo.yaml --replay_dir /path/to/replay --output_csv drift_log.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from demo.io.realsense_stream import ReplayStream
from demo.run_demo_realsense import ensure_rgb_depth_match, normalize_config_paths, resize_to_target
from demo.tracking.foundationpose_runner import FoundationPoseRunner
from Utils import set_logging_format, set_seed

try:
  import yaml
except Exception:
  yaml = None


def load_config(config_path: str) -> Dict:
  with open(config_path, "r", encoding="utf-8") as f:
    text = f.read()
  if yaml is None:
    raise RuntimeError("PyYAML is required.")
  return yaml.safe_load(text) or {}


def make_center_mask_hw(h: int, w: int, frac_h: float = 0.55, frac_w: float = 0.55) -> np.ndarray:
  mask = np.zeros((h, w), dtype=np.uint8)
  fh = int(h * frac_h)
  fw = int(w * frac_w)
  y0 = max(0, (h - fh) // 2)
  x0 = max(0, (w - fw) // 2)
  mask[y0 : y0 + fh, x0 : x0 + fw] = 255
  return mask


def main():
  here = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description="Offline drift metrics CSV from replay.")
  parser.add_argument("--config", type=str, default=str(here / "configs" / "demo.yaml"))
  parser.add_argument("--replay_dir", type=str, required=True)
  parser.add_argument("--output_csv", type=str, required=True)
  parser.add_argument("--mesh_file", type=str, default="")
  parser.add_argument("--mask_frac_h", type=float, default=0.55)
  parser.add_argument("--mask_frac_w", type=float, default=0.55)
  args = parser.parse_args()

  cfg = normalize_config_paths(load_config(args.config), args.config)
  set_logging_format()
  set_seed(0)

  mesh_file = args.mesh_file or cfg["mesh"]["file"]
  perf = cfg.get("performance", {})
  tracker = FoundationPoseRunner(
    mesh_file=mesh_file,
    est_refine_iter=cfg["tracker"]["est_refine_iter"],
    track_refine_iter=cfg["tracker"]["track_refine_iter"],
    rot_grid_min_views=cfg["tracker"].get("rot_grid_min_views", 20),
    rot_grid_inplane_step=cfg["tracker"].get("rot_grid_inplane_step", 90),
    score_max_candidates=cfg["tracker"].get("score_max_candidates", 64),
    render_bs_cap=cfg["tracker"].get("render_bs_cap", 32),
    score_render_bs_cap=cfg["tracker"].get("score_render_bs_cap", 8),
    score_forward_bs_cap=cfg["tracker"].get("score_forward_bs_cap", 16),
    max_translation_jump_m=cfg["tracker"]["max_translation_jump_m"],
    max_rotation_jump_deg=cfg["tracker"]["max_rotation_jump_deg"],
    min_mask_pixels=cfg["tracker"]["min_mask_pixels"],
    debug=cfg["debug"]["level"],
    debug_dir=cfg["debug"]["dir"],
    fast_depth_filter=bool(perf.get("fast_depth_filter", False)),
    drift=cfg.get("drift"),
  )
  stream = ReplayStream(args.replay_dir)

  rows = []
  fields = [
    "frame_id",
    "timestamp_ms",
    "state",
    "reason",
    "quality_state",
    "jump_xy_m",
    "jump_z_m",
    "rot_deg",
    "rpy_rate_deg_s",
    "depth_median_m",
    "depth_valid_ratio",
    "z_depth_residual_m",
    "drift_reason",
    "consecutive_depth_bad_frames",
  ]

  mask_u8: Optional[np.ndarray] = None
  for frame in stream:
    rgb, depth, K = resize_to_target(
      frame.rgb,
      frame.depth,
      frame.K,
      (cfg["camera"]["target_width"], cfg["camera"]["target_height"]),
    )
    rgb, depth = ensure_rgb_depth_match(rgb, depth)
    h, w = rgb.shape[:2]
    if mask_u8 is None:
      mask_u8 = make_center_mask_hw(h, w, args.mask_frac_h, args.mask_frac_w)

    if not tracker.initialized:
      res = tracker.register(
        K=K,
        rgb=rgb,
        depth=depth,
        mask_u8=mask_u8,
        timestamp_ms=frame.timestamp_ms,
      )
    else:
      res = tracker.track(
        K=K,
        rgb=rgb,
        depth=depth,
        timestamp_ms=frame.timestamp_ms,
        mask_u8=mask_u8,
      )

    row = {
      "frame_id": frame.frame_id,
      "timestamp_ms": frame.timestamp_ms,
      "state": res.state,
      "reason": res.reason,
      "quality_state": res.quality_state,
      "jump_xy_m": f"{res.jump_xy_m:.6f}",
      "jump_z_m": f"{res.jump_z_m:.6f}",
      "rot_deg": f"{res.rot_deg:.6f}",
      "rpy_rate_deg_s": f"{res.rpy_rate_deg_s:.6f}",
      "depth_median_m": f"{res.depth_median_m:.6f}",
      "depth_valid_ratio": f"{res.depth_valid_ratio:.6f}",
      "z_depth_residual_m": f"{res.z_depth_residual_m:.6f}",
      "drift_reason": res.drift_reason,
      "consecutive_depth_bad_frames": res.consecutive_depth_bad_frames,
    }
    rows.append(row)
    if res.state == "reinit_required":
      logging.warning("reinit_required at frame %s: %s", frame.frame_id, res.reason)

  out_path = Path(args.output_csv).expanduser()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
      w.writerow(r)
  logging.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
  main()
