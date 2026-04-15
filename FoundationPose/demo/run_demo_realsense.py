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
    
from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
from demo.fpp.kalman_filter_6d import KalmanFilter6D
from demo.io.frame_sync import TimestampSyncer
from demo.io.realsense_stream import RealSenseStream, ReplayRecorder, ReplayStream
from demo.seg.fastsam_bridge import build_fastsam_bridge
from demo.tracking.foundationpose_runner import FoundationPoseRunner

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
  if yaml is not None:
    cfg = yaml.safe_load(text)
  else:
    raise RuntimeError("PyYAML is required to read demo config file.")
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

  mesh_cfg = cfg.get("mesh", {})
  if isinstance(mesh_cfg, dict) and mesh_cfg.get("file"):
    mesh_cfg["file"] = _resolve_from_config_dir(mesh_cfg["file"], config_dir)

  fastsam_cfg = cfg.get("fastsam", {})
  if isinstance(fastsam_cfg, dict) and fastsam_cfg.get("model_path"):
    fastsam_cfg["model_path"] = _resolve_from_config_dir(fastsam_cfg["model_path"], config_dir)

  debug_cfg = cfg.get("debug", {})
  if isinstance(debug_cfg, dict) and debug_cfg.get("dir"):
    debug_cfg["dir"] = _resolve_from_config_dir(debug_cfg["dir"], config_dir)

  offline_cfg = cfg.get("offline", {})
  if isinstance(offline_cfg, dict):
    if offline_cfg.get("replay_dir"):
      offline_cfg["replay_dir"] = _resolve_from_config_dir(offline_cfg["replay_dir"], config_dir)
    if offline_cfg.get("record_dir"):
      offline_cfg["record_dir"] = _resolve_from_config_dir(offline_cfg["record_dir"], config_dir)

  return cfg


def parse_args():
  here = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description="FoundationPose demo with D435i + FastSAM.")
  parser.add_argument("--config", type=str, default=str(here / "configs" / "demo.yaml"))
  parser.add_argument("--mode", type=str, default="online", choices=["online", "offline"])
  parser.add_argument("--replay_dir", type=str, default="")
  parser.add_argument("--record_dir", type=str, default="")
  parser.add_argument("--mesh_file", type=str, default="")
  parser.add_argument("--fastsam_model_path", type=str, default="")
  parser.add_argument("--device", type=str, default="auto")
  parser.add_argument("--no_gui", action="store_true", help="Do not open OpenCV window (throughput profiling).")
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


def overlay_mask(rgb: np.ndarray, mask_u8: Optional[np.ndarray], alpha: float = 0.4) -> np.ndarray:
  vis = rgb.copy()
  if mask_u8 is None:
    return vis
  overlay = np.zeros_like(vis)
  overlay[..., 1] = 255
  fg = (mask_u8 > 0)[..., None]
  vis = np.where(fg, (vis * (1 - alpha) + overlay * alpha).astype(np.uint8), vis)
  return vis


def resize_to_target(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, target_size: Tuple[int, int]):
  target_w, target_h = target_size
  H, W = rgb.shape[:2]
  if (W, H) == (target_w, target_h):
    return rgb, depth, K
  sx = float(target_w) / float(W)
  sy = float(target_h) / float(H)
  rgb_r = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
  depth_r = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
  K_r = K.copy().astype(np.float32)
  K_r[0, 0] *= sx
  K_r[1, 1] *= sy
  K_r[0, 2] *= sx
  K_r[1, 2] *= sy
  return rgb_r, depth_r, K_r


def ensure_mask_matches_depth(mask_u8: Optional[np.ndarray], depth: np.ndarray) -> Optional[np.ndarray]:
  if mask_u8 is None:
    return None
  target_h, target_w = depth.shape[:2]
  if mask_u8.shape[:2] == (target_h, target_w):
    return mask_u8
  logging.warning(
    "Mask/depth shape mismatch, resizing mask from %s to %s",
    mask_u8.shape[:2],
    (target_h, target_w),
  )
  return cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def ensure_rgb_depth_match(rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if rgb.shape[:2] == depth.shape[:2]:
    return rgb, depth
  target_h, target_w = rgb.shape[:2]
  logging.warning(
    "RGB/depth shape mismatch, resizing depth from %s to %s",
    depth.shape[:2],
    (target_h, target_w),
  )
  depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
  return rgb, depth


def build_demo_components(cfg: Dict, args) -> Dict:
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
  )
  fastsam_bridge = build_fastsam_bridge(cfg, args)
  syncer = TimestampSyncer(tolerance_ms=cfg["sync"]["tolerance_ms"])
  return {"tracker": tracker, "fastsam": fastsam_bridge, "syncer": syncer}


def main():
  args = parse_args()
  cfg = load_config(args.config)
  cfg = normalize_config_paths(cfg, args.config)

  set_logging_format()
  set_seed(0)
  os.makedirs(cfg["debug"]["dir"], exist_ok=True)
  logging.info("Demo started, mode=%s", args.mode)

  components = build_demo_components(cfg, args)
  tracker: FoundationPoseRunner = components["tracker"]
  fastsam_bridge = components["fastsam"]
  syncer: TimestampSyncer = components["syncer"]
  fpp_cfg = cfg.get("fpp", {})
  cutie_enabled = bool(fpp_cfg.get("cutie_enabled", False))
  kalman_6d_enabled = bool(fpp_cfg.get("kalman_6d_enabled", False)) and cutie_enabled
  draw_cutie_bbox = bool(fpp_cfg.get("draw_cutie_bbox", False))
  clear_mask_on_pose_reinit = bool(fpp_cfg.get("clear_mask_on_pose_reinit", False))

  click = ClickPromptState()
  init_mask_u8: Optional[np.ndarray] = None
  cutie_tracker = None
  kf_6d: Optional[KalmanFilter6D] = None
  kf_mean: Optional[np.ndarray] = None
  kf_cov: Optional[np.ndarray] = None
  last_cutie_bbox = None
  last_sync_error_ms = -1.0
  last_state = "waiting_init"
  last_reason = "startup"
  pred_hz_ema = 0.0
  pred_hz_inst = 0.0
  pred_count_window = 0
  pred_window_start = time.perf_counter()
  pred_log_interval_s = float(cfg.get("debug", {}).get("hz_log_interval_s", 2.0))
  fastsam_ms_last = 0.0
  box_hz_ema = 0.0
  box_hz_inst = 0.0
  box_count_window = 0
  box_window_start = time.perf_counter()
  box_last_ts = None
  last_vis_bgr: Optional[np.ndarray] = None

  if not args.no_gui:
    cv2.namedWindow("foundationpose_demo", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("foundationpose_demo", create_click_callback(click))

  replay_stream = None
  rs_stream = None
  recorder = None
  main_loop_frame_timeout_ms = 1000
  recover_after_consecutive_timeouts = 3
  camera_restart_cooldown_s = 0.3
  consecutive_read_timeouts = 0
  if args.mode == "offline":
    replay_dir = args.replay_dir or cfg["offline"]["replay_dir"]
    replay_stream = ReplayStream(replay_dir)
    frame_iter = iter(replay_stream)
  else:
    cam = cfg.get("camera", {})
    main_loop_frame_timeout_ms = int(cam.get("main_loop_frame_timeout_ms", 1000))
    recover_after_consecutive_timeouts = max(1, int(cam.get("recover_after_consecutive_timeouts", 3)))
    camera_restart_cooldown_s = float(cam.get("camera_restart_cooldown_s", 0.3))
    rs_stream = RealSenseStream(
      width=cam["width"],
      height=cam["height"],
      fps=cam["fps"],
      depth_scale_override=cam.get("depth_scale_override"),
      async_queue=bool(cfg.get("performance", {}).get("async_camera", False)),
      first_frame_timeout_ms=int(cam.get("async_first_frame_timeout_ms", 8000)),
      frame_timeout_ms=int(cam.get("async_frame_timeout_ms", 10000)),
      wait_for_frames_retries=int(cam.get("wait_for_frames_retries", 3)),
      grab_failures_before_abort=int(cam.get("grab_failures_before_abort", 200)),
    )
    rs_stream.start()
    frame_iter = None
    record_dir = args.record_dir or cfg["offline"]["record_dir"]
    if record_dir:
      recorder = ReplayRecorder(record_dir)
      logging.info("Recording enabled: %s", record_dir)

  try:
    while True:
      should_quit = False
      if args.mode == "offline":
        try:
          frame = next(frame_iter)
        except StopIteration:
          break
      else:
        try:
          frame = rs_stream.wait_for_frame(timeout_ms=main_loop_frame_timeout_ms)
          consecutive_read_timeouts = 0
        except RuntimeError as e:
          msg = str(e).lower()
          is_timeout = ("didn't arrive" in msg) or ("queue empty" in msg)
          if not is_timeout:
            raise
          consecutive_read_timeouts += 1
          if consecutive_read_timeouts == 1 or consecutive_read_timeouts % 10 == 0:
            logging.warning(
              "Camera read timeout (%d consecutive): %s",
              consecutive_read_timeouts,
              e,
            )
          if consecutive_read_timeouts >= recover_after_consecutive_timeouts:
            logging.warning(
              "Camera timed out %d times; restarting RealSense pipeline...",
              consecutive_read_timeouts,
            )
            try:
              rs_stream.stop()
              time.sleep(max(0.0, camera_restart_cooldown_s))
              rs_stream.start()
              consecutive_read_timeouts = 0
              logging.info("RealSense pipeline restarted.")
            except Exception as restart_err:
              logging.error("RealSense pipeline restart failed: %s", restart_err)
          if not args.no_gui:
            if last_vis_bgr is not None:
              timeout_vis = last_vis_bgr.copy()
            else:
              cam_h = int(cfg["camera"]["target_height"])
              cam_w = int(cfg["camera"]["target_width"])
              timeout_vis = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            cv2.putText(
              timeout_vis,
              f"Camera timeout x{consecutive_read_timeouts}, recovering...",
              (8, 24),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.6,
              (0, 0, 255),
              2,
            )
            cv2.imshow("foundationpose_demo", timeout_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
              should_quit = True
          if should_quit:
            break
          continue
      rgb, depth, K = resize_to_target(
        frame.rgb, frame.depth, frame.K, (cfg["camera"]["target_width"], cfg["camera"]["target_height"])
      )
      rgb, depth = ensure_rgb_depth_match(rgb, depth)
      if recorder is not None:
        recorder.push(frame)

      syncer.update_rgb(rgb, frame.timestamp_ms, frame.frame_id)
      syncer.update_depth(depth, frame.timestamp_ms, frame.frame_id)

      if click.request_reinit and len(click.points) > 0:
        t_fastsam_start = time.perf_counter()
        init_mask_u8 = fastsam_bridge.infer_point_prompt(rgb=rgb, points_xy=click.points, point_labels=click.labels)
        fastsam_ms_last = (time.perf_counter() - t_fastsam_start) * 1000.0
        logging.info("fastsam one-shot time_ms=%.1f", fastsam_ms_last)
        init_mask_u8 = ensure_mask_matches_depth(init_mask_u8, depth)
        syncer.update_mask(init_mask_u8, frame.timestamp_ms, frame.frame_id)
        click.request_reinit = False
        click.clear()
        tracker.initialized = False
        tracker.last_pose = None
        if cutie_tracker is not None:
          del cutie_tracker
          torch.cuda.empty_cache()
        cutie_tracker = None
        kf_6d = None
        kf_mean = None
        kf_cov = None
        last_cutie_bbox = None
      elif init_mask_u8 is not None:
        init_mask_u8 = ensure_mask_matches_depth(init_mask_u8, depth)
        syncer.update_mask(init_mask_u8, frame.timestamp_ms, frame.frame_id)

      sample = syncer.try_get_synced_sample()
      pose = None
      if not tracker.initialized:
        if sample is not None:
          sample_mask = ensure_mask_matches_depth(sample["mask"], sample["depth"])
          t_pred_start = time.perf_counter()
          result = tracker.register(
            K=K,
            rgb=sample["rgb"],
            depth=sample["depth"],
            mask_u8=sample_mask,
            timestamp_ms=sample["timestamp_ms"],
          )
          pred_dt = time.perf_counter() - t_pred_start
          # 忽略等待初始化阶段的快速返回（例如 mask 太小），避免 Hz 虚高
          should_count_pred = (result.state == "tracking") and (pred_dt >= 0.002)
          if should_count_pred:
            pred_hz_inst = 1.0 / pred_dt
            pred_hz_ema = pred_hz_inst if pred_hz_ema <= 0 else (0.9 * pred_hz_ema + 0.1 * pred_hz_inst)
            pred_count_window += 1
          pose = result.pose
          if result.state == "tracking" and cutie_enabled:
            from demo.fpp.cutie_tracker import CutieTracker

            ccfg = cfg.get("cutie", {})
            sm = (sample_mask > 0).astype(np.uint8)
            sm[sm > 0] = 1
            if cutie_tracker is None:
              cutie_tracker = CutieTracker(
                cutie_seg_threshold=float(ccfg.get("seg_threshold", 0.1)),
                erosion_size=int(ccfg.get("erosion_size", 5)),
                max_internal_size=int(ccfg.get("max_internal_size", 480)),
              )
            cutie_tracker.initialize(sample["rgb"], {"mask": sm})
            if kalman_6d_enabled:
              kf_6d = KalmanFilter6D(float(fpp_cfg.get("kf_measurement_noise_scale", 1.0)))
              kf_mean, kf_cov = FoundationPoseRunner.init_fpp_kalman(result.pose, kf_6d)
          last_state = result.state
          last_reason = result.reason
          last_sync_error_ms = sample["sync_error_ms"]
      else:
        last_cutie_bbox = None
        if cutie_enabled and cutie_tracker is not None:
          last_cutie_bbox = cutie_tracker.track(rgb)
          kf_mean, kf_cov = tracker.apply_fpp_prior(
            K,
            last_cutie_bbox,
            kf_6d if kalman_6d_enabled else None,
            kf_mean,
            kf_cov,
            use_kalman=kalman_6d_enabled,
          )
        t_pred_start = time.perf_counter()
        result = tracker.track(K=K, rgb=rgb, depth=depth, timestamp_ms=frame.timestamp_ms)
        pred_dt = time.perf_counter() - t_pred_start
        if pred_dt >= 0.002:
          pred_hz_inst = 1.0 / pred_dt
          pred_hz_ema = pred_hz_inst if pred_hz_ema <= 0 else (0.9 * pred_hz_ema + 0.1 * pred_hz_inst)
          pred_count_window += 1
        pose = result.pose
        if kalman_6d_enabled and kf_6d is not None and kf_mean is not None and result.state == "tracking":
          kf_mean, kf_cov = FoundationPoseRunner.fpp_kalman_predict(kf_6d, kf_mean, kf_cov)
        last_state = result.state
        last_reason = result.reason
        if result.state == "reinit_required" and clear_mask_on_pose_reinit:
          init_mask_u8 = None

      now_perf = time.perf_counter()
      if (now_perf - pred_window_start) >= pred_log_interval_s:
        window_s = max(1e-6, now_perf - pred_window_start)
        pred_hz_window = pred_count_window / window_s
        logging.info(
          "pred_hz window=%.2f ema=%.2f inst=%.2f n=%d",
          pred_hz_window,
          pred_hz_ema,
          pred_hz_inst,
          pred_count_window,
        )
        pred_count_window = 0
        pred_window_start = now_perf

      vis_rgb = overlay_mask(rgb, init_mask_u8 if not tracker.initialized else None)
      if (
        draw_cutie_bbox
        and last_cutie_bbox is not None
        and len(last_cutie_bbox) >= 4
        and last_cutie_bbox[0] >= 0
      ):
        x, y, w, h = (int(last_cutie_bbox[0]), int(last_cutie_bbox[1]), int(last_cutie_bbox[2]), int(last_cutie_bbox[3]))
        cv2.rectangle(vis_rgb, (x, y), (x + w, y + h), (255, 64, 64), 2)
      should_draw_box = (pose is not None) and tracker.initialized
      if should_draw_box:
        now_box_ts = time.perf_counter()
        if box_last_ts is not None:
          dt_box = max(1e-6, now_box_ts - box_last_ts)
          box_hz_inst = 1.0 / dt_box
          box_hz_ema = box_hz_inst if box_hz_ema <= 0 else (0.9 * box_hz_ema + 0.1 * box_hz_inst)
        box_last_ts = now_box_ts
        box_count_window += 1
        center_pose = pose @ np.linalg.inv(tracker.to_origin)
        vis_rgb = draw_posed_3d_box(K=K, img=vis_rgb, ob_in_cam=center_pose, bbox=tracker.bbox)
        vis_rgb = draw_xyz_axis(
          color=vis_rgb,
          ob_in_cam=center_pose,
          scale=cfg["visualization"]["axis_scale_m"],
          K=K,
          thickness=cfg["visualization"]["axis_thickness"],
          transparency=0,
          is_input_rgb=True,
        )
      if (now_perf - box_window_start) >= pred_log_interval_s:
        box_window_s = max(1e-6, now_perf - box_window_start)
        box_hz_window = box_count_window / box_window_s
        logging.info("box_hz window=%.2f ema=%.2f inst=%.2f n=%d", box_hz_window, box_hz_ema, box_hz_inst, box_count_window)
        box_count_window = 0
        box_window_start = now_perf
      vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
      cv2.putText(
        vis_bgr,
        f"box_hz_ema={box_hz_ema:.2f} box_hz={box_hz_inst:.2f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
      )
      last_vis_bgr = vis_bgr.copy()
      if not args.no_gui:
        cv2.imshow("foundationpose_demo", vis_bgr)
        key = cv2.waitKey(1) & 0xFF
      else:
        key = 0
      if key == ord("q"):
        break
      if key == ord("c"):
        click.clear()
        init_mask_u8 = None
        tracker.initialized = False
        tracker.last_pose = None
        if cutie_tracker is not None:
          del cutie_tracker
          torch.cuda.empty_cache()
        cutie_tracker = None
        kf_6d = None
        kf_mean = None
        kf_cov = None
        last_cutie_bbox = None
        last_state = "waiting_init"
        last_reason = "manual_clear"
  finally:
    if rs_stream is not None:
      rs_stream.stop()
    if recorder is not None:
      info = recorder.close()
      logging.info("Record closed: %s", info)
    if not args.no_gui:
      cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
