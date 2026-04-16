from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

from demo.fpp.kalman_filter_6d import KalmanFilter6D
from demo.fpp.pose_utils import (
  adjust_pose_to_image_point,
  apply_z_blend_and_rotation_slerp,
  compute_depth_median_and_valid_ratio,
  get_6d_pose_arr_from_mat,
  get_mat_from_6d_pose_arr,
  get_pose_xy_from_image_point,
)

@dataclass
class TrackResult:
  pose: np.ndarray
  state: str
  reason: str
  timestamp_ms: float
  quality_state: str = "unknown"
  jump_xy_m: float = 0.0
  jump_z_m: float = 0.0
  rot_deg: float = 0.0
  rpy_rate_deg_s: float = 0.0
  depth_median_m: float = 0.0
  depth_valid_ratio: float = 0.0
  z_depth_residual_m: float = 0.0
  drift_reason: str = ""
  consecutive_depth_bad_frames: int = 0


def _rotation_angle_deg(R_prev: np.ndarray, R_cur: np.ndarray) -> float:
  dR = R_prev.T @ R_cur
  trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(trace)))


def _default_drift_cfg(
  max_translation_jump_m: float,
  max_rotation_jump_deg: float,
  drift_yaml: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
  base = {
    "enabled": True,
    "max_xy_jump_m": float(max_translation_jump_m),
    "max_z_jump_m": float(max_translation_jump_m),
    "max_rot_deg": float(max_rotation_jump_deg),
    "max_rpy_rate_deg_s": 480.0,
    "max_z_depth_residual_m": 0.12,
    "min_depth_valid_ratio": 0.2,
    "bad_frames_for_reinit": 5,
    "suspect_ratio": 0.55,
    "z_median_blend": 0.42,
    "rotation_slerp": 0.5,
    "lock_roll_pitch_on_bad_short": False,
  }
  if drift_yaml and isinstance(drift_yaml, dict):
    for k, v in drift_yaml.items():
      if v is not None:
        base[k] = v
  return base


class FoundationPoseRunner:
  """Stateful wrapper: first-frame register + continuous track + optional reinit."""

  def __init__(
    self,
    mesh_file: str,
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    rot_grid_min_views: int = 20,
    rot_grid_inplane_step: int = 90,
    score_max_candidates: int = 64,
    render_bs_cap: int = 32,
    score_render_bs_cap: int = 8,
    score_forward_bs_cap: int = 16,
    max_translation_jump_m: float = 0.20,
    max_rotation_jump_deg: float = 45.0,
    min_mask_pixels: int = 64,
    debug: int = 0,
    debug_dir: str = "debug",
    fast_depth_filter: bool = False,
    drift: Optional[Dict[str, Any]] = None,
  ):
    mesh_path = Path(mesh_file).expanduser()
    if not mesh_path.is_absolute():
      mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
      raise FileNotFoundError(
        "Mesh file not found.\n"
        f"  given: {mesh_file}\n"
        f"  resolved: {mesh_path}\n"
        "Please fix `demo/configs/demo.yaml` `mesh.file` or pass `--mesh_file`."
      )
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
      meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
      if len(meshes) == 0:
        raise ValueError(f"Loaded scene has no mesh geometry: {mesh_path}")
      self.mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
      self.mesh = loaded
    else:
      raise TypeError(f"Unsupported mesh type from trimesh.load: {type(loaded)}")
    self.est_refine_iter = int(est_refine_iter)
    self.track_refine_iter = int(track_refine_iter)
    self.rot_grid_min_views = int(rot_grid_min_views)
    self.rot_grid_inplane_step = int(rot_grid_inplane_step)
    self.score_max_candidates = int(score_max_candidates)
    self.render_bs_cap = int(render_bs_cap)
    self.score_render_bs_cap = int(score_render_bs_cap)
    self.score_forward_bs_cap = int(score_forward_bs_cap)
    self.max_translation_jump_m = float(max_translation_jump_m)
    self.max_rotation_jump_deg = float(max_rotation_jump_deg)
    self.min_mask_pixels = int(min_mask_pixels)
    self.debug = int(debug)
    self.debug_dir = debug_dir
    self.fast_depth_filter = bool(fast_depth_filter)

    self._drift = _default_drift_cfg(max_translation_jump_m, max_rotation_jump_deg, drift)
    self._R_smooth: Optional[np.ndarray] = None
    self._last_ts_ms: Optional[float] = None
    self._consecutive_depth_bad: int = 0

    to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
    self.to_origin = to_origin
    self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    self.estimator = FoundationPose(
      model_pts=self.mesh.vertices,
      model_normals=self.mesh.vertex_normals,
      mesh=self.mesh,
      scorer=scorer,
      refiner=refiner,
      debug=self.debug,
      debug_dir=self.debug_dir,
      glctx=glctx,
      rot_grid_min_views=self.rot_grid_min_views,
      rot_grid_inplane_step=self.rot_grid_inplane_step,
      score_max_candidates=self.score_max_candidates,
    )
    self.estimator.refiner.cfg["render_bs_cap"] = self.render_bs_cap
    self.estimator.scorer.cfg["score_render_bs_cap"] = self.score_render_bs_cap
    self.estimator.scorer.cfg["score_forward_bs_cap"] = self.score_forward_bs_cap

    self.initialized = False
    self.last_pose: Optional[np.ndarray] = None

  def reset_drift_state(self) -> None:
    self._R_smooth = None
    self._last_ts_ms = None
    self._consecutive_depth_bad = 0

  @staticmethod
  def init_fpp_kalman(pose_uncentered_numpy: np.ndarray, kf: KalmanFilter6D) -> Tuple[np.ndarray, np.ndarray]:
    """First-frame KF state; `pose_uncentered_numpy` matches `FoundationPose.register` return (4x4)."""
    return kf.initiate(get_6d_pose_arr_from_mat(pose_uncentered_numpy))

  def apply_fpp_prior(
    self,
    K_np: np.ndarray,
    bbox_xywh: Optional[list],
    kf: Optional[KalmanFilter6D],
    kf_mean: Optional[np.ndarray],
    kf_cov: Optional[np.ndarray],
    use_kalman: bool,
  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Before `track_one`: inject 2D bbox (+ optional 6D KF) into `estimator.pose_last` (FoundationPose++ order)."""
    if self.estimator.pose_last is None:
      return kf_mean, kf_cov
    if bbox_xywh is None or len(bbox_xywh) < 4 or bbox_xywh[0] < 0 or bbox_xywh[2] <= 0:
      return kf_mean, kf_cov
    cx = float(bbox_xywh[0] + bbox_xywh[2] / 2.0)
    cy = float(bbox_xywh[1] + bbox_xywh[3] / 2.0)
    pl = self.estimator.pose_last
    K = torch.as_tensor(K_np, dtype=pl.dtype, device=pl.device)

    if use_kalman and kf is not None and kf_mean is not None and kf_cov is not None:
      kf_mean, kf_cov = kf.update(kf_mean, kf_cov, get_6d_pose_arr_from_mat(pl))
      tx, ty = get_pose_xy_from_image_point(pl, K, cx, cy)
      measurement_xy = np.array([float(tx), float(ty)], dtype=np.float64)
      kf_mean, kf_cov = kf.update_from_xy(kf_mean, kf_cov, measurement_xy)
      mat = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).to(dtype=pl.dtype, device=pl.device)
      self.estimator.pose_last = mat
    else:
      adjusted = adjust_pose_to_image_point(ob_in_cam=pl, K=K, x=cx, y=cy)
      self.estimator.pose_last = adjusted
    return kf_mean, kf_cov

  @staticmethod
  def fpp_kalman_predict(kf: KalmanFilter6D, kf_mean: np.ndarray, kf_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Call after successful `track_one` when using FPP 6D Kalman (one step behind, same as FPP)."""
    return kf.predict(kf_mean, kf_cov)

  def register(self, K: np.ndarray, rgb: np.ndarray, depth: np.ndarray, mask_u8: np.ndarray, timestamp_ms: float) -> TrackResult:
    mask = (mask_u8 > 0)
    if int(mask.sum()) < self.min_mask_pixels:
      return TrackResult(
        pose=np.eye(4, dtype=np.float32),
        state="waiting_init",
        reason=f"mask_too_small:{int(mask.sum())}",
        timestamp_ms=timestamp_ms,
        quality_state="na",
      )
    pose = self.estimator.register(
      K=K,
      rgb=rgb,
      depth=depth,
      ob_mask=mask,
      iteration=self.est_refine_iter,
    )
    self.initialized = True
    self.last_pose = pose.copy()
    self.reset_drift_state()
    self._R_smooth = pose[:3, :3].copy()
    self._last_ts_ms = float(timestamp_ms)
    return TrackResult(
      pose=pose,
      state="tracking",
      reason="register_ok",
      timestamp_ms=timestamp_ms,
      quality_state="good",
    )

  def _legacy_jump_gate(self, pose: np.ndarray) -> Dict[str, float]:
    if self.last_pose is None:
      return {"jump": 0.0, "rot_deg": 0.0, "ok": 1.0}
    jump = np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3])
    rot_deg = _rotation_angle_deg(self.last_pose[:3, :3], pose[:3, :3])
    ok = float((jump <= self.max_translation_jump_m) and (rot_deg <= self.max_rotation_jump_deg))
    return {"jump": float(jump), "rot_deg": float(rot_deg), "ok": ok}

  def _compute_drift_metrics(
    self,
    pose: np.ndarray,
    depth: np.ndarray,
    mask_u8: Optional[np.ndarray],
    timestamp_ms: float,
  ) -> Dict[str, float]:
    dcfg = self._drift
    dt_s = 1.0 / 30.0
    if self._last_ts_ms is not None:
      dt_s = max(1e-4, (float(timestamp_ms) - float(self._last_ts_ms)) / 1000.0)

    jump_xy_m = 0.0
    jump_z_m = 0.0
    rot_deg = 0.0
    if self.last_pose is not None:
      dt = pose[:3, 3] - self.last_pose[:3, 3]
      jump_xy_m = float(np.hypot(float(dt[0]), float(dt[1])))
      jump_z_m = float(abs(float(dt[2])))
      rot_deg = _rotation_angle_deg(self.last_pose[:3, :3], pose[:3, :3])
    rpy_rate_deg_s = float(rot_deg / dt_s)

    median_z, valid_ratio = compute_depth_median_and_valid_ratio(depth, mask_u8)
    tz = float(pose[2, 3])
    z_res = abs(tz - median_z) if (mask_u8 is not None and np.any(mask_u8 > 0) and median_z > 1e-6) else 0.0

    return {
      "jump_xy_m": jump_xy_m,
      "jump_z_m": jump_z_m,
      "rot_deg": rot_deg,
      "rpy_rate_deg_s": rpy_rate_deg_s,
      "depth_median_m": median_z,
      "depth_valid_ratio": valid_ratio,
      "z_depth_residual_m": z_res,
      "dt_s": dt_s,
    }

  def _classify_drift(
    self,
    m: Dict[str, float],
    mask_u8: Optional[np.ndarray],
  ) -> Tuple[str, str, List[str]]:
    """Returns quality_state, top reason tag, list of triggered reasons."""
    dcfg = self._drift
    sr = float(dcfg.get("suspect_ratio", 0.55))
    reasons: List[str] = []

    max_xy = float(dcfg["max_xy_jump_m"])
    max_z = float(dcfg["max_z_jump_m"])
    max_rot = float(dcfg["max_rot_deg"])
    max_rate = float(dcfg["max_rpy_rate_deg_s"])
    max_zres = float(dcfg["max_z_depth_residual_m"])
    min_valid = float(dcfg["min_depth_valid_ratio"])

    hard_pose = (
      (m["jump_xy_m"] > max_xy)
      or (m["jump_z_m"] > max_z)
      or (m["rot_deg"] > max_rot)
      or (m["rpy_rate_deg_s"] > max_rate)
    )
    if m["jump_xy_m"] > max_xy:
      reasons.append("jump_xy")
    if m["jump_z_m"] > max_z:
      reasons.append("jump_z")
    if m["rot_deg"] > max_rot:
      reasons.append("rot_deg")
    if m["rpy_rate_deg_s"] > max_rate:
      reasons.append("rpy_rate")

    has_mask = mask_u8 is not None and np.any(np.asarray(mask_u8) > 0)
    depth_hard = False
    if has_mask:
      if m["depth_valid_ratio"] < min_valid:
        depth_hard = True
        reasons.append("depth_valid_low")
      if m["z_depth_residual_m"] > max_zres:
        depth_hard = True
        reasons.append("z_depth_residual")

    if hard_pose:
      return "bad", "pose_hard", reasons or ["pose_hard"]

    if depth_hard:
      return "bad", "depth_hard", reasons

    sx = max_xy * sr
    sz = max_z * sr
    srot = max_rot * sr
    srate = max_rate * sr
    szres = max_zres * sr
    svalid = min_valid + (1.0 - min_valid) * (1.0 - sr)

    suspect = (
      (m["jump_xy_m"] > sx)
      or (m["jump_z_m"] > sz)
      or (m["rot_deg"] > srot)
      or (m["rpy_rate_deg_s"] > srate)
    )
    if has_mask:
      suspect = suspect or (m["z_depth_residual_m"] > szres) or (m["depth_valid_ratio"] < svalid)

    if suspect:
      tag = "suspect_" + (reasons[0] if reasons else "soft")
      return "suspect", tag, reasons

    return "good", "ok", []

  def track(
    self,
    K: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    timestamp_ms: float,
    mask_u8: Optional[np.ndarray] = None,
  ) -> TrackResult:
    if not self.initialized:
      return TrackResult(
        pose=np.eye(4, dtype=np.float32),
        state="waiting_init",
        reason="not_initialized",
        timestamp_ms=timestamp_ms,
        quality_state="na",
      )

    pose = self.estimator.track_one(
      rgb=rgb,
      depth=depth,
      K=K,
      iteration=self.track_refine_iter,
      fast_depth_filter=self.fast_depth_filter,
    )
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

    if not bool(self._drift.get("enabled", True)):
      stat = self._legacy_jump_gate(pose)
      m = self._compute_drift_metrics(pose, depth, mask_u8, timestamp_ms)
      self._last_ts_ms = float(timestamp_ms)
      if stat["ok"] < 0.5:
        self.initialized = False
        self.last_pose = None
        self.reset_drift_state()
        return TrackResult(
          pose=pose,
          state="reinit_required",
          reason=f"pose_jump jump={stat['jump']:.3f} rot={stat['rot_deg']:.1f}",
          timestamp_ms=timestamp_ms,
          quality_state="bad",
          jump_xy_m=m["jump_xy_m"],
          jump_z_m=m["jump_z_m"],
          rot_deg=m["rot_deg"],
          rpy_rate_deg_s=m["rpy_rate_deg_s"],
          depth_median_m=m["depth_median_m"],
          depth_valid_ratio=m["depth_valid_ratio"],
          z_depth_residual_m=m["z_depth_residual_m"],
          drift_reason="legacy_gate",
        )
      self.last_pose = pose.copy()
      return TrackResult(
        pose=pose,
        state="tracking",
        reason="track_ok",
        timestamp_ms=timestamp_ms,
        quality_state="good",
        jump_xy_m=m["jump_xy_m"],
        jump_z_m=m["jump_z_m"],
        rot_deg=m["rot_deg"],
        rpy_rate_deg_s=m["rpy_rate_deg_s"],
        depth_median_m=m["depth_median_m"],
        depth_valid_ratio=m["depth_valid_ratio"],
        z_depth_residual_m=m["z_depth_residual_m"],
        drift_reason="",
      )

    m = self._compute_drift_metrics(pose, depth, mask_u8, timestamp_ms)
    quality, drift_tag, _reasons = self._classify_drift(m, mask_u8)

    depth_bad_frame = bool(
      mask_u8 is not None
      and np.any(np.asarray(mask_u8) > 0)
      and (
        (m["depth_valid_ratio"] < float(self._drift["min_depth_valid_ratio"]))
        or (m["z_depth_residual_m"] > float(self._drift["max_z_depth_residual_m"]))
      )
    )

    hard_pose_fail = (
      (m["jump_xy_m"] > float(self._drift["max_xy_jump_m"]))
      or (m["jump_z_m"] > float(self._drift["max_z_jump_m"]))
      or (m["rot_deg"] > float(self._drift["max_rot_deg"]))
      or (m["rpy_rate_deg_s"] > float(self._drift["max_rpy_rate_deg_s"]))
    )

    n_bad = int(self._drift.get("bad_frames_for_reinit", 5))

    out_pose = pose.copy()
    drift_reason = drift_tag

    if hard_pose_fail:
      self._consecutive_depth_bad = 0
      self.initialized = False
      self.last_pose = None
      self.reset_drift_state()
      self._last_ts_ms = float(timestamp_ms)
      return TrackResult(
        pose=out_pose,
        state="reinit_required",
        reason=f"pose_drift {drift_tag} xy={m['jump_xy_m']:.3f} z={m['jump_z_m']:.3f} rot={m['rot_deg']:.1f} rate={m['rpy_rate_deg_s']:.1f}",
        timestamp_ms=timestamp_ms,
        quality_state="bad",
        jump_xy_m=m["jump_xy_m"],
        jump_z_m=m["jump_z_m"],
        rot_deg=m["rot_deg"],
        rpy_rate_deg_s=m["rpy_rate_deg_s"],
        depth_median_m=m["depth_median_m"],
        depth_valid_ratio=m["depth_valid_ratio"],
        z_depth_residual_m=m["z_depth_residual_m"],
        drift_reason=drift_reason,
        consecutive_depth_bad_frames=self._consecutive_depth_bad,
      )

    if depth_bad_frame:
      self._consecutive_depth_bad += 1
    else:
      self._consecutive_depth_bad = 0

    if self._consecutive_depth_bad >= n_bad:
      self.initialized = False
      self.last_pose = None
      self.reset_drift_state()
      self._last_ts_ms = float(timestamp_ms)
      return TrackResult(
        pose=out_pose,
        state="reinit_required",
        reason=f"depth_drift_consecutive {self._consecutive_depth_bad}/{n_bad} zres={m['z_depth_residual_m']:.3f} valid={m['depth_valid_ratio']:.2f}",
        timestamp_ms=timestamp_ms,
        quality_state="bad",
        jump_xy_m=m["jump_xy_m"],
        jump_z_m=m["jump_z_m"],
        rot_deg=m["rot_deg"],
        rpy_rate_deg_s=m["rpy_rate_deg_s"],
        depth_median_m=m["depth_median_m"],
        depth_valid_ratio=m["depth_valid_ratio"],
        z_depth_residual_m=m["z_depth_residual_m"],
        drift_reason="depth_consecutive",
        consecutive_depth_bad_frames=self._consecutive_depth_bad,
      )

    z_blend = float(self._drift.get("z_median_blend", 0.42))
    rot_slerp = float(self._drift.get("rotation_slerp", 0.5))
    z_blend_depth = float(self._drift.get("z_median_blend_depth_recovery", min(0.85, z_blend + 0.25)))

    soft_depth_recovery = bool(
      depth_bad_frame and (quality == "bad") and (self._consecutive_depth_bad < n_bad) and (not hard_pose_fail)
    )
    do_soft = (quality == "suspect") or soft_depth_recovery
    zb = z_blend_depth if soft_depth_recovery else z_blend

    if do_soft:
      med = m["depth_median_m"]
      out_pose, r_sm = apply_z_blend_and_rotation_slerp(
        out_pose,
        med,
        self._R_smooth,
        zb,
        rot_slerp,
      )
      self._R_smooth = r_sm
      drift_reason = drift_tag if quality == "suspect" else "depth_soft_recovery"
      if soft_depth_recovery:
        quality = "suspect"
    else:
      if self._R_smooth is None:
        self._R_smooth = out_pose[:3, :3].copy()
      else:
        self._R_smooth = out_pose[:3, :3].copy()

    self.last_pose = out_pose.copy()
    self._last_ts_ms = float(timestamp_ms)

    sync_estimator_pose = torch.from_numpy(out_pose).to(
      dtype=self.estimator.pose_last.dtype,
      device=self.estimator.pose_last.device,
    )
    self.estimator.pose_last = sync_estimator_pose

    return TrackResult(
      pose=out_pose,
      state="tracking",
      reason="track_ok",
      timestamp_ms=timestamp_ms,
      quality_state=quality,
      jump_xy_m=m["jump_xy_m"],
      jump_z_m=m["jump_z_m"],
      rot_deg=m["rot_deg"],
      rpy_rate_deg_s=m["rpy_rate_deg_s"],
      depth_median_m=m["depth_median_m"],
      depth_valid_ratio=m["depth_valid_ratio"],
      z_depth_residual_m=m["z_depth_residual_m"],
      drift_reason=drift_reason if quality == "suspect" else "",
      consecutive_depth_bad_frames=self._consecutive_depth_bad,
    )
