from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

from demo.fpp.kalman_filter_6d import KalmanFilter6D
from demo.fpp.pose_utils import (
  adjust_pose_to_image_point,
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


def _rotation_angle_deg(R_prev: np.ndarray, R_cur: np.ndarray) -> float:
  dR = R_prev.T @ R_cur
  trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(trace)))


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
    return TrackResult(pose=pose, state="tracking", reason="register_ok", timestamp_ms=timestamp_ms)

  def _is_pose_jump_too_large(self, pose: np.ndarray) -> Dict[str, float]:
    if self.last_pose is None:
      return {"jump": 0.0, "rot_deg": 0.0, "ok": 1.0}
    jump = np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3])
    rot_deg = _rotation_angle_deg(self.last_pose[:3, :3], pose[:3, :3])
    ok = float((jump <= self.max_translation_jump_m) and (rot_deg <= self.max_rotation_jump_deg))
    return {"jump": float(jump), "rot_deg": float(rot_deg), "ok": ok}

  def track(self, K: np.ndarray, rgb: np.ndarray, depth: np.ndarray, timestamp_ms: float) -> TrackResult:
    if not self.initialized:
      return TrackResult(
        pose=np.eye(4, dtype=np.float32),
        state="waiting_init",
        reason="not_initialized",
        timestamp_ms=timestamp_ms,
      )
    pose = self.estimator.track_one(
      rgb=rgb,
      depth=depth,
      K=K,
      iteration=self.track_refine_iter,
      fast_depth_filter=self.fast_depth_filter,
    )
    stat = self._is_pose_jump_too_large(pose)
    if stat["ok"] < 0.5:
      self.initialized = False
      self.last_pose = None
      return TrackResult(
        pose=pose,
        state="reinit_required",
        reason=f"pose_jump jump={stat['jump']:.3f} rot={stat['rot_deg']:.1f}",
        timestamp_ms=timestamp_ms,
      )
    self.last_pose = pose.copy()
    return TrackResult(pose=pose, state="tracking", reason="track_ok", timestamp_ms=timestamp_ms)
